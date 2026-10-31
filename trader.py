"""Wirtualny trader (edukacyjny eksperyment, bez prawdziwych pieniędzy i transakcji), od 2026-10-31.

Przejmuje portfel na żywo z sim.py (ten sam stan gotówki i BTC na 30.10.2026) i od tej chwili decyduje przy każdej aktualizacji
w chmurze, czyli co ok. 6 godzin, po bieżącej cenie z Bitstamp (kurs złotego z NBP). Zasady (spisane 2026-10-31, jawne):
  - wpłaty: 10 000 zł przy pierwszej decyzji w każdym miesiącu,
  - cel: jak najwięcej złotych na koniec, czyli maksymalny wzrost wartości portfela w długim okresie,
  - udział BTC wyznaczają te same trzy strategie co w sim.py, ważone algorytmem Hedge (strategia, która w ostatnim roku
    pomnożyłaby pieniądze najbardziej, dostaje największy głos); sygnał trendu liczony z bieżącej ceny, nie z wczorajszego zamknięcia,
  - handel tylko, gdy cel różni się od obecnego udziału o więcej niż 5 pkt proc. (prowizja 0,2% od obrotu zjadałaby zysk z drobnych ruchów),
  - każdą decyzję zapisujemy z uzasadnieniem i z warunkiem, przy którym trader zmieni zdanie; dziennik jest tylko dopisywany,
  - każdą transakcję rozliczamy po 1 i 7 dniach: o ile złotych lepiej lub gorzej wyszła niż pozostawienie portfela bez zmian.
Decyzje zapisuje tylko przebieg w chmurze (GITHUB_ACTIONS) albo uruchomienie z BTC_TRADER=1; zwykły przebieg lokalny tylko czyta dziennik."""
import os

import numpy as np
import pandas as pd
import requests

import config as C
import sim as SI

LOG = C.DATA_DIR / "trader_log.csv"
START = pd.Timestamp("2026-10-31")
MIN_GAP_H = 3                                        # dwie decyzje nie częściej niż co 3 godziny
TZ = "Europe/Warsaw"


def _live_price():
    try:
        r = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=20).json()
        return float(r["last"]), float(r["open"]), "Bitstamp"
    except Exception:
        return None, None, None


def _usdpln(fallback):
    try:
        r = requests.get("https://api.nbp.pl/api/exchangerates/rates/a/usd/?format=json", timeout=20).json()
        return float(r["rates"][0]["mid"]), "NBP"
    except Exception:
        return float(fallback), "ostatni znany kurs"


def _zl(v):
    return f"{v:,.0f}".replace(",", " ") + " zł"


def _pc(v):
    return f"{v * 100:.0f}%"


def _start_state():
    """Stan portfela przejęty z dziennika sim.py (ostatni dzień przed startem tradera)."""
    s = pd.read_csv(SI.LIVE_LOG)
    s = s[pd.to_datetime(s["date"]) < START]
    last = s.iloc[-1]
    return {"cash": float(last["cash"]), "btc": float(last["btc"]), "deposited": float(last["deposited"]),
            "last_month": str(last["last_month"]), "dca_btc": float(last["dca_btc"])}


def _explain(ctx, t, p, w_now, target, action, amt, deposit):
    """Uzasadnienie w trzech częściach: co widzę, co robię i dlaczego, kiedy zmienię zdanie."""
    best = max(SI.STRATS, key=lambda k: p[k])
    chg = ctx["chg24"]
    dec = lambda v, f: format(v, f).replace(".", ",")
    usd = format(ctx["usd"], ",.0f").replace(",", " ")
    seen = (f"Bitcoin kosztuje {usd} USD, czyli {_zl(ctx['px'])}"
            + (f"; w ciągu doby {dec(chg * 100, '+.1f')}%" if chg is not None else "")
            + f". W ciągu 28 dni cena {'wzrosła' if ctx['mom28'] > 0 else 'spadła'} o {dec(abs(np.expm1(ctx['mom28'])) * 100, '.1f')}%"
            + f", więc trend jest {'wzrostowy' if ctx['mom28'] > 0 else 'spadkowy'}. Prognozowane wahania: {ctx['vol_y']:.0f}% rocznie wobec typowych {ctx['vol_n']:.0f}%"
            + (" (spokojniej niż zwykle)." if ctx["vol_y"] < ctx["vol_n"] else " (burzliwiej niż zwykle).")
            + (f" Szansa burzliwego tygodnia: {_pc(ctx['storm'])}." if ctx.get("storm") is not None else ""))
    why = (f"Trzy strategie chcą: regularne {_pc(t['regularne'])}, ryzyko {_pc(t['ryzyko'])}, trend {_pc(t['trend'])} w BTC. "
           f"Najwięcej zarobiłaby w ostatnim roku „{best}”, więc ma największy głos ({_pc(p[best])}). Średnia ważona daje cel {_pc(target)} w BTC; teraz mam {_pc(w_now)}.")
    dep = f"Wpłynęło {_zl(deposit)} miesięcznej wpłaty. " if deposit else ""
    act = {"KUP": f"Kupuję BTC za {_zl(amt)}.",
           "SPRZEDAJ": f"Sprzedaję BTC za {_zl(amt)}.",
           "TRZYMAJ": f"Nic nie robię: różnica to tylko {abs(target - w_now) * 100:.0f} pkt proc., a prowizja zjadłaby zysk z takiej korekty."}[action]
    flip = ctx["px28"]
    if ctx["mom28"] > 0:
        cond = (f"Zmienię zdanie, jeśli cena spadnie poniżej {_zl(flip)} (poziom sprzed 28 dni): trend się odwróci, "
                f"a strategia trendu zejdzie ze 100% do 25% w BTC, więc część sprzedam.")
    else:
        cond = (f"Zmienię zdanie, jeśli cena wzrośnie powyżej {_zl(flip)} (poziom sprzed 28 dni): trend się odwróci "
                f"i strategia trendu podniesie udział BTC do 100%, więc dokupię.")
    return {"seen": seen, "do": dep + act + " " + why, "change": cond}


def _score(log, px_now, now):
    """Rozliczenie transakcji: wynik po 1 i 7 dniach względem pozostawienia portfela bez zmian (bez prowizji = zysk z decyzji minus koszt)."""
    out = []
    tr = log[log["action"] != "TRZYMAJ"]
    for _, r in tr.iterrows():
        t0 = pd.Timestamp(r["ts"])
        dq = float(r["btc_after"]) - float(r["btc_before"])            # ile BTC przybyło (+) lub ubyło (−)
        res = {"ts": r["ts"], "action": r["action"], "amount_pln": float(r["amount_pln"])}
        for h, lab in ((1, "d1"), (7, "d7")):
            later = log[pd.to_datetime(log["ts"]) >= t0 + pd.Timedelta(days=h)]
            p1 = float(later.iloc[0]["price_pln"]) if len(later) else (px_now if now >= t0 + pd.Timedelta(days=h) else None)
            res[lab] = None if p1 is None else dq * (p1 - float(r["price_pln"])) - float(r["amount_pln"]) * SI.FEE
        out.append(res)
    return out


def run(df, sig, VE=None):
    X, T, P, _ = SI.inputs(df, SI.causal_sigma(df))
    last = X.index[-1]
    x, t, p = X.loc[last], T.loc[last].to_dict(), P.loc[last].to_dict()
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    may_write = bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("BTC_TRADER"))
    due = len(old) == 0 or now - pd.Timestamp(old.iloc[-1]["ts"]) >= pd.Timedelta(hours=MIN_GAP_H)
    usd, open24, src = _live_price()
    if usd is None:
        usd, open24, src = float(df["close"].iloc[-1]), float(df["close"].iloc[-2]), "zamknięcie dnia"
    fx, fx_src = _usdpln(x["pln"])
    px = usd * fx
    c28 = float(df["close"].iloc[-28])                               # zamknięcie sprzed 28 dni (jak w strategii trendu)
    mom28 = float(np.log(usd / c28))
    t_live = dict(t, trend=1.0 if mom28 > 0 else 0.25)
    target = float(sum(p[k] * t_live[k] for k in SI.STRATS))
    storm = None
    try:
        storm = float(VE["burza7"]["p_now"]) if VE and VE.get("burza7") else None
    except Exception:
        pass
    ctx = {"usd": usd, "px": px, "chg24": usd / open24 - 1 if open24 else None, "mom28": mom28, "px28": c28 * fx,
           "vol_y": float(x["sig"] * np.sqrt(365) * 100), "vol_n": float(x["sig_norm"] * np.sqrt(365) * 100), "storm": storm}
    new = None
    if may_write and due:
        st = (dict(old.iloc[-1][["cash", "btc", "deposited", "last_month", "dca_btc"]]) if len(old) else _start_state())
        cash, btc, dep, dca_btc = float(st["cash"]), float(st["btc"]), float(st["deposited"]), float(st["dca_btc"])
        ym, deposit = now.strftime("%Y-%m"), 0.0
        if ym != str(st["last_month"]):
            deposit = SI.DEPOSIT
            cash += deposit
            dep += deposit
            dca_btc += deposit * (1 - SI.FEE) / px
        value = cash + btc * px
        w_now = btc * px / value if value > 0 else 0.0
        btc0, action, amt = btc, "TRZYMAJ", 0.0
        if abs(target - w_now) > SI.BAND:
            amt = abs(target - w_now) * value
            if target > w_now:
                amt = min(amt, cash)
                btc += amt * (1 - SI.FEE) / px
                cash -= amt
                action = "KUP"
            else:
                btc -= amt / px
                cash += amt * (1 - SI.FEE)
                action = "SPRZEDAJ"
        ex = _explain(ctx, t_live, p, w_now, target, action, amt, deposit)
        new = {"ts": now.strftime("%Y-%m-%d %H:%M"), "local": now.tz_localize("UTC").tz_convert(TZ).strftime("%Y-%m-%d %H:%M"),
               "price_usd": round(usd, 2), "usdpln": round(fx, 4), "price_pln": round(px, 2), "src": f"{src}, kurs {fx_src}",
               "deposit": deposit, "action": action, "amount_pln": round(amt, 2), "target": round(target, 4), "w_before": round(w_now, 4),
               "btc_before": btc0, "btc_after": btc, "btc": btc, "cash": round(cash, 2), "value": round(cash + btc * px, 2),
               "deposited": dep, "dca_btc": dca_btc, "dca_value": round(dca_btc * px, 2), "last_month": ym,
               **{f"t_{k}": round(t_live[k], 4) for k in SI.STRATS}, **{f"p_{k}": round(p[k], 4) for k in SI.STRATS},
               "seen": ex["seen"], "do": ex["do"], "change": ex["change"]}
        old = pd.concat([old, pd.DataFrame([new])], ignore_index=True)
        C.DATA_DIR.mkdir(exist_ok=True)
        old.to_csv(LOG, index=False)
    if not len(old):
        return {"ctx": ctx, "target": target, "log": [], "score": []}
    L = old.iloc[-1]
    val_now = float(L["cash"]) + float(L["btc"]) * px
    tr = old[old["action"] != "TRZYMAJ"]
    return {"ctx": ctx, "target": target, "n": int(len(old)), "n_trades": int(len(tr)), "since": str(old.iloc[0]["local"]),
            "value": val_now, "deposited": float(L["deposited"]), "cash": float(L["cash"]), "btc": float(L["btc"]),
            "w": float(L["btc"]) * px / val_now if val_now > 0 else 0.0, "dca_value": float(L["dca_btc"]) * px,
            "fees": float(tr["amount_pln"].sum() * SI.FEE), "wrote": new is not None,
            "log": old.iloc[::-1].to_dict("records"), "score": _score(old, px, now)}
