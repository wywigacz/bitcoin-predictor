"""Wirtualny, aktywnie zarządzany portfel (edukacyjny eksperyment, bez prawdziwych pieniędzy i transakcji).

Zasady (ustalone z góry 2026-10-29, jawne i niezmieniane po fakcie):
  - wpłaty: 10 000 zł na start i 10 000 zł pierwszego dnia każdego miesiąca,
  - codziennie (po zamknięciu dnia) program wybiera, jaką część portfela trzymać w BTC, a jaką w gotówce (PLN),
  - trzy strategie proponują udział BTC:
      regularne    - zawsze 100% (jak zwykłe regularne kupowanie),
      ryzyko       - kontrola zmienności: min(100%, normalna zmienność / prognozowana zmienność) (Moreira i Muir 2017),
      trend        - 100%, gdy cena wzrosła w ostatnich 28 dniach, inaczej 25% (potwierdzona hipoteza H7),
  - uczenie się: wagi strategii według algorytmu Hedge (wagi wykładnicze): waga ~ exp(ETA × log-zysk strategii z ostatnich 365 dni),
    najmniej 5% na strategię; uczy się tylko na przeszłości,
  - handel tylko, gdy cel różni się od obecnego udziału o więcej niż 5 punktów procentowych; prowizja 0,2% od obrotu,
  - kupno/sprzedaż po cenie zamknięcia dnia; decyzja używa wyłącznie danych znanych na koniec tego dnia.
Porównanie: zwykłe regularne kupowanie (każda wpłata od razu w BTC, bez sprzedaży, ta sama prowizja).
Podatki pominięte (w Polsce zysk z kryptowalut podlega 19% podatkowi, PIT-38)."""
import json

import numpy as np
import pandas as pd

import config as C

DEPOSIT = 10000.0
FEE = 0.002
BAND = 0.05
ETA = 2.0
WINDOW = 365
FLOOR = 0.05
LIVE_START = pd.Timestamp("2026-10-28")
BACKTEST_START = pd.Timestamp("2018-01-01")
EXAM_START = pd.Timestamp("2024-10-29")              # egzamin na ostatnich 2 latach
LIVE_LOG = C.DATA_DIR / "sim_live.csv"
STRATS = ("regularne", "ryzyko", "trend")


def inputs(df, sig):
    import world
    w = pd.read_csv(world.WORLD_CSV, index_col=0, parse_dates=True)["y_pln"]
    pln = w.reindex(df.index.union(w.index)).sort_index().ffill(limit=7).reindex(df.index)
    X = pd.DataFrame(index=df.index)
    X["px"] = df["close"] * pln                                  # cena BTC w złotych
    X["usd"] = df["close"]
    X["pln"] = pln
    X["sig"] = sig
    X["sig_norm"] = pd.Series(sig, df.index).loc["2013":].expanding(365).median().reindex(df.index)
    X["mom28"] = np.log(df["close"] / df["close"].shift(28))
    X = X.dropna(subset=["px", "sig"])
    r = X["px"].pct_change().fillna(0)
    T = pd.DataFrame(index=X.index)
    T["regularne"] = 1.0
    T["ryzyko"] = (X["sig_norm"] / X["sig"]).clip(upper=1.0).fillna(1.0)
    T["trend"] = np.where(X["mom28"] > 0, 1.0, 0.25)
    # wyniki strategii: dzienny zwrot = udział BTC z wczoraj × zwrot BTC dziś (to, co każda zarobiłaby osobno)
    G = np.log1p(T.shift(1).fillna(1.0).mul(r, axis=0))
    S = G.rolling(WINDOW, min_periods=60).sum().fillna(0)
    E = np.exp(ETA * (S - S.max(axis=1).to_numpy()[:, None]))
    P = E.div(E.sum(axis=1), axis=0)
    P = (P.clip(lower=FLOOR)).div(P.clip(lower=FLOOR).sum(axis=1), axis=0)
    X["target"] = (P * T).sum(axis=1)
    return X, T, P, S


def _why(d, x, t, p, w_now, target, action, amt_pln):
    pc = lambda v: f"{v * 100:.0f}%"
    s_y, n_y = x["sig"] * np.sqrt(365) * 100, x["sig_norm"] * np.sqrt(365) * 100
    risk = (f"prognozowane wahania {s_y:.0f}% rocznie wobec normy {n_y:.0f}%, więc „ryzyko” chce {pc(t['ryzyko'])} w BTC"
            if np.isfinite(n_y) else f"„ryzyko” chce {pc(t['ryzyko'])} w BTC")
    trend = f"cena w 28 dni {x['mom28'] * 100:+.1f}%, więc „trend” chce {pc(t['trend'])}".replace(".", ",")
    best = max(STRATS, key=lambda k: p[k])
    learn = (f"wagi wyuczone na ostatnim roku: regularne {pc(p['regularne'])}, ryzyko {pc(p['ryzyko'])}, trend {pc(p['trend'])} "
             f"(najlepiej radzi sobie „{best}”)")
    head = f"Cel {pc(target)} w BTC, było {pc(w_now)}. "
    kw = f"{amt_pln:,.0f}".replace(",", " ")
    act = {"KUP": f"Kupuję BTC za {kw} zł.", "SPRZEDAJ": f"Sprzedaję BTC za {kw} zł.",
           "TRZYMAJ": f"Różnica mniejsza niż {BAND * 100:.0f} pkt proc., więc nic nie robię (oszczędzam prowizję)."}[action]
    return head + act + " Dlaczego: " + risk + "; " + trend + "; " + learn + "."


def simulate(X, T, P, start, end=None, log=True, state=None):
    """Silnik: dzień po dniu, od start do end. state pozwala kontynuować zapisany portfel."""
    D = X.loc[start:end] if end is not None else X.loc[start:]
    cash, btc, dep = (state or {}).get("cash", 0.0), (state or {}).get("btc", 0.0), (state or {}).get("deposited", 0.0)
    last_month = (state or {}).get("last_month")
    first = state is None
    dca_btc, dca_dep = (state or {}).get("dca_btc", 0.0), dep
    rows = []
    for d, x in D.iterrows():
        px = float(x["px"])
        deposit = 0.0
        ym = d.strftime("%Y-%m")
        if first or ym != last_month:
            deposit = DEPOSIT
            cash += deposit
            dep += deposit
            dca_btc += deposit * (1 - FEE) / px
            dca_dep += deposit
            last_month, first = ym, False
        value = cash + btc * px
        w_now = btc * px / value if value > 0 else 0.0
        target = float(x["target"])
        action, amt = "TRZYMAJ", 0.0
        if abs(target - w_now) > BAND:
            amt = abs(target - w_now) * value
            if target > w_now:
                amt = min(amt, cash)
                btc += amt * (1 - FEE) / px
                cash -= amt
                action = "KUP"
            else:
                q = amt / px
                btc -= q
                cash += amt * (1 - FEE)
                action = "SPRZEDAJ"
        value = cash + btc * px
        row = {"date": d.strftime("%Y-%m-%d"), "price_pln": round(px, 2), "deposit": deposit, "action": action, "amount_pln": round(amt, 2),
               "target": round(target, 4), "w_before": round(w_now, 4), "btc": btc, "cash": round(cash, 2), "value": round(value, 2),
               "deposited": dep, "dca_value": round(dca_btc * px, 2), "dca_btc": dca_btc,
               **{f"t_{k}": round(float(T.at[d, k]), 4) for k in STRATS}, **{f"p_{k}": round(float(P.at[d, k]), 4) for k in STRATS}}
        if log:
            row["why"] = _why(d, x, T.loc[d], P.loc[d], w_now, target, action, amt)
        row["last_month"] = last_month
        rows.append(row)
    return pd.DataFrame(rows)


def _twr(v, dep):
    """Zwrot ważony czasem (bez wpływu wpłat): dzienny zwrot = (wartość dziś − wpłata dziś) / wartość wczoraj."""
    g = np.ones(len(v))
    g[1:] = (v[1:] - dep[1:]) / v[:-1]
    return np.cumprod(g)


def _stats(R, col="value"):
    idx = _twr(R[col].to_numpy(float), R["deposit"].to_numpy(float))
    yrs = max(len(R) / 365, 1e-9)
    dd = float((idx / np.maximum.accumulate(idx) - 1).min())
    return {"final": float(R[col].iloc[-1]), "deposited": float(R["deposited"].iloc[-1]),
            "gain_pct": float(R[col].iloc[-1] / R["deposited"].iloc[-1] - 1),
            "twr_ann": float(idx[-1] ** (1 / yrs) - 1), "max_dd": dd}


def run(df, sig):
    X, T, P, S = inputs(df, sig)
    done = X.index[X.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]   # tylko zamknięte dni
    X, T, P = X.loc[done], T.loc[done], P.loc[done]
    # test na historii (liczony od nowa przy każdej aktualizacji, z tymi samymi zasadami)
    B = simulate(X, T, P, BACKTEST_START, log=False)
    bt = {"active": _stats(B), "dca": _stats(B, "dca_value"),
          "trades": int((B["action"] != "TRZYMAJ").sum()), "fees_est": float(B["amount_pln"].sum() * FEE),
          "avg_btc": float(B["w_before"].mean()), "start": str(BACKTEST_START.date()),
          "curve": {"d": B["date"].iloc[::14].tolist() + [B["date"].iloc[-1]],
                    "a": B["value"].iloc[::14].round(0).tolist() + [float(B["value"].iloc[-1])],
                    "b": B["dca_value"].iloc[::14].round(0).tolist() + [float(B["dca_value"].iloc[-1])],
                    "dep": B["deposited"].iloc[::14].tolist() + [float(B["deposited"].iloc[-1])]}}
    # egzamin: start z pierwszą wpłatą dokładnie 2 lata temu, te same zasady
    Ex = simulate(X, T, P, EXAM_START, log=False)
    bt["exam"] = {"start": str(EXAM_START.date()), "active": _stats(Ex), "dca": _stats(Ex, "dca_value"),
                  "trades": int((Ex["action"] != "TRZYMAJ").sum())}
    # portfel na żywo: dziennik tylko dopisywany (przeszłych decyzji nigdy nie przeliczamy)
    old = pd.read_csv(LIVE_LOG) if LIVE_LOG.exists() else pd.DataFrame()
    if len(old):
        last = old.iloc[-1]
        state = {"cash": float(last["cash"]), "btc": float(last["btc"]), "deposited": float(last["deposited"]),
                 "last_month": str(last["last_month"]), "dca_btc": float(last["dca_btc"])}
        nxt = pd.Timestamp(last["date"]) + pd.Timedelta(days=1)
        new = simulate(X, T, P, nxt, state=state) if nxt <= X.index[-1] else pd.DataFrame()
    else:
        new = simulate(X, T, P, LIVE_START)
    live = pd.concat([old, new], ignore_index=True) if len(new) else old
    if len(live):
        C.DATA_DIR.mkdir(exist_ok=True)
        live.to_csv(LIVE_LOG, index=False)
    out = {"backtest": bt, "live_n": int(len(live)), "rules": {"deposit": DEPOSIT, "fee": FEE, "band": BAND, "eta": ETA, "window": WINDOW}}
    if len(live):
        L = live.iloc[-1]
        out["live"] = {"date": L["date"], "value": float(L["value"]), "deposited": float(L["deposited"]), "cash": float(L["cash"]),
                       "btc": float(L["btc"]), "dca_value": float(L["dca_value"]), "target": float(L["target"]),
                       "w": float(L["btc"] * L["price_pln"] / L["value"]) if L["value"] > 0 else 0.0,
                       "p": {k: float(L[f"p_{k}"]) for k in STRATS}, "t": {k: float(L[f"t_{k}"]) for k in STRATS},
                       "log": live.tail(20).iloc[::-1][["date", "action", "amount_pln", "price_pln", "target", "w_before", "value", "deposit", "why"]].to_dict("records")}
    return out
