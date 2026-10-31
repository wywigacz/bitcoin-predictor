"""Test potwierdzający dla ruchów dużych graczy: kilka hipotez spisanych Z GÓRY (2026-10-29), zanim ktokolwiek spojrzał na wynik.

Dlaczego osobno od skanera: w skanerze jest ~8500 testów, więc prawdziwy, ale słaby efekt utonąłby w korekcie na liczbę prób.
Tu jest tylko 6 prób, każda z kierunkiem przewidzianym z góry, więc próg jest łagodniejszy, a wynik nadal uczciwy:
  1. zwrot BTC liczony od zamknięcia dnia PO dniu, w którym dane były publicznie znane (bez podglądania przyszłości),
  2. korelacja rang (Spearman), liczba obserwacji pomniejszona o nakładanie się zwrotów (n / horyzont),
  3. test jednostronny w przewidzianym kierunku, poprawka Bonferroniego: p < 0,05 / 6,
  4. ten sam kierunek w obu połowach próby.
Hipoteza, która przejdzie wszystko, trafia do zespołu modeli jako dodatkowa cecha. Hipotez i progów nie zmieniamy po fakcie."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as tdist

import config as C
import whales as WH

ALPHA = 0.05
MIN_DAYS = 250
HYP = [
    {"id": "H1", "col": "etf3", "h": 7, "sign": +1, "text": "Duży napływ pieniędzy do funduszy ETF (suma 3 dni sesyjnych) → wyższy zwrot BTC w kolejnym tygodniu"},
    {"id": "H2", "col": "etfstreak", "h": 7, "sign": +1, "text": "Długa seria dni z napływami do ETF (a nie odpływami) → wyższy zwrot w kolejnym tygodniu"},
    {"id": "H3", "col": "mstr30", "h": 30, "sign": +1, "text": "Duże zakupy firmy Strategy w ostatnich 30 dniach → wyższy zwrot w kolejnych 30 dniach"},
    {"id": "H4", "col": "hl_smart", "h": 7, "sign": +1, "text": "Najlepsi traderzy Hyperliquid ustawieni na wzrost → wyższy zwrot w kolejnym tygodniu"},
    {"id": "H5", "col": "hl_spread", "h": 7, "sign": +1, "text": "Najlepsi traderzy bardziej na wzrost niż najgorsi → wyższy zwrot w kolejnym tygodniu"},
    {"id": "H6", "col": "hl_fund7", "h": 7, "sign": -1, "text": "Wysoka opłata za zakłady na wzrost na Hyperliquid (tłok) → niższy zwrot w kolejnym tygodniu"},
]
for _h in HYP:
    _h["batch"] = 1
# Partia 2 (spisana 2026-10-29, przed obejrzeniem wyników): efekty najlepiej udokumentowane w badaniach nad Bitcoinem
# (Liu i Tsyvinski 2021: trend 1–4 tygodni i uwaga inwestorów) oraz połączenie słabych sygnałów w jeden.
HYP += [
    {"id": "H7", "col": "mom28", "h": 7, "sign": +1, "batch": 2, "text": "Wzrost ceny w ostatnich 4 tygodniach → wyższy zwrot w kolejnym tygodniu (trend)"},
    {"id": "H8", "col": "mom7", "h": 7, "sign": +1, "batch": 2, "text": "Wzrost ceny w ostatnim tygodniu → wyższy zwrot w kolejnym tygodniu (krótki trend)"},
    {"id": "H9", "col": "attn", "h": 7, "sign": +1, "batch": 2, "text": "Rosnące zainteresowanie Bitcoinem na Wikipedii → wyższy zwrot w kolejnym tygodniu"},
    {"id": "H10", "col": "combo", "h": 7, "sign": +1, "batch": 2, "text": "Połączenie słabych sygnałów (trend, uwaga, napływ do ETF) → wyższy zwrot w kolejnym tygodniu"},
]
# Partia 3 (spisana 2026-10-29, po raportach Deep Research, przed obejrzeniem wyników): stablecoiny i rynek opcji.
HYP += [
    {"id": "H11", "col": "peg7", "h": 7, "sign": -1, "batch": 3, "text": "USDT droższy niż 1 dolar (ucieczka do „cyfrowej gotówki”) → niższy zwrot BTC w kolejnym tygodniu"},
    {"id": "H12", "col": "vrp", "h": 30, "sign": +1, "batch": 3, "text": "Opcje drogie względem faktycznych wahań (premia za ryzyko zmienności) → wyższy zwrot w kolejnych 30 dniach"},
    {"id": "H13", "col": "rr25", "h": 1, "sign": +1, "batch": 3, "text": "Droższe opcje na wzrost niż na spadek (risk reversal 25-delta) → wyższy zwrot następnego dnia"},
]


def _z(s):
    return ((s - s.expanding(365).mean()) / s.expanding(365).std()).clip(-4, 4)


def features(idx, df=None):
    """Cechy dzienne wyrównane do indeksu BTC i przesunięte o 1 dzień (wartość z dnia D używana od dnia D+1)."""
    F = pd.DataFrame(index=idx)
    if df is not None:
        lc = np.log(df["close"])
        F["mom28"] = lc - lc.shift(28)
        F["mom7"] = lc - lc.shift(7)
        if "wiki_en" in df:
            v = df["wiki_en"].where(df["wiki_en"] > 0)
            F["attn"] = np.log(v.rolling(7, min_periods=5).mean() / v.rolling(90, min_periods=60).mean())
        if "dvol" in df:                                     # wariancja z opcji minus zrealizowana z 30 dni (obie w skali roku)
            r2 = (lc.diff() ** 2).rolling(30, min_periods=25).mean() * 365
            F["vrp"] = (df["dvol"] / 100) ** 2 - r2
    if not WH.OUT.exists():
        return F.shift(1)
    w = pd.read_csv(WH.OUT, index_col=0, parse_dates=True)
    if "w_etf_flow" in w:
        f = w["w_etf_flow"].dropna()
        f3 = f.rolling(3).sum() / 1e6                                         # mln USD, dni sesyjne
        sgn = np.sign(f)
        streak = sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount().add(1) * sgn
        F["etf3"] = f3.reindex(idx).ffill(limit=4)
        F["etfstreak"] = streak.reindex(idx).ffill(limit=4)
    if "w_mstr_btc" in w:
        b = w["w_mstr_btc"].reindex(pd.date_range("2020-08-01", idx[-1])).fillna(0)
        F["mstr30"] = (b.rolling(30).sum() / 21e6 * 100).reindex(idx)        # % wszystkich bitcoinów
    if "w_hl_smart" in w:
        F["hl_smart"] = w["w_hl_smart"].reindex(idx)
        if "w_hl_dumb" in w:
            F["hl_spread"] = (w["w_hl_smart"] - w["w_hl_dumb"]).reindex(idx)
    if "w_hl_fund" in w:
        F["hl_fund7"] = w["w_hl_fund"].rolling(7, min_periods=5).mean().reindex(idx)
    if "w_usdt_px" in w:
        F["peg7"] = ((w["w_usdt_px"] - 1) * 100).rolling(7, min_periods=5).mean().reindex(idx)
    try:                                                         # H13: asymetria opcji odtworzona z transakcji Deribit (od 2018, spójna w czasie)
        import opthist
        O = opthist.load()
        if O is not None and O["rr25"].notna().sum() > 300:
            F["rr25"] = O["rr25"].reindex(idx)
    except Exception:
        pass
    if "rr25" not in F and "w_rr25" in w:
        F["rr25"] = w["w_rr25"].reindex(idx)
    try:                                                         # partia 6: regresory zmienności
        if df is not None:
            wk = [c for c in df.columns if c.startswith("wiki_")]
            if wk:
                tot = df[wk].sum(axis=1, min_count=6).rolling(7, min_periods=5).mean()
                F["attn_jump"] = np.log(tot / tot.shift(7))
            if {"fees", "minerrev"} <= set(df.columns):
                F["feeshare7"] = (df["fees"] / df["minerrev"]).rolling(7, min_periods=5).mean()
        import extra as XT
        if XT.OUT.exists():
            ex = pd.read_csv(XT.OUT, index_col=0, parse_dates=True)
            if {"x_exin_btc", "x_exout_btc", "x_exsupply"} <= set(ex.columns):
                F["exflow_int"] = ((ex["x_exin_btc"] - ex["x_exout_btc"]).abs().rolling(7, min_periods=5).sum() / ex["x_exsupply"]).reindex(idx)
            if "x_gdelt_tone" in ex:
                F["tone7"] = ex["x_gdelt_tone"].rolling(7, min_periods=4).mean().reindex(idx)
    except Exception:
        pass
    try:                                                         # partia 5: dźwignia z archiwum Binance
        import lev
        Lv = lev.load()
        if Lv is not None and df is not None:
            if "oi_btc" in Lv and "supply" in df:
                F["oi_sup"] = (Lv["oi_btc"].reindex(idx) / df["supply"].ffill()) * 100
            if "premium" in Lv:
                F["basis7"] = Lv["premium"].reindex(idx).rolling(7, min_periods=5).mean()
            if "top_ls_pos" in Lv:
                F["topls7"] = Lv["top_ls_pos"].reindex(idx).rolling(7, min_periods=5).mean()
    except Exception:
        pass
    parts = [_z(F[c]) for c in ("mom28", "attn", "etf3") if c in F]
    if parts:
        F["combo"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return F.shift(1)


def _spear(x, y):
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _p_hac(x, y, h, sign):
    """Test odporności: nachylenie regresji zwrotu na rangę sygnału, błąd standardowy Neweya-Westa z oknem h
    (nakładające się zwroty). Tylko informacyjnie: o statusie hipotezy decydują zasady spisane z góry."""
    xr = (rankdata(x) - 0.5) / len(x)
    xr = (xr - xr.mean()) / xr.std()
    yc = y - y.mean()
    b = float((xr * yc).sum() / (xr * xr).sum())
    u = (yc - b * xr) * xr
    n = len(u)
    S = (u * u).sum()
    for k in range(1, h + 1):
        S += 2 * (1 - k / (h + 1)) * (u[k:] * u[:-k]).sum()
    se = np.sqrt(S) / (xr * xr).sum()
    return float(1 - tdist.cdf(sign * b / se, n - 2)) if se > 0 else 1.0


# Partia 4 (spisana 2026-10-30): H14 = H7 sprawdzana WYŁĄCZNIE na danych po dniu rejestracji. Wynik H7 pochodzi z historii,
# na której ją wybraliśmy; jedynym czystym dowodem jest przyszłość. Werdykt dopiero przy n/h >= 250 (to lata).
HYP += [
    {"id": "H14", "col": "mom28", "h": 7, "sign": +1, "batch": 4, "from": "2026-10-30", "min_n": 250 * 7,
     "text": "Kontynuacja H7 na nowych danych (od 30.10.2026): wzrost w 4 tygodnie → wyższy zwrot w kolejnym tygodniu"},
]

# Partia 5 (spisana 2026-10-30, przed obejrzeniem danych): dźwignia z archiwum Binance (od 2020) jako ryzyko krachu, nie kierunek.
# Cel: czy w ciągu 30 dni cena spadnie choć raz (najniższa cena dnia) o 10% lub więcej (zgodnie z „Crypto carry”, BIS 2023).
HYP += [
    {"id": "H15", "col": "oi_sup", "h": 30, "sign": +1, "batch": 5, "target": "drop10",
     "text": "Dużo otwartych pozycji na kontraktach (względem podaży BTC) → częstszy spadek o 10% w ciągu 30 dni"},
    {"id": "H16", "col": "basis7", "h": 30, "sign": +1, "batch": 5, "target": "drop10",
     "text": "Wysoka premia kontraktów wieczystych nad ceną (tłok chętnych na dźwignię) → częstszy spadek o 10% w ciągu 30 dni"},
    {"id": "H17", "col": "topls7", "h": 30, "sign": +1, "batch": 5, "target": "drop10",
     "text": "Najwięksi traderzy mocno ustawieni na wzrost (stosunek długich do krótkich) → częstszy spadek o 10% w ciągu 30 dni"},
]

# Partia 6 (spisana 2026-10-31, przed obejrzeniem wyników): regresory ZMIENNOŚCI, nie kierunku. Cel: czy przez kolejne 7 dni
# wahania będą większe, niż przewiduje sama zmienność z przeszłości (log z ilorazu: zrealizowana wariancja 7 dni / prognoza).
# Potwierdzony regresor byłby kandydatem do modelu zmienności (tam musiałby jeszcze poprawić QLIKE na odłożonych danych).
HYP += [
    {"id": "H18", "col": "attn_jump", "h": 7, "sign": +1, "batch": 6, "target": "volratio",
     "text": "Nagły wzrost zainteresowania bitcoinem na Wikipedii (12 języków) → większe wahania w kolejnym tygodniu, niż przewiduje model"},
    {"id": "H19", "col": "exflow_int", "h": 7, "sign": +1, "batch": 6, "target": "volratio",
     "text": "Duże przepływy na giełdy i z giełd (względem zapasu na giełdach) → większe wahania w kolejnym tygodniu"},
    {"id": "H20", "col": "tone7", "h": 7, "sign": -1, "batch": 6, "target": "volratio",
     "text": "Bardziej negatywny ton newsów o bitcoinie (GDELT) → większe wahania w kolejnym tygodniu"},
    {"id": "H21", "col": "feeshare7", "h": 7, "sign": +1, "batch": 6, "target": "volratio",
     "text": "Duży udział opłat w przychodach górników (zatłoczona sieć) → większe wahania w kolejnym tygodniu"},
]

# Reguła emerytury (spisana 2026-10-30): potwierdzony sygnał przestaje być używany, gdy na dwóch kolejnych corocznych przeglądach
# (29 października, od 2027) w oknie ostatnich 3 lat jednocześnie p (Newey-West, jednostronnie) > 0,10 i korelacja w przewidzianym kierunku < 0,03.
RETIRE_FIRST = pd.Timestamp("2027-10-29")


def _window_stats(x, y, end, h, sign, years=3):
    m = x.notna() & y.notna() & (x.index > end - pd.DateOffset(years=years)) & (x.index <= end - pd.Timedelta(days=h))
    if m.sum() < 200:
        return None
    xv, yv = x[m].to_numpy(float), y[m].to_numpy(float)
    return {"rho": _spear(xv, yv) * sign, "p_hac": _p_hac(xv, yv, h, sign), "n": int(m.sum())}


def health(x, y, h, sign, today):
    """Zdrowie potwierdzonego sygnału: wynik z ostatnich 3 lat i coroczne przeglądy według reguły emerytury."""
    now = _window_stats(x, y, today, h, sign)
    checks, fails, retired = [], 0, False
    c = RETIRE_FIRST
    while c <= today:
        s = _window_stats(x, y, c, h, sign)
        bad = s is not None and s["p_hac"] > 0.10 and s["rho"] < 0.03
        fails = fails + 1 if bad else 0
        retired = retired or fails >= 2
        checks.append({"date": str(c.date()), **(s or {}), "fail": bool(bad)})
        c = c + pd.DateOffset(years=1)
    return {"now": now, "checks": checks, "retired": retired, "next_check": str(c.date())}


def run(df):
    F = features(df.index, df)
    lr = np.log(df["close"])
    today = df.index[-1]
    rows = []
    for H in HYP:
        k = sum(1 for x in HYP if x["batch"] == H["batch"])            # poprawka na liczbę prób w obrębie partii
        r = dict(H)
        r["alpha"] = ALPHA / k
        x = F[H["col"]] if H["col"] in F else pd.Series(np.nan, index=df.index)
        if H.get("from"):
            x = x.where(x.index >= pd.Timestamp(H["from"]))
        if H.get("target") == "volratio":                       # zrealizowana wariancja h dni / prognoza z przeszłości (EWMA)
            r2 = lr.diff() ** 2
            ew = r2.ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean()
            y = np.log(r2[::-1].rolling(H["h"]).mean()[::-1].shift(-1) / ew)
        elif H.get("target") == "drop10":                       # zdarzenie: najniższa cena w ciągu h dni ≤ 90% dzisiejszego zamknięcia
            low = np.log(df["low"] if "low" in df else df["close"])
            fmin = low[::-1].rolling(H["h"]).min()[::-1].shift(-1)
            y = (fmin - lr <= np.log(0.9)).astype(float).where(fmin.notna())
        else:
            y = lr.shift(-H["h"]) - lr
        m = x.notna() & y.notna()
        n = int(m.sum())
        n_eff0 = max(n / H["h"], 5)
        r.update({"n": n, "now": None if x.dropna().empty else float(x.dropna().iloc[-1]),
                  "first": None if x.dropna().empty else x.dropna().index[0].strftime("%Y-%m-%d"),
                  # najmniejsza korelacja, którą test wykryłby w 4 na 5 przypadków przy obecnej liczbie niezależnych obserwacji
                  "mde_rho": float((norm.ppf(1 - ALPHA / k) + norm.ppf(0.8)) / np.sqrt(max(n_eff0 if n >= MIN_DAYS else 250, 5)))})
        if n < H.get("min_n", MIN_DAYS):
            r.update({"status": "collect", "r": None, "p": None})
            rows.append(r)
            continue
        xv, yv = x[m].to_numpy(float), y[m].to_numpy(float)
        rho = _spear(xv, yv)
        n_eff = max(n / H["h"], 5)
        tt = rho * np.sqrt((n_eff - 2) / max(1 - rho * rho, 1e-9))
        p = float(1 - tdist.cdf(H["sign"] * tt, n_eff - 2))
        half = n // 2
        ra, rb = _spear(xv[:half], yv[:half]), _spear(xv[half:], yv[half:])
        stable = H["sign"] * ra > 0 and H["sign"] * rb > 0
        q = pd.qcut(pd.Series(xv).rank(method="first"), 3, labels=False)
        ret = pd.Series(yv) * 100 if H.get("target") == "drop10" else (np.exp(pd.Series(yv)) - 1) * 100   # dla zdarzeń: częstość w %
        ok = p < ALPHA / k and stable
        r.update({"r": rho, "p": p, "r_a": ra, "r_b": rb, "stable": bool(stable), "n_eff": n_eff, "p_hac": _p_hac(xv, yv, H["h"], H["sign"]),
                  # dla zmienności mediana: średnią z ilorazów wariancji psują pojedyncze wybuchy
                  "ret_lo": float((ret[q == 0].median() if H.get("target") == "volratio" else ret[q == 0].mean())),
                  "ret_hi": float((ret[q == 2].median() if H.get("target") == "volratio" else ret[q == 2].mean())),
                  "status": "confirmed" if ok else ("reverse" if H["sign"] * rho < 0 and (1 - p) < ALPHA / k else "rejected")})
        if r["status"] == "confirmed":
            r["health"] = health(x, y, H["h"], H["sign"], today)
            if r["health"]["retired"]:
                r["status"] = "retired"
        rows.append(r)
    # do kierunku idą tylko sygnały zwrotu; potwierdzone regresory zmienności i zdarzeń trzymamy osobno
    conf = list(dict.fromkeys(r["col"] for r in rows if r["status"] == "confirmed" and not r.get("target")))
    vconf = list(dict.fromkeys(r["col"] for r in rows if r["status"] == "confirmed" and r.get("target") == "volratio"))
    return {"rows": rows, "confirmed": conf, "vol_confirmed": vconf, "features": F, "alpha_each": ALPHA / sum(1 for x in HYP if x["batch"] == 1)}
