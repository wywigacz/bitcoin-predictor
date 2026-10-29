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
from scipy.stats import rankdata, t as tdist

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
    if "w_rr25" in w:
        F["rr25"] = w["w_rr25"].reindex(idx)
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


def run(df):
    F = features(df.index, df)
    lr = np.log(df["close"])
    rows = []
    for H in HYP:
        k = sum(1 for x in HYP if x["batch"] == H["batch"])            # poprawka na liczbę prób w obrębie partii
        r = dict(H)
        r["alpha"] = ALPHA / k
        x = F[H["col"]] if H["col"] in F else pd.Series(np.nan, index=df.index)
        y = lr.shift(-H["h"]) - lr
        m = x.notna() & y.notna()
        n = int(m.sum())
        r.update({"n": n, "now": None if x.dropna().empty else float(x.dropna().iloc[-1]),
                  "first": None if x.dropna().empty else x.dropna().index[0].strftime("%Y-%m-%d")})
        if n < MIN_DAYS:
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
        ret = (np.exp(pd.Series(yv)) - 1) * 100
        ok = p < ALPHA / k and stable
        r.update({"r": rho, "p": p, "r_a": ra, "r_b": rb, "stable": bool(stable), "n_eff": n_eff, "p_hac": _p_hac(xv, yv, H["h"], H["sign"]),
                  "ret_lo": float(ret[q == 0].mean()), "ret_hi": float(ret[q == 2].mean()),
                  "status": "confirmed" if ok else ("reverse" if H["sign"] * rho < 0 and (1 - p) < ALPHA / k else "rejected")})
        rows.append(r)
    conf = [r["col"] for r in rows if r["status"] == "confirmed"]
    return {"rows": rows, "confirmed": conf, "features": F, "alpha_each": ALPHA / sum(1 for x in HYP if x["batch"] == 1)}
