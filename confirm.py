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


def features(idx):
    """Cechy dzienne wyrównane do indeksu BTC i przesunięte o 1 dzień (wartość z dnia D używana od dnia D+1)."""
    if not WH.OUT.exists():
        return pd.DataFrame(index=idx)
    w = pd.read_csv(WH.OUT, index_col=0, parse_dates=True)
    F = pd.DataFrame(index=idx)
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
    return F.shift(1)


def _spear(x, y):
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def run(df):
    F = features(df.index)
    lr = np.log(df["close"])
    k = len(HYP)
    rows = []
    for H in HYP:
        r = dict(H)
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
        r.update({"r": rho, "p": p, "r_a": ra, "r_b": rb, "stable": bool(stable), "n_eff": n_eff,
                  "ret_lo": float(ret[q == 0].mean()), "ret_hi": float(ret[q == 2].mean()),
                  "status": "confirmed" if ok else ("reverse" if H["sign"] * rho < 0 and (1 - p) < ALPHA / k else "rejected")})
        rows.append(r)
    conf = [r["col"] for r in rows if r["status"] == "confirmed"]
    return {"rows": rows, "confirmed": conf, "features": F, "alpha_each": ALPHA / k}
