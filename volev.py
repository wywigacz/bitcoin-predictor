"""Zdarzenia zmienności, czyli to, co ten system naprawdę potrafi przewidywać:
  burza7  - czy następny tydzień będzie burzliwy (zmienność 7 dni powyżej swojej mediany z ostatniego roku),
  dzien5  - czy w ciągu 7 dni zdarzy się dzień z ruchem ceny o co najmniej 5% (w górę lub w dół),
  dzien10 - czy w ciągu 30 dni zdarzy się dzień z ruchem o co najmniej 10%.
Model: regresja logistyczna na dzisiejszej prognozie zmienności, dopasowywana co tydzień tylko na zdarzeniach,
których wynik był już znany. Punkty odniesienia: stała częstość z przeszłości oraz „jak ostatnio” (częstość zdarzenia
pod warunkiem tego, co działo się w ostatnim okresie). Egzamin: odcięte 2 lata od 2024-10-29.
Reguła (spisana 2026-10-30, przed wynikami): „przewaga potwierdzona” tylko gdy model bije „jak ostatnio” i przed, i po odcięciu."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HOLDOUT = pd.Timestamp("2024-10-29")
START = pd.Timestamp("2015-09-01")
STEP = 7
EVENTS = {
    "burza7": {"H": 7, "label": "Burzliwy tydzień", "q": "Czy następny tydzień będzie burzliwy?"},
    "dzien5": {"H": 7, "label": "Dzień z ruchem ≥5% w 7 dni", "q": "Czy w ciągu tygodnia zdarzy się dzień z ruchem o 5% lub więcej?"},
    "dzien10": {"H": 30, "label": "Dzień z ruchem ≥10% w 30 dni", "q": "Czy w ciągu miesiąca zdarzy się dzień z ruchem o 10% lub więcej?"},
}


def _series(df, sig):
    r = np.log(df["close"]).diff()
    r2 = r ** 2
    rv7 = r2.rolling(7).mean()                                   # zmienność z ostatnich 7 dni (znana w dniu t)
    med = rv7.rolling(365, min_periods=200).median()
    absr = r.abs()
    S = pd.DataFrame(index=df.index)
    S["sig"] = sig
    S["x_rel"] = np.log(S["sig"] ** 2 / med)
    S["x_abs"] = np.log(S["sig"])
    # zdarzenia w przyszłości (to, co oceniamy) i ich odpowiedniki z przeszłości (do „jak ostatnio”)
    S["y_burza7"] = (rv7.shift(-7) > med).astype(float).where(rv7.shift(-7).notna() & med.notna())
    S["p_burza7"] = (rv7 > med).astype(float).where(med.notna())
    fwd = lambda H: absr[::-1].rolling(H).max()[::-1].shift(-1)            # max |r| w dniach t+1..t+H
    S["y_dzien5"] = (fwd(7) >= np.log(1.05)).astype(float).where(fwd(7).notna())
    S["p_dzien5"] = (absr.rolling(7).max() >= np.log(1.05)).astype(float)
    S["y_dzien10"] = (fwd(30) >= np.log(1.10)).astype(float).where(fwd(30).notna())
    S["p_dzien10"] = (absr.rolling(30).max() >= np.log(1.10)).astype(float)
    return S


def run(df, sig):
    S = _series(df, sig)
    idx = S.index
    out = {}
    for ev, meta in EVENTS.items():
        H = meta["H"]
        x = S["x_rel" if ev == "burza7" else "x_abs"].to_numpy(float)
        y = S[f"y_{ev}"].to_numpy(float)
        prev = S[f"p_{ev}"].to_numpy(float)
        rows = []
        first = int(np.searchsorted(idx, START))
        for t in range(first, len(idx) - H, STEP):
            known = np.arange(0, t - H + 1)
            known = known[np.isfinite(x[known]) & np.isfinite(y[known]) & np.isfinite(prev[known])]
            if len(known) < 500 or not (np.isfinite(x[t]) and np.isfinite(y[t]) and np.isfinite(prev[t])):
                continue
            m = LogisticRegression(C=1.0).fit(x[known][:, None], y[known])
            p_mod = float(m.predict_proba([[x[t]]])[0, 1])
            p_clim = float(y[known].mean())
            same = known[prev[known] == prev[t]]
            p_pers = float(y[same].mean()) if len(same) >= 30 else p_clim
            rows.append((idx[t], y[t], p_mod, p_clim, p_pers))
        R = pd.DataFrame(rows, columns=["d", "y", "mod", "clim", "pers"])
        b = lambda col, msk: float(((R[col][msk] - R["y"][msk]) ** 2).mean())
        pre = (R["d"] + pd.Timedelta(days=H) < HOLDOUT).to_numpy()
        post = (R["d"] >= HOLDOUT).to_numpy()
        res = {}
        for nm, msk in (("przed", pre), ("po", post), ("calosc", np.ones(len(R), bool))):
            bm, bc, bp = b("mod", msk), b("clim", msk), b("pers", msk)
            res[nm] = {"n": int(msk.sum()), "brier": bm, "skill_clim": 1 - bm / bc, "skill_pers": 1 - bm / bp}
        # rzetelność: w koszykach przewidywanej szansy (cała historia)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (R["mod"] >= lo) & (R["mod"] < hi)
            if mk.sum() >= 15:
                rel.append({"lo": lo, "hi": hi, "n": int(mk.sum()), "pred": float(R["mod"][mk].mean()), "real": float(R["y"][mk].mean())})
        # szansa na dziś: model dopasowany na wszystkich znanych wynikach
        t = len(idx) - 1
        known = np.arange(0, t - H + 1)
        known = known[np.isfinite(x[known]) & np.isfinite(y[known])]
        m = LogisticRegression(C=1.0).fit(x[known][:, None], y[known])
        p_now = float(m.predict_proba([[x[t]]])[0, 1]) if np.isfinite(x[t]) else None
        confirmed = res["przed"]["skill_pers"] > 0 and res["po"]["skill_pers"] > 0
        out[ev] = {**meta, "p_now": p_now, "base_rate": float(y[known].mean()), "eval": res, "reliability": rel, "confirmed": bool(confirmed)}
    return out
