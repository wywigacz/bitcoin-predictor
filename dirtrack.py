"""Osobny tor kierunku (punkt 12 planu): szansa, że za 7 dni cena będzie wyższa, liczona wprost z potwierdzonych sygnałów.

Dlaczego osobno: główny test metod porównuje cały rozkład (7 kwantyli), a słaby, ale prawdziwy sygnał kierunku przesuwa
rozkład tak mało, że tego testu nie może zdać (sygnał wielkości H7 daje ok. 0,8% poprawy przy progu 2%). Tu oceniamy tylko kierunek.
Model: regresja logistyczna na średniej ze sprawdzonych z góry sygnałów (dziś trend 28 dni i ich połączenie), dopasowywana co tydzień
tylko na wynikach już znanych; prognoza codziennie od 2015. Ocena: wynik Briera względem stałej historycznej częstości wzrostów
(liczonej tylko z przeszłości), test DM z poprawką HLN, trafność kierunku.
Reguła (spisana 2026-10-31, przed wynikami): szansa z tego toru zastępuje 50% na panelu tylko, jeśli bije stałą częstość
z p < 0,10 i przed 2024-10-29, i na egzaminie. Uwaga: sygnały potwierdziliśmy na całej historii, więc okres „przed” nie jest
czysty; jedynym czystym sprawdzianem jest egzamin i przyszłość."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import confirm as CO
import ensemble as EN

H = 7
START = pd.Timestamp("2015-09-01")
HOLDOUT = pd.Timestamp("2024-10-29")


def run(df, CF):
    if not CF or not CF.get("confirmed"):
        return None
    F = CF["features"]
    x = pd.concat([CO._z(F[c]) for c in CF["confirmed"]], axis=1).mean(axis=1).to_numpy(float)
    lr = np.log(df["close"])
    y = ((lr.shift(-H) - lr) > 0).astype(float).where((lr.shift(-H)).notna()).to_numpy(float)
    idx = df.index
    rows, model, next_fit = [], None, 0
    for t in range(int(np.searchsorted(idx, START)), len(idx)):
        if not np.isfinite(x[t]):
            continue
        known = np.arange(0, max(t - H + 1, 0))
        known = known[np.isfinite(x[known]) & np.isfinite(y[known])]
        if len(known) < 500:
            continue
        if model is None or t >= next_fit:
            model = LogisticRegression(C=1.0).fit(x[known][:, None], y[known])
            next_fit = t + 7
        p = float(model.predict_proba([[x[t]]])[0, 1])
        clim = float(y[known].mean())
        rows.append((idx[t], p, clim, y[t]))
    R = pd.DataFrame(rows, columns=["d", "p", "clim", "y"])
    live = R.iloc[-1] if len(R) else None
    R = R.dropna(subset=["y"])
    lag = H + 1
    res = {}
    for nm, m in (("przed", (R["d"] + pd.Timedelta(days=H) < HOLDOUT).to_numpy()), ("po", (R["d"] >= HOLDOUT).to_numpy())):
        if m.sum() < 60:
            continue
        lp, lc = (R["p"][m] - R["y"][m]) ** 2, (R["clim"][m] - R["y"][m]) ** 2
        _, pv = EN._dm(lc.to_numpy(), lp.to_numpy(), lag)
        hit = ((R["p"][m] > 0.5) == (R["y"][m] > 0.5)).astype(float)
        res[nm] = {"n": int(m.sum()), "n_eff": float(m.sum() / H), "skill": float(1 - lp.mean() / lc.mean()), "p": pv,
                   "hit": float(hit.mean()), "mde": EN._mde(lc.to_numpy(), lp.to_numpy(), lag), "up_rate": float(R["y"][m].mean())}
    passed = all(k in res and res[k]["skill"] > 0 and res[k]["p"] == res[k]["p"] and res[k]["p"] < 0.10 for k in ("przed", "po"))
    return {"eval": res, "passed": bool(passed), "p_now": None if live is None else float(live["p"]),
            "clim_now": None if live is None else float(live["clim"]), "signals": list(CF["confirmed"])}
