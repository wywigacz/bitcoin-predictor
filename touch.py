"""Szansa, że cena DOTKNIE danego poziomu w ciągu H dni (a nie tylko, gdzie skończy).

Metoda: symulacja tysięcy ścieżek. Dzienny ruch = dzisiejsza zmienność × losowo wybrany historyczny ruch standaryzowany
(prawdziwe „grube ogony” Bitcoina, bez założenia rozkładu normalnego). Zmienność wraca do swojej długoterminowej mediany
z okresem połowicznym 30 dni. Kierunek neutralny (średni ruch zero), zgodnie z prognozą bazową.

Kalibracja: co tydzień od 2016 roku liczymy tę samą szansę tylko z danymi znanymi tego dnia i sprawdzamy, jak często
cena naprawdę dotykała poziomu. Jeśli model mówi 30%, a w historii wychodziło 30%, to jest uczciwie skalibrowany."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
HALF = 30.0
START = "2013-01-01"


def _resid(df, sig):
    r = np.log(df["close"]).diff().to_numpy(float)
    s = np.r_[np.nan, sig[:-1]]                         # zmienność znana dzień wcześniej
    z = r / s
    z[(df.index < START) | ~np.isfinite(z)] = np.nan
    return z


def _paths(s0, s_lr, pool, H, n):
    t = np.arange(1, H + 1)
    sp = s_lr + (s0 - s_lr) * np.exp(-np.log(2) * t / HALF)
    z = RNG.choice(pool, size=(n, H))
    z = z - pool.mean()
    return np.cumsum(z * sp, axis=1)                    # log(P_t / P_0)


def probs(lp, rel):
    hi, lo = lp.max(axis=1), lp.min(axis=1)
    out = []
    for x in rel:
        l = np.log(1 + x)
        out.append(float((hi >= l).mean()) if x > 0 else float((lo <= l).mean()))
    return out


def levels(price):
    step = 10000 if price > 40000 else 5000 if price > 15000 else 1000
    base = np.floor(price / step) * step
    up = [base + step * k for k in range(1, 5)]
    dn = [base - step * k for k in range(0, 4) if base - step * k > 0]
    return sorted(set(dn + up))


def run(df, sig, horizons=(7, 30, 90), n=20000):
    z = _resid(df, sig)
    pool = z[np.isfinite(z)]
    s_hist = pd.Series(sig, df.index)
    s_lr = float(s_hist.loc[START:].median())
    s0 = float(sig[-1])
    P = float(df["close"].iloc[-1])
    lv = levels(P)
    table = []
    sims = {H: _paths(s0, s_lr, pool, H, n) for H in horizons}
    for L in lv:
        row = {"level": L, "rel": L / P - 1}
        for H in horizons:
            row[f"p{H}"] = probs(sims[H], [L / P - 1])[0]
        table.append(row)
    rels = [-0.2, -0.1, 0.1, 0.2]
    simple = {H: dict(zip(rels, probs(sims[H], rels))) for H in horizons}
    return {"price": P, "table": table, "simple": simple, "sigma_now": s0, "sigma_lr": s_lr, "calib": calibrate(df, sig, z)}


def calibrate(df, sig, z, H=30, rels=(-0.2, -0.1, 0.1, 0.2), n=3000, step=7):
    """Test wstecz: przewidywana szansa dotknięcia vs to, co się stało. Tylko dane znane w dniu prognozy."""
    lp_all = np.log(df["close"].to_numpy(float))
    idx = df.index
    s_hist = pd.Series(sig, idx)
    first = int(np.searchsorted(idx, pd.Timestamp("2016-01-01")))
    pred, hit, which, when = [], [], [], []
    for t in range(first, len(df) - H, step):
        pool = z[:t + 1]
        pool = pool[np.isfinite(pool)]
        if len(pool) < 500:
            continue
        s_lr = float(s_hist.iloc[:t + 1].loc[START:].median())
        lp = _paths(float(sig[t]), s_lr, pool, H, n)
        p = probs(lp, list(rels))
        fut = lp_all[t + 1:t + H + 1] - lp_all[t]
        for x, pr in zip(rels, p):
            l = np.log(1 + x)
            pred.append(pr)
            which.append(x)
            when.append(t)
            hit.append(float(fut.max() >= l) if x > 0 else float(fut.min() <= l))
    pred, hit, which, when = np.array(pred), np.array(hit), np.array(which), np.array(when)
    # punkt odniesienia: dla danego poziomu historyczna częstość dotknięć, liczona TYLKO z prób zakończonych przed dniem prognozy
    base = np.full(len(pred), np.nan)
    for i in range(len(pred)):
        m = (which == which[i]) & (when <= when[i] - H)
        if m.sum() >= 30:
            base[i] = hit[m].mean()
    ok = np.isfinite(base)
    pred, hit, which, base = pred[ok], hit[ok], which[ok], base[ok]
    bins = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (pred >= a) & (pred < b)
        if m.sum() >= 20:
            rows.append({"lo": a, "hi": b, "n": int(m.sum()), "pred": float(pred[m].mean()), "real": float(hit[m].mean())})
    brier = float(np.mean((pred - hit) ** 2))
    brier0 = float(np.mean((base - hit) ** 2))
    side = {float(x): (float(pred[which == x].mean()), float(hit[which == x].mean())) for x in rels}
    return {"rows": rows, "n": int(len(pred)), "brier": brier, "brier_base": brier0, "skill": 1 - brier / brier0, "H": H, "side": side}
