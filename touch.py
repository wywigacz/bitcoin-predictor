"""Szansa, że cena DOTKNIE danego poziomu w ciągu H dni (a nie tylko, gdzie skończy).

Metoda: symulacja tysięcy ścieżek. Dzienny ruch = zmienność × losowo wybrany historyczny ruch standaryzowany
(prawdziwe „grube ogony” Bitcoina, bez założenia rozkładu normalnego). Warianty, porównywane tym samym testem wstecz:
  base   - zmienność wraca do długoterminowej mediany z okresem połowicznym 30 dni, kierunek neutralny,
  fhs    - filtrowana symulacja historyczna (Barone-Adesi i in. 1998): zmienność w każdej ścieżce reaguje na wylosowane
           ruchy według GJR-GARCH (parametry dopasowane tylko do danych znanych w dniu prognozy, odświeżane co rok),
  drift  - jak base, ale z dryfem równym średniemu dziennemu ruchowi z ostatnich 4 lat (tylko przeszłość),
  fhs_drift - oba naraz.
Do tabeli na panelu idzie wariant z najlepszym wynikiem Briera w teście wstecz (wybór spośród 4 wariantów na tym samym
teście jest lekko optymistyczny, dlatego pokazujemy wyniki wszystkich).

Kalibracja: co tydzień od 2016 roku liczymy szansę tylko z danymi znanymi tego dnia i sprawdzamy, jak często cena
naprawdę dotykała poziomu. Punkt odniesienia: historyczna częstość dotknięć liczona wyłącznie z przeszłości."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
HALF = 30.0
START = "2013-01-01"
VARIANTS = ("base", "fhs", "drift", "fhs_drift")
HOLDOUT = pd.Timestamp("2024-10-29")                  # egzamin: ostatnie 2 lata odcięte od wyboru wariantu
LABELS = {"base": "Zmienność wraca do normy", "fhs": "Zmienność reaguje na ruchy (FHS, GARCH)",
          "drift": "Z dryfem z ostatnich 4 lat", "fhs_drift": "FHS z dryfem"}


def _resid(df, sig):
    r = np.log(df["close"]).diff().to_numpy(float)
    s = np.r_[np.nan, sig[:-1]]                         # zmienność znana dzień wcześniej
    z = r / s
    z[(df.index < START) | ~np.isfinite(z)] = np.nan
    return z


def _garch(r_pct):
    import ensemble as EN
    _, p = EN.garch_gjr(r_pct)
    w, a, g, b = p
    return float(a), float(g), float(b)


def _paths(s0, s_lr, pool, H, n, mode="base", gp=None, mu=0.0):
    z = RNG.choice(pool, size=(n, H))
    z = (z - pool.mean()) / pool.std()                 # średnio zero, wariancja 1
    if mode in ("fhs", "fhs_drift") and gp is not None:
        a, g, b = gp
        hl = s_lr ** 2
        h = np.full(n, s0 ** 2)
        out = np.empty((n, H))
        for t in range(H):
            r = np.sqrt(h) * z[:, t]
            out[:, t] = r
            h = hl * (1 - a - b - 0.5 * g) + (a + g * (r < 0)) * r ** 2 + b * h
            h = np.clip(h, hl * 0.05, hl * 50)
        steps = out
    else:
        t = np.arange(1, H + 1)
        sp = s_lr + (s0 - s_lr) * np.exp(-np.log(2) * t / HALF)
        steps = z * sp
    if mode in ("drift", "fhs_drift"):
        steps = steps + mu
    return np.cumsum(steps, axis=1)                    # log(P_t / P_0)


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


def _mu(lr_diff, t):
    """Średni dzienny ruch z ostatnich 4 lat, znany w dniu t."""
    w = lr_diff[max(0, t - 1460):t + 1]
    w = w[np.isfinite(w)]
    return float(w.mean()) if len(w) > 500 else 0.0


def run(df, sig, horizons=(7, 30, 90), n=20000):
    z = _resid(df, sig)
    pool = z[np.isfinite(z)]
    s_hist = pd.Series(sig, df.index)
    s_lr = float(s_hist.loc[START:].median())
    s0 = float(sig[-1])
    P = float(df["close"].iloc[-1])
    lr = np.log(df["close"].to_numpy(float))
    dlr = np.r_[np.nan, np.diff(lr)]
    r_pct = np.nan_to_num(dlr * 100)
    gp = _garch(r_pct[np.searchsorted(df.index, pd.Timestamp(START)):])
    cal = calibrate(df, sig, z)
    best = max(cal, key=lambda m: cal[m]["skill_pre"])           # wybór tylko na danych sprzed egzaminu
    mu = _mu(dlr, len(df) - 1)
    sims = {H: _paths(s0, s_lr, pool, H, n, best, gp, mu) for H in horizons}
    table = []
    for L in levels(P):
        row = {"level": L, "rel": L / P - 1}
        for H in horizons:
            row[f"p{H}"] = probs(sims[H], [L / P - 1])[0]
        table.append(row)
    rels = [-0.2, -0.1, 0.1, 0.2]
    simple = {H: dict(zip(rels, probs(sims[H], rels))) for H in horizons}
    return {"price": P, "table": table, "simple": simple, "sigma_now": s0, "sigma_lr": s_lr,
            "best": best, "calib": cal[best], "variants": {m: {k: v for k, v in c.items() if k != "rows"} for m, c in cal.items()}}


def _brier_parts(pred, hit, bins=10):
    """Rozkład Murphy'ego: Brier = niezawodność − rozdzielczość + niepewność."""
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    base = hit.mean()
    rel = res = 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            w = m.mean()
            rel += w * (pred[m].mean() - hit[m].mean()) ** 2
            res += w * (hit[m].mean() - base) ** 2
    return float(rel), float(res), float(base * (1 - base))


def calibrate(df, sig, z, H=30, rels=(-0.2, -0.1, 0.1, 0.2), n=3000, step=7):
    """Test wstecz wszystkich wariantów: przewidywana szansa dotknięcia vs to, co się stało. Tylko dane znane w dniu prognozy."""
    lp_all = np.log(df["close"].to_numpy(float))
    dlr = np.r_[np.nan, np.diff(lp_all)]
    r_pct = np.nan_to_num(dlr * 100)
    idx = df.index
    s_hist = pd.Series(sig, idx)
    first = int(np.searchsorted(idx, pd.Timestamp("2016-01-01")))
    i0 = int(np.searchsorted(idx, pd.Timestamp(START)))
    gp, next_fit = None, 0
    acc = {m: {"pred": [], "hit": [], "which": [], "when": []} for m in VARIANTS}
    for t in range(first, len(df) - H, step):
        pool = z[:t + 1]
        pool = pool[np.isfinite(pool)]
        if len(pool) < 500:
            continue
        if gp is None or t >= next_fit:                 # GARCH dopasowany tylko do przeszłości, odświeżany co rok
            gp = _garch(r_pct[i0:t + 1])
            next_fit = t + 365
        s_lr = float(s_hist.iloc[:t + 1].loc[START:].median())
        mu = _mu(dlr, t)
        fut = lp_all[t + 1:t + H + 1] - lp_all[t]
        hits = [float(fut.max() >= np.log(1 + x)) if x > 0 else float(fut.min() <= np.log(1 + x)) for x in rels]
        for m in VARIANTS:
            lp = _paths(float(sig[t]), s_lr, pool, H, n, m, gp, mu)
            for x, pr, h in zip(rels, probs(lp, list(rels)), hits):
                a = acc[m]
                a["pred"].append(pr); a["hit"].append(h); a["which"].append(x); a["when"].append(t)
    out = {}
    for m, a in acc.items():
        pred, hit, which, when = (np.array(a[k]) for k in ("pred", "hit", "which", "when"))
        # punkt odniesienia: historyczna częstość dotknięć danego poziomu, TYLKO z prób zakończonych przed dniem prognozy
        base = np.full(len(pred), np.nan)
        for i in range(len(pred)):
            mm = (which == which[i]) & (when <= when[i] - H)
            if mm.sum() >= 30:
                base[i] = hit[mm].mean()
        ok = np.isfinite(base)
        pred, hit, which, base, when = pred[ok], hit[ok], which[ok], base[ok], when[ok]
        cut = int(np.searchsorted(idx, HOLDOUT))
        pre, post = when + H < cut, when >= cut                # egzamin: wybór na „pre”, sprawdzian na „post”
        sk = lambda m_: float(1 - np.mean((pred[m_] - hit[m_]) ** 2) / np.mean((base[m_] - hit[m_]) ** 2)) if m_.sum() > 20 else np.nan
        exam = {"skill_pre": sk(pre), "skill_post": sk(post), "n_post": int(post.sum())}
        bins = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (pred >= lo) & (pred < hi)
            if mm.sum() >= 20:
                rows.append({"lo": lo, "hi": hi, "n": int(mm.sum()), "pred": float(pred[mm].mean()), "real": float(hit[mm].mean())})
        brier = float(np.mean((pred - hit) ** 2))
        brier0 = float(np.mean((base - hit) ** 2))
        rel, res, unc = _brier_parts(pred, hit)
        rel0, res0, _ = _brier_parts(base, hit)
        side = {float(x): (float(pred[which == x].mean()), float(hit[which == x].mean())) for x in rels}
        out[m] = {"rows": rows, "n": int(len(pred)), "brier": brier, "brier_base": brier0, "skill": 1 - brier / brier0, "H": H,
                  "side": side, "reliability": rel, "resolution": res, "uncertainty": unc, "reliability_base": rel0, "resolution_base": res0, **exam}
    return out
