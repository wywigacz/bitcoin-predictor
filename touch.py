"""Szansa, że cena DOTKNIE danego poziomu w ciągu H dni (a nie tylko, gdzie skończy).

Metoda: symulacja tysięcy ścieżek. Każdy dzień ścieżki to losowo wybrany prawdziwy dzień z historii: jego zmiana ceny na
zamknięcie oraz to, jak wysoko i nisko cena sięgnęła w ciągu dnia (wszystko wyrażone w jednostkach ówczesnej zmienności
i przeskalowane do dzisiejszej). Dzięki temu „dotknięcie” liczy się tak, jak obiecuje panel: choć raz w ciągu dnia,
a nie tylko na zamknięciu. Warianty, porównywane tym samym testem wstecz:
  base   - zmienność wraca do długoterminowej mediany z okresem połowicznym 30 dni, kierunek neutralny,
  fhs    - filtrowana symulacja historyczna (Barone-Adesi i in. 1998): zmienność w każdej ścieżce reaguje na wylosowane
           ruchy według GJR-GARCH (parametry dopasowane tylko do danych znanych w dniu prognozy, odświeżane co rok),
  drift  - jak base, ale z dryfem równym średniemu dziennemu ruchowi z ostatnich 4 lat (tylko przeszłość),
  fhs_drift - oba naraz.
Wariant wybieramy wyłącznie na prognozach rozstrzygniętych przed 2024-10-29; ostatnie 2 lata to egzamin.

Reguła (spisana 2026-10-30, przed wynikami): dotknięcie liczone po maksimach i minimach dnia zastępuje liczenie po
zamknięciach, jeśli najlepszy wariant tej definicji bije stałą historyczną częstość (też liczoną po maksimach/minimach)
i przed odcięciem, i na egzaminie. Inaczej zostajemy przy zamknięciach i panel mówi „na zamknięcie dnia”.
Ochrona przed pojedynczymi „knotami” jednej giełdy: zasięg dnia przycinamy do 0,1–99,9 percentyla historii."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
HALF = 30.0
START = "2013-01-01"
VARIANTS = ("base", "fhs", "drift", "fhs_drift")
HOLDOUT = pd.Timestamp("2024-10-29")                  # egzamin: ostatnie 2 lata odcięte od wyboru wariantu
LABELS = {"base": "Zmienność wraca do normy", "fhs": "Zmienność reaguje na ruchy (FHS, GARCH)",
          "drift": "Z dryfem z ostatnich 4 lat", "fhs_drift": "FHS z dryfem"}


def _days(df, sig):
    """Standaryzowane dni: zmiana na zamknięcie (zc) i zasięg w górę/dół od wczorajszego zamknięcia (zh, zl)."""
    c = np.log(df["close"].to_numpy(float))
    hi = np.log(df["high"].to_numpy(float)) if "high" in df else c
    lo = np.log(df["low"].to_numpy(float)) if "low" in df else c
    s = np.r_[np.nan, sig[:-1]]                         # zmienność znana dzień wcześniej
    prev = np.r_[np.nan, c[:-1]]
    zc, zh, zl = (c - prev) / s, (hi - prev) / s, (lo - prev) / s
    bad = (df.index < START) | ~np.isfinite(zc) | ~np.isfinite(zh) | ~np.isfinite(zl)
    zc[bad], zh[bad], zl[bad] = np.nan, np.nan, np.nan
    zh = np.maximum(zh, np.maximum(zc, 0.0))            # maksimum dnia nie niżej niż otwarcie i zamknięcie
    zl = np.minimum(zl, np.minimum(zc, 0.0))
    ok = np.isfinite(zc)
    if ok.sum() > 100:
        zh = np.minimum(zh, np.nanpercentile(zh[ok], 99.9))
        zl = np.maximum(zl, np.nanpercentile(zl[ok], 0.1))
    return np.column_stack([zc, zh, zl])


def _garch(r_pct):
    import ensemble as EN
    _, p = EN.garch_gjr(r_pct)
    w, a, g, b = p
    return float(a), float(g), float(b)


def _paths(s0, s_lr, pool, H, n, mode="base", gp=None, mu=0.0):
    """Zwraca (najwyższy punkt ścieżki, najniższy punkt ścieżki, maks. zamknięcie, min. zamknięcie) w log(P/P0)."""
    k = RNG.integers(0, len(pool), size=(n, H))
    m, sd = pool[:, 0].mean(), pool[:, 0].std()
    zc, zh, zl = ((pool[k, j] - m) / sd for j in range(3))   # przesunięcie i skala wspólne dla całego dnia
    if mode in ("fhs", "fhs_drift") and gp is not None:
        a, g, b = gp
        hl = s_lr ** 2
        h = np.full(n, s0 ** 2)
        sc = np.empty((n, H))
        for t in range(H):
            sc[:, t] = np.sqrt(h)
            r = sc[:, t] * zc[:, t]
            h = hl * (1 - a - b - 0.5 * g) + (a + g * (r < 0)) * r ** 2 + b * h
            h = np.clip(h, hl * 0.05, hl * 50)
    else:
        t = np.arange(1, H + 1)
        sc = np.broadcast_to(s_lr + (s0 - s_lr) * np.exp(-np.log(2) * t / HALF), (n, H))
    d = mu if mode in ("drift", "fhs_drift") else 0.0
    close = np.cumsum(zc * sc + d, axis=1)
    prev = np.column_stack([np.zeros(n), close[:, :-1]])
    up, dn = prev + zh * sc + d, prev + zl * sc + d
    return up.max(axis=1), dn.min(axis=1), close.max(axis=1), close.min(axis=1)


def probs(ext, rel, hl=True):
    hi, lo = (ext[0], ext[1]) if hl else (ext[2], ext[3])
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
    D = _days(df, sig)
    pool = D[np.isfinite(D).all(axis=1)]
    s_hist = pd.Series(sig, df.index)
    s_lr = float(s_hist.loc[START:].median())
    s0 = float(sig[-1])
    P = float(df["close"].iloc[-1])
    lr = np.log(df["close"].to_numpy(float))
    dlr = np.r_[np.nan, np.diff(lr)]
    r_pct = np.nan_to_num(dlr * 100)
    gp = _garch(r_pct[np.searchsorted(df.index, pd.Timestamp(START)):])
    cal = {"hl": calibrate(df, sig, D, hl=True), "close": calibrate(df, sig, D, hl=False)}
    best = {d: max(c, key=lambda m: c[m]["skill_pre"]) for d, c in cal.items()}    # wybór tylko na danych sprzed egzaminu
    b_hl = cal["hl"][best["hl"]]
    use_hl = b_hl["skill_pre"] > 0 and b_hl["skill_post"] > 0
    dfn = "hl" if use_hl else "close"
    mode = best[dfn]
    mu = _mu(dlr, len(df) - 1)
    sims = {H: _paths(s0, s_lr, pool, H, n, mode, gp, mu) for H in horizons}
    table = []
    for L in levels(P):
        row = {"level": L, "rel": L / P - 1}
        for H in horizons:
            row[f"p{H}"] = probs(sims[H], [L / P - 1], use_hl)[0]
        table.append(row)
    rels = [-0.2, -0.1, 0.1, 0.2]
    simple = {H: dict(zip(rels, probs(sims[H], rels, use_hl))) for H in horizons}
    strip = lambda c: {m: {k: v for k, v in x.items() if k != "rows"} for m, x in c.items()}
    return {"price": P, "table": table, "simple": simple, "sigma_now": s0, "sigma_lr": s_lr,
            "definition": dfn, "best": mode, "calib": cal[dfn][mode], "variants": strip(cal[dfn]),
            "other_definition": {"definition": "close" if use_hl else "hl", "best": best["close" if use_hl else "hl"],
                                 "variants": strip(cal["close" if use_hl else "hl"])}}


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


def calibrate(df, sig, D, hl=True, H=30, rels=(-0.2, -0.1, 0.1, 0.2), n=3000, step=7):
    """Test wstecz wszystkich wariantów: przewidywana szansa dotknięcia vs to, co się stało. Tylko dane znane w dniu prognozy.
    hl=True: dotknięcie po maksimach/minimach dnia; hl=False: tylko po zamknięciach."""
    lp_all = np.log(df["close"].to_numpy(float))
    hi_all = np.log(df["high"].to_numpy(float)) if hl and "high" in df else lp_all
    lo_all = np.log(df["low"].to_numpy(float)) if hl and "low" in df else lp_all
    dlr = np.r_[np.nan, np.diff(lp_all)]
    r_pct = np.nan_to_num(dlr * 100)
    idx = df.index
    s_hist = pd.Series(sig, idx)
    first = int(np.searchsorted(idx, pd.Timestamp("2016-01-01")))
    i0 = int(np.searchsorted(idx, pd.Timestamp(START)))
    gp, next_fit = None, 0
    acc = {m: {"pred": [], "hit": [], "which": [], "when": []} for m in VARIANTS}
    okD = np.isfinite(D).all(axis=1)
    for t in range(first, len(df) - H, step):
        pool = D[: t + 1][okD[: t + 1]]
        if len(pool) < 500:
            continue
        if gp is None or t >= next_fit:                 # GARCH dopasowany tylko do przeszłości, odświeżany co rok
            gp = _garch(r_pct[i0:t + 1])
            next_fit = t + 365
        s_lr = float(s_hist.iloc[:t + 1].loc[START:].median())
        mu = _mu(dlr, t)
        fh = np.nanmax(hi_all[t + 1:t + H + 1]) - lp_all[t]
        fl = np.nanmin(lo_all[t + 1:t + H + 1]) - lp_all[t]
        hits = [float(fh >= np.log(1 + x)) if x > 0 else float(fl <= np.log(1 + x)) for x in rels]
        for m in VARIANTS:
            ext = _paths(float(sig[t]), s_lr, pool, H, n, m, gp, mu)
            for x, pr, h in zip(rels, probs(ext, list(rels), hl), hits):
                a = acc[m]
                a["pred"].append(pr); a["hit"].append(h); a["which"].append(x); a["when"].append(t)
    out = {}
    cut = int(np.searchsorted(idx, HOLDOUT))
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
        pre, post = when + H < cut, when >= cut                # egzamin: wybór na „pre”, sprawdzian na „post”
        sk = lambda m_: float(1 - np.mean((pred[m_] - hit[m_]) ** 2) / np.mean((base[m_] - hit[m_]) ** 2)) if m_.sum() > 20 else np.nan
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
                  "side": side, "reliability": rel, "resolution": res, "uncertainty": unc, "reliability_base": rel0, "resolution_base": res0,
                  "skill_pre": sk(pre), "skill_post": sk(post), "n_post": int(post.sum())}
    return out
