"""Zespół modeli rozkładu przyszłego zwrotu BTC z uczciwą oceną walk-forward.

Modele (wszystkie prognozują kwantyle standaryzowanego zwrotu z = ln(P[t+H]/P[t]) / (sigma_t * sqrt(H))):
  naive  - bez zmian: empiryczny rozkład historycznych z, wyśrodkowany na zero (benchmark)
  comp   - regresja na wskaźniku złożonym + empiryczne reszty
  ridge  - regresja grzbietowa na ~20 cechach (wskaźniki, momentum, reżimy HMM, power law, zdarzenia, nastroje)
  knn    - analogi: podobne dni z przeszłości i ich rzeczywiste zwroty
  gbq    - gradient boosting z funkcją straty kwantylowej
Zmienność (sigma) to mieszanka EWMA, GJR-GARCH i HAR na estymatorze Parkinsona (wagi z QLIKE).
Zespół: wagi online z dotychczasowego pinball loss (tylko przeszłość), a na końcu kalibracja pokrycia pasma.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import t as tdist
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
import events as EV
import model as M

warnings.filterwarnings("ignore")
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
MODELS = ["naive", "comp", "ridge", "knn", "gbq", "scan", "conf"]
LABELS = {"naive": "Bez zmian (benchmark)", "comp": "Wskaźnik złożony", "ridge": "Regresja grzbietowa",
          "knn": "Analogi historyczne", "gbq": "Gradient boosting kwantylowy", "scan": "Skaner świata (selekcja cech)",
          "conf": "Potwierdzone sygnały (test z góry)", "ens": "Zespół (ensemble)"}
GENESIS = pd.Timestamp("2009-01-03")


# ------------------------------------------------------------------ zmienność
def garch_gjr(x):
    """GJR-GARCH(1,1), QMLE (Gaussian). x: zwroty w %. Zwraca wariancję prognozowaną na następny dzień."""
    n = len(x)
    v0 = np.var(x[:60])

    def run(p):
        w, a, g, b = p
        h = np.empty(n)
        h[0] = v0
        for t in range(1, n):
            h[t] = w + (a + g * (x[t - 1] < 0)) * x[t - 1] ** 2 + b * h[t - 1]
        return h

    def nll(p):
        if p[0] <= 0 or p[1] < 0 or p[3] < 0 or p[1] + p[3] + 0.5 * p[2] >= 0.9995 or p[1] + p[2] < 0:
            return 1e10
        h = run(p)
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    res = minimize(nll, [0.1, 0.08, 0.04, 0.88], method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-3})
    w, a, g, b = res.x
    h = run(res.x)
    hn = w + (a + g * (x < 0)) * x ** 2 + b * h          # prognoza wariancji na dzień t+1 (info do t)
    return hn, res.x


def vol_models(df):
    c = df["close"]
    r = np.log(c).diff().to_numpy().copy()
    r[0] = 0.0
    ew = np.sqrt((pd.Series(r ** 2).ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean()).to_numpy())
    hn, gjr = garch_gjr(r * 100)
    sg = np.sqrt(hn) / 100
    # HAR na estymatorze Parkinsona (wysoki/niski dzień)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    park = np.log(hi / lo) ** 2 / (4 * np.log(2))
    park = pd.Series(np.where(np.isfinite(park) & (park > 0), park, np.nan)).ffill().bfill()
    scale = np.nanmean(r[30:] ** 2) / park.mean()
    pv = park * scale
    d, w, m = np.log(pv + 1e-12), np.log(pv.rolling(5).mean() + 1e-12), np.log(pv.rolling(22).mean() + 1e-12)
    y = np.log(pv.rolling(5).mean().shift(-5) + 1e-12)
    Xh = np.column_stack([np.ones(len(d)), d, w, m])
    ok = np.isfinite(Xh).all(axis=1) & np.isfinite(y.to_numpy())
    beta = np.linalg.lstsq(Xh[ok], y.to_numpy()[ok], rcond=None)[0]
    pred = Xh @ beta
    resid = y.to_numpy()[ok] - Xh[ok] @ beta
    har = np.sqrt(np.exp(pred + 0.5 * resid.var()))
    # wagi z QLIKE względem zrealizowanej wariancji z następnych 5 dni
    rv = pd.Series(r ** 2).rolling(5).mean().shift(-5).to_numpy()
    cand = np.column_stack([ew ** 2, sg ** 2, har ** 2])
    sel = np.isfinite(rv).copy()
    sel[:1000] = False
    sel &= np.isfinite(cand).all(axis=1) & (rv > 0)
    best, bw = 1e9, (1, 0, 0)
    for a in np.arange(0, 1.01, 0.25):
        for b in np.arange(0, 1.01 - a, 0.25):
            wv = (a, b, 1 - a - b)
            s2 = cand[sel] @ np.array(wv)
            q = np.mean(rv[sel] / s2 - np.log(rv[sel] / s2) - 1)
            if q < best:
                best, bw = q, wv
    sig = np.sqrt(cand @ np.array(bw))
    sig = pd.Series(sig).ffill().to_numpy()
    info = {"weights": dict(zip(["ewma", "garch", "har"], map(float, bw))), "qlike": float(best),
            "phi": float(np.clip(gjr[1] + gjr[3] + 0.5 * gjr[2], 0.5, 0.999))}     # trwałość szoków zmienności (GJR-GARCH)
    # zmienność z opcji (DVOL, Deribit, od 2021): rynek opcji wycenia przyszłe wahania. Opcje zwykle je przeszacowują,
    # więc skalę liczymy tylko z przeszłości: mediana (zrealizowana wariancja 5 dni) / (DVOL sprzed 5 dni)^2
    if "dvol" in df and df["dvol"].notna().sum() > 400:
        dv2 = (df["dvol"].ffill(limit=3).to_numpy(float) / 100) ** 2 / 365
        known_rv = pd.Series(r ** 2).rolling(5).mean().to_numpy()                 # znana w dniu t
        ratio = pd.Series(known_rv / pd.Series(dv2).shift(5).to_numpy())
        k = ratio.where(np.isfinite(ratio) & (ratio > 0)).expanding(200).median().to_numpy()
        iv = k * dv2
        s3 = sig ** 2
        ok2 = sel & np.isfinite(iv)
        if ok2.sum() > 300:
            base = np.mean(rv[ok2] / s3[ok2] - np.log(rv[ok2] / s3[ok2]) - 1)
            bq, bwi = base, 0.0
            for wi in np.arange(0.1, 1.01, 0.1):
                s2 = (1 - wi) * s3[ok2] + wi * iv[ok2]
                q = np.mean(rv[ok2] / s2 - np.log(rv[ok2] / s2) - 1)
                if q < bq:
                    bq, bwi = q, float(wi)
            mix = np.where(np.isfinite(iv), (1 - bwi) * s3 + bwi * iv, s3)
            sig = np.sqrt(mix)
            info.update({"implied_w": bwi, "qlike_hist_only": float(base), "qlike_with_implied": float(bq), "implied_scale": float(k[-1]),
                         "implied_gain": float(1 - bq / base) if base > 0 else 0.0})
    sig, rinfo = _rv_blend(df, sig, rv)
    info.update(rinfo)
    return sig, info


def _qlike(target, s2):
    return float(np.mean(target / s2 - np.log(target / s2) - 1))


def _rv_blend(df, sig, target):
    """Zrealizowana zmienność z danych 5-minutowych (HAR-RV i HAR z semiwariancją i skokami, Corsi 2009; Patton i Sheppard 2015).
    Uczciwy test: model i waga dopasowane na pierwszych 70% okresu z danymi 5-min, ocena na ostatnich 30%.
    Do prognozy trafia tylko wtedy, gdy poprawia wynik na tej odłożonej części."""
    try:
        import rv as RVM
        R = RVM.load()
    except Exception:
        R = None
    if R is None or len(R) < 1000:
        return sig, {}
    idx = df.index
    R = R.reindex(idx)
    lg = lambda s: np.log(s.clip(lower=1e-10))
    rvd, rvw, rvm = R["rv"], R["rv"].rolling(5, min_periods=4).mean(), R["rv"].rolling(22, min_periods=18).mean()
    negw = R["rs_neg"].rolling(5, min_periods=4).mean()
    jw = R["jump"].rolling(5, min_periods=4).mean()
    X1 = pd.concat([lg(rvd), lg(rvw), lg(rvm)], axis=1)
    X2 = pd.concat([lg(rvd), lg(rvw), lg(rvm), lg(negw), np.sqrt(jw)], axis=1)
    y = pd.Series(target, idx)
    s3 = sig ** 2
    ok = X2.notna().all(axis=1) & y.notna() & (y > 0) & np.isfinite(s3)
    pos = np.where(ok.to_numpy())[0]
    if len(pos) < 800:
        return sig, {}
    split = pos[int(len(pos) * 0.7)]
    tr = ok & (np.arange(len(idx)) < split)
    te = ok & (np.arange(len(idx)) >= split)

    def fit(X, mask):
        A = np.column_stack([np.ones(mask.sum()), X[mask].to_numpy()])
        b = np.linalg.lstsq(A, np.log(y[mask].to_numpy()), rcond=None)[0]
        res = np.log(y[mask].to_numpy()) - A @ b
        return b, res.var()

    def pred(X, b, v):
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        return np.exp(A @ b + 0.5 * v)

    trm, tem = tr.to_numpy(), te.to_numpy()
    yt = y.to_numpy()
    best = {"name": None, "w": 0.0, "q_tr": _qlike(yt[trm], s3[trm])}
    cand = {}
    for name, X in (("har_rv", X1), ("har_rsj", X2)):
        b, v = fit(X, tr)
        p = np.full(len(idx), np.nan)
        m = X.notna().all(axis=1).to_numpy()
        p[m] = pred(X[m], b, v)
        cand[name] = (p, X)
        for w in np.arange(0.1, 1.01, 0.1):
            mix = (1 - w) * s3[trm] + w * p[trm]
            q = _qlike(yt[trm], mix)
            if q < best["q_tr"]:
                best = {"name": name, "w": float(w), "q_tr": q}
    q_old = _qlike(yt[tem], s3[tem])
    out = {"rv_q_test_old": q_old, "rv_n_test": int(tem.sum()), "rv_test_from": str(idx[split].date())}
    for name, (p, _) in cand.items():
        out[f"rv_q_test_{name}"] = _qlike(yt[tem], p[tem])
    if best["name"] is None:
        out.update({"rv_w": 0.0, "rv_gain_test": 0.0})
        return sig, out
    p = cand[best["name"]][0]
    q_new = _qlike(yt[tem], (1 - best["w"]) * s3[tem] + best["w"] * p[tem])
    gain = 1 - q_new / q_old
    out.update({"rv_model": best["name"], "rv_w": best["w"], "rv_gain_test": float(gain)})
    if gain <= 0:                                      # na odłożonej części nie pomaga: nie używamy
        out["rv_w"] = 0.0
        return sig, out
    # do bieżącej prognozy: ten sam model dopasowany na całej historii z danymi 5-min, ta sama waga
    b, v = fit(cand[best["name"]][1], ok)
    X = cand[best["name"]][1]
    pf = np.full(len(idx), np.nan)
    m = X.notna().all(axis=1).to_numpy()
    pf[m] = pred(X[m], b, v)
    mixed = np.where(np.isfinite(pf), (1 - best["w"]) * s3 + best["w"] * pf, s3)
    return pd.Series(np.sqrt(mixed)).ffill().to_numpy(), out


# ------------------------------------------------------------------ HMM (3 stany, filtrowany, bez podglądania)
def _emis(x, m, v):
    return np.exp(-0.5 * (x[:, None] - m[None, :]) ** 2 / v[None, :]) / np.sqrt(2 * np.pi * v[None, :]) + 1e-300


def _forward(B, A, pi):
    n, K = B.shape
    al = np.empty((n, K))
    c = np.empty(n)
    a = pi * B[0]
    c[0] = a.sum()
    al[0] = a / c[0]
    for t in range(1, n):
        a = (al[t - 1] @ A) * B[t]
        c[t] = a.sum()
        al[t] = a / c[t]
    return al, c


def hmm_fit(x, K=3, iters=25, init=None):
    if init is None:
        q = np.quantile(np.abs(x), [0.33, 0.66])
        v = np.array([np.var(x[np.abs(x) <= q[0]]), np.var(x[(np.abs(x) > q[0]) & (np.abs(x) <= q[1])]), np.var(x[np.abs(x) > q[1]])]) + 1e-8
        m, A, pi = np.zeros(K), np.full((K, K), 0.04) + np.eye(K) * 0.88, np.full(K, 1 / K)
    else:
        m, v, A, pi = init
    for _ in range(iters):
        B = _emis(x, m, v)
        al, c = _forward(B, A, pi)
        n = len(x)
        be = np.empty_like(al)
        be[-1] = 1
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        xi = A * (al[:-1].T @ (B[1:] * be[1:] / c[1:, None]))
        A = xi / xi.sum(axis=1, keepdims=True)
        pi = g[0]
        m = (g * x[:, None]).sum(0) / g.sum(0)
        v = (g * (x[:, None] - m[None, :]) ** 2).sum(0) / g.sum(0) + 1e-8
    order = np.argsort(v)
    return m[order], v[order], A[np.ix_(order, order)], pi[order]


def hmm_features(df):
    """P(stan wysokiej zmienności) i oczekiwany dzienny zwrot z filtra, parametry z ekspandującego okna."""
    r = np.log(df["close"]).diff().fillna(0).to_numpy()
    n = len(r)
    p_hi, e_mu = np.full(n, np.nan), np.full(n, np.nan)
    pts = list(range(1100, n, 365)) + [n]
    prm = None
    for k, end in enumerate(pts[:-1]):
        prm = hmm_fit(r[:end], init=prm, iters=25 if prm is None else 10)
        m, v, A, pi = prm
        seg_end = pts[k + 1]
        al, _ = _forward(_emis(r[:seg_end], m, v), A, pi)
        p_hi[end:seg_end] = al[end:seg_end, -1]
        e_mu[end:seg_end] = al[end:seg_end] @ m
    return p_hi, e_mu, prm


# ------------------------------------------------------------------ power law
def powerlaw_z(df):
    x = np.log10((df.index - GENESIS).days.to_numpy(float))
    y = np.log10(df["close"].to_numpy(float))
    n = len(x)
    cn = np.arange(1, n + 1)
    sx, sy, sxx, sxy, syy = np.cumsum(x), np.cumsum(y), np.cumsum(x * x), np.cumsum(x * y), np.cumsum(y * y)
    z = np.full(n, np.nan)
    for t in range(500, n):
        k = cn[t]
        b = (k * sxy[t] - sx[t] * sy[t]) / (k * sxx[t] - sx[t] ** 2)
        a = (sy[t] - b * sx[t]) / k
        sse = syy[t] - a * sy[t] - b * sxy[t]
        sd = np.sqrt(max(sse, 1e-12) / (k - 2))
        z[t] = (y[t] - (a + b * x[t])) / sd
    return z


# ------------------------------------------------------------------ macierz cech
def _expz(s, minp=365):
    s = pd.Series(s)
    mu, sd = s.expanding(minp).mean(), s.expanding(minp).std()
    return ((s - mu) / sd).clip(-4, 4)


def build_features(df, f, sc, comp, ev, sig, p_hi, e_mu, pl, reg=None):
    idx = df.index
    X = pd.DataFrame(index=idx)
    if reg is not None:
        rm = reg["rolling"]
        for k in ("hurst", "pent", "vr5", "acabs", "hill"):
            X[k] = rm[k].to_numpy()
        X["kalman"] = reg["kalman"]["series"].reindex(idx, method="ffill").to_numpy()
    X["comp"] = comp
    X["val"] = sc[["mvrv", "ma200w", "mayer", "puell"]].mean(axis=1)
    X["macro"] = sc[["m2", "dxy", "ndx", "vix", "rates"]].mean(axis=1)
    X["sent"] = sc[["fng", "funding", "stable"]].mean(axis=1)
    X["cycle"] = sc["halving"]
    lc = np.log(df["close"])
    X["mom30"] = (lc - lc.shift(30)) / (pd.Series(sig, idx) * np.sqrt(30))
    X["mom90"] = (lc - lc.shift(90)) / (pd.Series(sig, idx) * np.sqrt(90))
    X["volpct"] = pd.Series(sig, idx).rolling(365).apply(lambda a: (a[-1] > a).mean(), raw=True)
    X["powerlaw"] = pl
    X["hmm_hi"] = p_hi
    X["hmm_mu"] = e_mu
    sh = ev[(ev["kind"] == "auto") & ev["cat"].str.startswith("auto_")]
    cnt = lambda cats, w: pd.Series(0.0, idx).add(
        sh[sh["cat"].isin(cats)].groupby("date").size().reindex(idx, fill_value=0), fill_value=0).rolling(w).sum()
    X["ev_eqdn30"] = cnt(["auto_eq_dn"], 30)
    X["ev_shock30"] = cnt([c for c in sh["cat"].unique()], 30)
    X["ev_hack30"] = pd.Series(ev[ev["cat"] == "hack"].groupby("date").size().reindex(idx, fill_value=0), idx).rolling(30).sum()
    X["stress"] = (df["vix"] > 25).astype(float) if "vix" in df else 0.0
    soon = lambda cat: pd.Series([float(((ev["cat"] == cat) & (ev["date"] >= d) & (ev["date"] <= d + pd.Timedelta(days=3))).any()) for d in idx], idx)
    X["fomc_soon"] = soon("fomc")
    X["opex_soon"] = soon("opex")
    if {"btckrw", "usdkrw"} <= set(df.columns):
        kim = df["btckrw"] / df["usdkrw"] / df["close"] - 1
        X["kimchi"] = kim.rolling(7).mean()
    if "wiki_en" in df:
        w = {k: np.log(df[k].rolling(30).mean() / df[k].rolling(365, min_periods=200).mean()) for k in df.columns if k.startswith("wiki_")}
        X["wiki_west"] = pd.concat([w[k] for k in ("wiki_en", "wiki_de", "wiki_fr", "wiki_es") if k in w], axis=1).mean(axis=1)
        X["wiki_asia"] = pd.concat([w[k] for k in ("wiki_ja", "wiki_ko", "wiki_zh") if k in w], axis=1).mean(axis=1)
    if "dvol" in df:
        X["dvol_vrp"] = np.log(df["dvol"] / (pd.Series(sig, idx) * np.sqrt(365) * 100))
    if "cot_lev" in df:
        X["cot"] = df["cot_lev"]
    binary = {"stress", "fomc_soon", "opex_soon"}
    Z = pd.DataFrame({c: (X[c] if c in binary else _expz(X[c], 365)) for c in X.columns})
    Z = Z.clip(-4, 4)
    return X, Z


# ------------------------------------------------------------------ modele
def _pinball(y, q):
    d = y[:, None] - q
    return np.maximum(QS * d, (QS - 1) * d).mean(axis=1)


def _wq(v, w, qs):
    """Kwantyle ważone (nowsze obserwacje mają większą wagę)."""
    o = np.argsort(v)
    v, w = v[o], w[o]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cum, v)


HALF_LIFE = 730.0


def _ridge_fit(X, z, lam=0.25):
    mx, mz = X.mean(0), z.mean()
    Xc = X - mx
    beta = np.linalg.solve(Xc.T @ Xc / len(X) + lam * np.eye(X.shape[1]), Xc.T @ (z - mz) / len(X))
    return mx, mz, beta


def walk_forward(Zv, zt, H, N, comp_col, knn_cols, Zraw_comp, dates=None, P=None, Wz=None, conf=None):
    """Zwraca {model: (nT,7)}, T (indeksy), y (nT), info (wyjaśnienie bieżącej prognozy)."""
    info = {}
    Wa = np.nan_to_num(Wz.to_numpy(float), nan=0.0) if Wz is not None else None
    Wvalid = Wz.notna().to_numpy() if Wz is not None else None
    sel, next_sel = None, 0
    step = _step(H)
    T = list(range(C.WF_START, N - H, step))
    T_all = T + [N - 1]
    out = {m: np.full((len(T_all), 7), np.nan) for m in MODELS}
    yv = np.array([zt[t] if t < N - H else np.nan for t in T_all])
    valid_row = np.isfinite(zt)
    gb_models, next_refit = None, 0
    kcols = [i for i, c in enumerate(Zv.columns) if c in knn_cols]
    Xa = Zv.to_numpy(float)
    Xa = np.nan_to_num(Xa, nan=0.0)
    comp_a = np.nan_to_num(Zraw_comp, nan=0.0)
    for j, t in enumerate(T_all):
        pool = np.arange(0, max(t - H + 1, 0))
        pool = pool[valid_row[pool] & (pool >= 400)]
        if len(pool) < C.MIN_TRAIN:
            continue
        z, Xp = zt[pool], Xa[pool]
        wt = 0.5 ** ((t - pool) / HALF_LIFE)
        med = _wq(z, wt, [0.5])[0]
        out["naive"][j] = _wq(z - med, wt, QS)
        # comp
        cp = comp_a[pool]
        b = np.cov(cp, z)[0, 1] / (cp.var() * (1 + C.SLOPE_RIDGE) + 1e-12)
        res = z - b * (cp - cp.mean())
        out["comp"][j] = b * (comp_a[t] - cp.mean()) + _wq(res - _wq(res, wt, [0.5])[0], wt, QS)
        # sygnały, które przeszły test potwierdzający (spisany z góry): ta sama prosta regresja, tylko na znanych w t danych
        if conf is not None and np.isfinite(conf[t]) and np.isfinite(conf[pool]).sum() >= C.MIN_TRAIN:
            okc = np.isfinite(conf[pool])
            cx, cz, cw = conf[pool][okc], z[okc], wt[okc]
            bc = np.cov(cx, cz)[0, 1] / (cx.var() * (1 + C.SLOPE_RIDGE) + 1e-12)
            rc = cz - bc * (cx - cx.mean())
            out["conf"][j] = bc * (conf[t] - cx.mean()) + _wq(rc - _wq(rc, cw, [0.5])[0], cw, QS)
            if j == len(T_all) - 1:
                info["conf"] = {"beta": float(bc), "x": float(conf[t] - cx.mean())}
        else:
            out["conf"][j] = out["naive"][j]
        # ridge
        mx, mz, beta = _ridge_fit(Xp, z)
        res = z - (mz + (Xp - mx) @ beta)
        pred = (Xa[t] - mx) @ beta
        out["ridge"][j] = pred + _wq(res - _wq(res, wt, [0.5])[0], wt, QS)
        # knn (analogi)
        d = np.sqrt(((Xp[:, kcols] - Xa[t, kcols]) ** 2).sum(1))
        kk = min(max(60, len(pool) // 25), len(pool))
        nn = np.argpartition(d, kk - 1)[:kk]
        out["knn"][j] = np.quantile(z[nn], QS)
        if j == len(T_all) - 1 and dates is not None:
            contrib = (Xa[t] - mx) * beta
            info["ridge"] = sorted(zip(Zv.columns, contrib.tolist(), (Xa[t] - mx).tolist()), key=lambda a: -abs(a[1]))[:8]
            # najbliższe analogi rozrzucone w czasie (min. 30 dni odstępu)
            order, picked = np.argsort(d), []
            for o in order:
                if all(abs(pool[o] - pool[p]) > 30 for p in picked):
                    picked.append(o)
                if len(picked) == 8:
                    break
            info["knn"] = [(dates[pool[o]].strftime("%Y-%m-%d"), float((P[pool[o] + H] / P[pool[o]] - 1) * 100), float(d[o])) for o in picked]
            nd = np.sort(pool[nn])
            info["knn_all"] = {"n": int(kk), "up": float(np.mean(z[nn] > 0)), "median_ret": float(np.median([(P[pool[o] + H] / P[pool[o]] - 1) * 100 for o in nn])),
                               "episodes": int(1 + np.sum(np.diff(nd) > 45))}
        # gradient boosting kwantylowy (refit raz na 365 dni)
        if gb_models is None or t >= next_refit:
            zc = np.clip(z, -6, 6)
            gb_models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=60, max_depth=3, learning_rate=0.06,
                                                       min_samples_leaf=60, l2_regularization=1.0, random_state=0).fit(Xp, zc) for q in QS]
            next_refit = t + 365
        out["gbq"][j] = np.sort([m.predict(Xa[t:t + 1])[0] for m in gb_models])
        # skaner świata: selekcja najsilniejszych cech z setek szeregów (tylko na danych znanych w t, odświeżana co rok)
        if Wa is not None:
            if sel is None or t >= next_sel:
                zc = (z - z.mean()) / (z.std() + 1e-9)
                cnt = Wvalid[pool].sum(0)
                cor = np.abs((Wa[pool] * zc[:, None]).mean(0))
                cor[cnt < 500] = 0
                sel = np.argsort(-cor)[:12]
                next_sel = t + 365
            Xs = Wa[pool][:, sel]
            mxs, mzs, bs = _ridge_fit(Xs, z, lam=1.0)
            rs = z - (mzs + (Xs - mxs) @ bs)
            out["scan"][j] = (Wa[t, sel] - mxs) @ bs + _wq(rs - _wq(rs, wt, [0.5])[0], wt, QS)
            if j == len(T_all) - 1 and dates is not None:
                info["scan"] = [(Wz.columns[i], float(((Wa[t, sel] - mxs) * bs)[k]), float(Wa[t, i])) for k, i in enumerate(sel)]
        else:
            out["scan"][j] = out["naive"][j]
    return out, np.array(T_all), yv, info


def _step(H):
    """Krok testu „jak na żywo”: krótkie horyzonty co dzień (7× więcej prób), długie co tydzień (koszt obliczeń).
    Zmiana z 2026-10-30 (punkt 4 planu); wcześniej wszystkie horyzonty co 7 dni."""
    return 1 if H <= 7 else C.WF_STEP


def _dm(l0, l1, lag):
    """Test Diebolda-Mariano (jednostronny: czy l1 ma mniejszą stratę), błąd Neweya-Westa z oknem lag,
    z poprawką Harveya, Leybourne'a i Newbolda (1997) dla małych prób i rozkładem t (od 2026-10-30)."""
    d = l0 - l1
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return np.nan, np.nan
    n = len(d)
    mu = d.mean()
    dc = d - mu
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k] / n)
    t = mu / np.sqrt(max(var, 1e-12) / n)
    h = max(lag, 1)
    t *= np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1.0) / n)
    return float(t), float(1 - tdist.cdf(t, n - 1))


def _mde(l0, l1, lag, alpha=0.10, power=0.80):
    """Najmniejsza przewaga (ułamek straty „bez zmian”), którą ten test wykryłby w 4 na 5 przypadków przy danej liczbie prób."""
    d = l0 - l1
    ok = np.isfinite(d)
    d, l0 = d[ok], l0[ok]
    if len(d) < 30:
        return np.nan
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k] / n)
    se = np.sqrt(max(var, 1e-12) / n)
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * se / l0.mean())


def calibrate(q, y, lag):
    """Online: skala pasma tak, by 80% przeszłych obserwacji mieściło się w [P10, P90]. Tylko przeszłość."""
    nT = len(q)
    cal = np.ones(nT)
    grid = np.arange(0.6, 2.01, 0.05)
    for k in range(nT):
        if not np.isfinite(q[k]).all():
            continue
        j = np.arange(0, k - lag + 1)
        j = j[np.isfinite(y[j]) & np.isfinite(q[j]).all(axis=1)] if len(j) else j
        if len(j) < 40:
            continue
        m = q[j, 3]
        fr = np.array([np.mean((y[j] >= m + s * (q[j, 1] - m)) & (y[j] <= m + s * (q[j, 5] - m))) for s in grid])
        cal[k] = grid[np.argmin(np.abs(fr - 0.8))]
    return q[:, [3]] + cal[:, None] * (q - q[:, [3]]), cal


def combine(out, y, H):
    """Zespół z wagami online (tylko przeszłość) + kalibracja pokrycia. Ostatni wiersz to prognoza bieżąca."""
    nT = len(y)
    lag = int(np.ceil(H / _step(H))) + 1
    raw = out
    out = {}
    for m in MODELS:
        out[m], _ = calibrate(raw[m], y, lag)          # każdy model skalibrowany osobno (uczciwe porównanie)
    L = {m: np.where(np.isfinite(y), _pinball(np.nan_to_num(y), out[m]), np.nan) for m in MODELS}
    ens = np.full((nT, 7), np.nan)
    wts = np.zeros((nT, len(MODELS)))
    alphas = np.zeros(nT)
    for k in range(nT):
        if not np.isfinite(out["naive"][k]).all():
            continue
        j = np.arange(0, k - lag + 1)
        j = j[np.isfinite(y[j]) & np.isfinite(L["naive"][j])] if len(j) else j
        base = out["naive"][k]
        if len(j) < 40:
            ens[k] = base
            wts[k, 0] = 1
            continue
        l0 = L["naive"][j].sum()
        sk = {m: 1 - np.nansum(L[m][j]) / l0 for m in MODELS[1:]}
        pos = {m: max(s, 0.0) for m, s in sk.items()}
        best = max(pos.values())
        if best <= 0:
            ens[k] = base
            wts[k, 0] = 1
            continue
        w = {m: pos[m] ** 2 for m in pos}
        sw = sum(w.values())
        mix = sum(w[m] / sw * out[m][k] for m in w if np.isfinite(out[m][k]).all())
        bm = max(pos, key=pos.get)
        _, pv = _dm(L["naive"][j], L[bm][j], lag)
        gate = float(np.clip((0.10 - pv) / 0.07, 0, 1)) if np.isfinite(pv) else 0.0   # wymaga istotności statystycznej
        alpha = min(1.0, best / 0.02) * gate
        if alpha <= 0:
            ens[k] = base
            wts[k, 0] = 1
            continue
        ens[k] = (1 - alpha) * base + alpha * mix
        alphas[k] = alpha
        wts[k, 0] = 1 - alpha
        for i, m in enumerate(MODELS[1:], 1):
            wts[k, i] = alpha * w[m] / sw
    cal = np.array([float(np.nanmedian(np.array([1.0])))] * nT)     # składniki są już skalibrowane osobno
    return ens, ens, wts, alphas, cal, L, out


def aci(q, y, lag, gamma=0.05):
    """Adaptacyjne pasma (Gibbs i Candès 2021, ACI): szerokość pasma rośnie, gdy ostatnio cena częściej z niego wypadała,
    i maleje, gdy mieściła się częściej niż powinna. Uczy się tylko na prognozach, których termin już minął (opóźnienie lag)."""
    n = len(q)
    out = q.copy()
    ls = 0.0
    for k in range(n):
        j = k - lag
        if j >= 0 and np.isfinite(y[j]) and np.isfinite(q[j]).all():
            s = np.exp(ls)
            lo, hi = q[j, 3] + s * (q[j, 1] - q[j, 3]), q[j, 3] + s * (q[j, 5] - q[j, 3])
            miss = float(not (lo <= y[j] <= hi))
            ls = float(np.clip(ls + gamma * (miss - 0.2), -0.7, 0.7))
        if np.isfinite(q[k]).all():
            out[k] = q[k, [3]] + np.exp(ls) * (q[k] - q[k, [3]])
    return out


def _cov_err(q, y):
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    c80 = np.mean((y[ok] >= q[ok, 1]) & (y[ok] <= q[ok, 5]))
    c90 = np.mean((y[ok] >= q[ok, 0]) & (y[ok] <= q[ok, 6]))
    return abs(c80 - 0.8) + abs(c90 - 0.9), float(c80), float(c90)


HOLDOUT = pd.Timestamp("2024-10-29")
# Reguła egzaminu (spisana 2026-10-29, przed zobaczeniem wyników): wybór metody tylko na prognozach, których wynik był znany
# przed HOLDOUT; zwycięzca wchodzi do prognozy od dziś tylko, jeśli na odciętych 2 latach też bije „bez zmian” z p < 0,10 (DM, HAC).
HOLDOUT_P = 0.10


def holdout(out, cens, y, H, dates):
    lag = int(np.ceil(H / _step(H))) + 1
    eq = np.sort(np.nanmean(np.stack([out[m] for m in MODELS[1:]]), axis=0), axis=1)
    cand = {**{m: out[m] for m in MODELS}, "eq": eq, "ens": cens}
    d = pd.DatetimeIndex(dates)
    pre = (d + pd.Timedelta(days=H) < HOLDOUT) & np.isfinite(y)
    post = (d >= HOLDOUT) & np.isfinite(y)
    rows = {}
    for m, q in cand.items():
        okp = pre & np.isfinite(q).all(axis=1) & np.isfinite(out["naive"]).all(axis=1)
        oko = post & np.isfinite(q).all(axis=1) & np.isfinite(out["naive"]).all(axis=1)
        if okp.sum() < 30 or oko.sum() < 10:
            continue
        lp, lo = _pinball(np.nan_to_num(y), q), _pinball(np.nan_to_num(y), out["naive"])
        t_, p_ = _dm(lo[oko], lp[oko], lag) if m != "naive" else (np.nan, np.nan)
        rows[m] = {"skill_pre": float(1 - lp[okp].mean() / lo[okp].mean()), "skill_post": float(1 - lp[oko].mean() / lo[oko].mean()),
                   "p_post": p_, "n_pre": int(okp.sum()), "n_post": int(oko.sum()), "n_eff_post": float(oko.sum() / lag),
                   "mde_post": _mde(lo[oko], lp[oko], lag) if m != "naive" else np.nan,
                   "cov80_post": float(np.mean((y[oko] >= q[oko, 1]) & (y[oko] <= q[oko, 5])))}
    cands = {m: r for m, r in rows.items() if m != "naive"}
    if not cands:
        return {"rows": rows}
    best = max(cands, key=lambda m: cands[m]["skill_pre"])
    b = cands[best]
    passed = b["skill_pre"] > 0 and b["skill_post"] > 0 and np.isfinite(b["p_post"]) and b["p_post"] < HOLDOUT_P
    # mniej niż 20 niezależnych obserwacji na egzaminie: brak wyniku to „nierozstrzygnięte”, a nie „nie zdany”
    # (dopuszczenia do prognozy to nie zmienia: nierozstrzygnięte nadal znaczy „nie używamy”)
    verdict = "passed" if passed else ("inconclusive" if b["n_eff_post"] < 20 else "failed")
    return {"rows": rows, "chosen": best, "passed": bool(passed), "verdict": verdict, "from": str(HOLDOUT.date()), "q_live": cand[best][-1].tolist()}


def evaluate(out, cens, y, H, L):
    lag = int(np.ceil(H / _step(H))) + 1
    ok = np.isfinite(y) & np.isfinite(cens).all(axis=1) & np.isfinite(out["naive"]).all(axis=1)
    rows = []
    l0 = L["naive"][ok]
    cover = lambda q: float(np.mean((y[ok] >= q[ok, 1]) & (y[ok] <= q[ok, 5])))
    for m in MODELS:
        lm = L[m][ok]
        t, p = _dm(l0, lm, lag) if m != "naive" else (np.nan, np.nan)
        rows.append({"model": m, "label": LABELS[m], "loss": float(np.nanmean(lm)), "skill": float(1 - np.nanmean(lm) / np.nanmean(l0)),
                     "dm_t": t, "dm_p": p, "cover80": cover(out[m]), "mde": _mde(l0, lm, lag) if m != "naive" else np.nan})
    # prosta średnia wszystkich metod (1/N): w badaniach często wygrywa z wyszukanym ważeniem („forecast combination puzzle”)
    eq = np.sort(np.nanmean(np.stack([out[m] for m in MODELS[1:]]), axis=0), axis=1)
    okq = ok & np.isfinite(eq).all(axis=1)
    lq = _pinball(np.nan_to_num(y), eq)[okq]
    tq, pq = _dm(L["naive"][okq], lq, lag)
    rows.append({"model": "eq", "label": "Prosta średnia metod (1/N)", "loss": float(lq.mean()), "skill": float(1 - lq.mean() / L["naive"][okq].mean()),
                 "dm_t": tq, "dm_p": pq, "cover80": float(np.mean((y[okq] >= eq[okq, 1]) & (y[okq] <= eq[okq, 5])))})
    le = _pinball(np.nan_to_num(y), cens)[ok]
    t, p = _dm(l0, le, lag)
    rows.append({"model": "ens", "label": LABELS["ens"], "loss": float(le.mean()), "skill": float(1 - le.mean() / l0.mean()),
                 "dm_t": t, "dm_p": p, "cover80": cover(cens), "cover90": float(np.mean((y[ok] >= cens[ok, 0]) & (y[ok] <= cens[ok, 6])))})
    return rows, int(ok.sum())


# ------------------------------------------------------------------ całość
def horizon_sigma(sig, H, phi, s_lr):
    """Zmienność na H dni, która wraca do normy (Diebold i in. 1998): zamiast sigma_t·√H średnia z prognoz wariancji
    na kolejne dni, gdy dzisiejsza odchyłka od normy wygasa w tempie phi. Norma = mediana z ostatnich 2 lat (tylko przeszłość)."""
    k = np.arange(1, H + 1)
    w = (phi ** k).mean()                               # średni udział dzisiejszej odchyłki w kolejnych H dniach
    lr2 = np.where(np.isfinite(s_lr), s_lr ** 2, sig ** 2)
    return np.sqrt(H * (lr2 + (sig ** 2 - lr2) * w))


# Reguła (spisana 2026-10-30, przed zobaczeniem wyników): skalowanie z powrotem do normy zastępuje sigma·√H na danym horyzoncie,
# jeśli rozkład bazowy („bez zmian”) ma z nim mniejszą stratę pinball przed odcięciem (DM, HAC, p < 0,10) i dodatnią przewagę po nim.
def scale_test(P, sig, sigH, H, dates):
    N = len(P)
    y = np.full(N, np.nan)
    y[: N - H] = np.log(P[H:] / P[: N - H])
    s_old = sig * np.sqrt(H)
    z_old, z_new = y / s_old, y / sigH
    lag = int(np.ceil(H / _step(H))) + 1
    terc = pd.Series(sig).rolling(1095, min_periods=365)
    lo_q, hi_q = terc.quantile(1 / 3).to_numpy(), terc.quantile(2 / 3).to_numpy()
    rows = []
    for t in range(C.WF_START, N - H, _step(H)):
        pool = np.arange(400, max(t - H + 1, 400))
        pool = pool[np.isfinite(z_old[pool]) & np.isfinite(z_new[pool])]
        if len(pool) < C.MIN_TRAIN or not (np.isfinite(sigH[t]) and np.isfinite(y[t])):
            continue
        wt = 0.5 ** ((t - pool) / HALF_LIFE)
        qo = _wq(z_old[pool], wt, QS); qo = qo - qo[3]
        qn = _wq(z_new[pool], wt, QS); qn = qn - qn[3]
        yt = y[t] / s_old[t]                            # ta sama skala oceny dla obu wariantów
        po, pn = qo, qn * sigH[t] / s_old[t]
        st = 0 if sig[t] <= lo_q[t] else (2 if sig[t] > hi_q[t] else 1)
        rows.append((dates[t], _pinball(np.array([yt]), po[None, :])[0], _pinball(np.array([yt]), pn[None, :])[0],
                     float(po[1] <= yt <= po[5]), float(pn[1] <= yt <= pn[5]), st if np.isfinite(lo_q[t]) else -1))
    R = pd.DataFrame(rows, columns=["d", "lo", "ln", "co", "cn", "st"])
    if len(R) < 60:
        return {"used": False}
    cut = HOLDOUT
    pre, post = (R["d"] + pd.Timedelta(days=H) < cut).to_numpy(), (R["d"] >= cut).to_numpy()
    sk = lambda m: float(1 - R["ln"][m].mean() / R["lo"][m].mean()) if m.sum() > 5 else float("nan")
    _, p_pre = _dm(R["lo"][pre].to_numpy(), R["ln"][pre].to_numpy(), lag)
    _, p_post = _dm(R["lo"][post].to_numpy(), R["ln"][post].to_numpy(), lag)
    used = bool(sk(pre) > 0 and np.isfinite(p_pre) and p_pre < 0.10 and sk(post) > 0)
    terc_cov = {}
    for s, name in ((0, "spokojnie"), (1, "przeciętnie"), (2, "burzliwie")):
        m = (R["st"] == s).to_numpy()
        if m.sum() >= 10:
            terc_cov[name] = {"old": float(R["co"][m].mean()), "new": float(R["cn"][m].mean()), "n": int(m.sum())}
    return {"used": used, "skill_pre": sk(pre), "skill_post": sk(post), "p_pre": p_pre, "p_post": p_post,
            "n_pre": int(pre.sum()), "n_post": int(post.sum()), "terciles": terc_cov}


def run(df, ev, f, sc, comp, reg=None, world_T=None, conf=None, horizons=None):
    N = len(df)
    P = df["close"].to_numpy(float)
    sig, vinfo = vol_models(df)
    p_hi, e_mu, _ = hmm_features(df)
    pl = powerlaw_z(df)
    Xraw, Z = build_features(df, f, sc, comp, ev, sig, p_hi, e_mu, pl, reg)
    Wz = None
    if world_T is not None:
        keep = [c for c in world_T.columns if not c.endswith("|s") and not c.startswith(("s_", "g_"))]
        Wz = pd.DataFrame({c: _expz(world_T[c], 365) for c in keep}, index=df.index).clip(-4, 4)
    knn_cols = ["comp", "mom90", "volpct", "powerlaw", "hmm_hi", "cycle"]
    res = {"vol": vinfo, "horizons": {}, "forecasts": []}
    ca = None if conf is None else conf.reindex(df.index).to_numpy(float)
    s_lr = pd.Series(sig).rolling(730, min_periods=365).median().to_numpy()
    for H in (horizons or C.HORIZONS):
        sigH = horizon_sigma(sig, H, vinfo["phi"], s_lr)
        st = scale_test(P, sig, sigH, H, df.index)
        scale_arr = sigH if st.get("used") else sig * np.sqrt(H)
        zt = np.full(N, np.nan)
        zt[: N - H] = np.log(P[H:] / P[: N - H]) / scale_arr[: N - H]
        out, T, y, info = walk_forward(Z, zt, H, N, "comp", knn_cols, Z["comp"].to_numpy(float), df.index, P, Wz, ca)
        cens, ens, wts, alphas, cal, L, out = combine(out, y, H)
        lag = int(np.ceil(H / _step(H))) + 1
        qa = aci(cens, y, lag)
        e0, c80_0, c90_0 = _cov_err(cens, y)
        e1, c80_1, c90_1 = _cov_err(qa, y)
        ok_ = np.isfinite(y) & np.isfinite(cens).all(axis=1) & np.isfinite(qa).all(axis=1)
        pb0 = float(_pinball(np.nan_to_num(y[ok_]), cens[ok_]).mean())
        pb1 = float(_pinball(np.nan_to_num(y[ok_]), qa[ok_]).mean())
        use_aci = e1 < e0 and pb1 <= pb0 * 1.01               # lepsze pokrycie i nie gorsza ogólna trafność
        aci_info = {"used": bool(use_aci), "cov80_before": c80_0, "cov90_before": c90_0, "cov80_aci": c80_1, "cov90_aci": c90_1,
                    "pinball_before": pb0, "pinball_aci": pb1}
        if use_aci:
            cens = qa
        rows, nok = evaluate(out, cens, y, H, L)
        ho = holdout(out, cens, y, H, df.index[T])
        q = cens[-1]
        if ho.get("passed") and ho["chosen"] != "ens":           # zdał egzamin na odciętych 2 latach: wchodzi do prognozy od dziś
            q = np.array(ho["q_live"])
        scale = float(scale_arr[-1])
        prices = {f"p{int(a * 100)}": float(P[-1] * np.exp(scale * v)) for a, v in zip(QS, q)}
        # audyt: pokrycie końcowego pasma 80% w spokojnych, przeciętnych i burzliwych okresach (tercyle z ostatnich 3 lat)
        tq = pd.Series(sig).rolling(1095, min_periods=365)
        lo_q, hi_q = tq.quantile(1 / 3).to_numpy()[T], tq.quantile(2 / 3).to_numpy()[T]
        okc = np.isfinite(y) & np.isfinite(cens).all(axis=1) & np.isfinite(lo_q)
        stt = np.where(sig[T] <= lo_q, 0, np.where(sig[T] > hi_q, 2, 1))
        inb = (y >= cens[:, 1]) & (y <= cens[:, 5])
        audit = {nm: {"cov80": float(inb[okc & (stt == s)].mean()), "n": int((okc & (stt == s)).sum())}
                 for s, nm in ((0, "spokojnie"), (1, "przeciętnie"), (2, "burzliwie")) if (okc & (stt == s)).sum() >= 10}
        cdf0 = float(np.interp(0.0, q, QS, left=0.0, right=1.0)) if np.all(np.diff(q) > 0) else 0.5
        res["horizons"][H] = {"table": rows, "n": nok, "weights": dict(zip(MODELS, map(float, wts[-1]))), "alpha": float(alphas[-1]),
                              "calibration": float(cal[-1]), "live_z": {m: out[m][-1].tolist() for m in MODELS}, "ens_z": q.tolist(),
                              "explain": info, "sigma_H": scale, "aci": aci_info, "scale_test": st, "audit": audit,
                              "holdout": {k: v for k, v in ho.items() if k != "q_live"}}
        view = np.nanmean([out["comp"][-1], out["ridge"][-1]], axis=0)       # pogląd modeli wskaźników (bez potwierdzonej przewagi)
        vcdf0 = float(np.interp(0.0, view, QS, left=0.0, right=1.0)) if np.all(np.diff(view) > 0) else 0.5
        res["forecasts"].append({"horizon": H, "target": (df.index[-1] + pd.Timedelta(days=H)).strftime("%Y-%m-%d"),
                                 "prob_up": float(1 - cdf0), **prices,
                                 "view": {**{f"p{int(a * 100)}": float(P[-1] * np.exp(scale * v)) for a, v in zip(QS, view)}, "prob_up": float(1 - vcdf0)},
                                 "naive": {f"p{int(a * 100)}": float(P[-1] * np.exp(scale * v)) for a, v in zip(QS, out["naive"][-1])}})
    res["features_now"] = {c: (None if pd.isna(Xraw[c].iloc[-1]) else float(Xraw[c].iloc[-1])) for c in Xraw.columns}
    res["hmm"] = {"p_hi": float(p_hi[-1]), "p_hi_series": pd.Series(p_hi, df.index), "e_mu": float(e_mu[-1])}
    res["powerlaw_z"] = float(pl[-1])
    res["sigma"] = float(sig[-1])
    res["Xraw"] = Xraw
    return res
