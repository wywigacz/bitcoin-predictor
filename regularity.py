"""Regularność układu: czy zwroty BTC mają strukturę (pamięć, trend, grupowanie, ogony, cykle), a nie tylko szum.
Wszystkie miary liczone na oknach kroczących, wyłącznie z przeszłości danego dnia."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def hurst_dfa(r, scales=(8, 12, 16, 24, 32, 48, 64, 96)):
    """DFA-1 na zwrotach: alfa ~0,5 = szum, >0,5 = trwałość (trend), <0,5 = powrót do średniej."""
    y = np.cumsum(r - r.mean())
    F, S = [], []
    for s in scales:
        k = len(y) // s
        if k < 3:
            continue
        seg = y[: k * s].reshape(k, s)
        t = np.arange(s)
        res = []
        for v in seg:
            p = np.polyfit(t, v, 1)
            res.append(np.mean((v - np.polyval(p, t)) ** 2))
        F.append(np.sqrt(np.mean(res)))
        S.append(s)
    if len(S) < 4:
        return np.nan
    return float(np.polyfit(np.log(S), np.log(F), 1)[0])


def perm_entropy(r, m=4):
    """Znormalizowana entropia permutacyjna: 1 = czysty szum, niżej = więcej uporządkowanych wzorców."""
    n = len(r) - m + 1
    if n < 50:
        return np.nan
    idx = np.stack([r[i:i + n] for i in range(m)], axis=1)
    codes = np.argsort(idx, axis=1)
    key = (codes * (m ** np.arange(m))).sum(axis=1)
    _, cnt = np.unique(key, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / np.log(np.math.factorial(m) if hasattr(np, "math") else _fact(m)))


def _fact(m):
    out = 1
    for i in range(2, m + 1):
        out *= i
    return out


def variance_ratio(r, q=5):
    """Lo-MacKinlay: VR(q) - 1 i statystyka z (odporna na heteroskedastyczność). >0 trend, <0 powrót do średniej."""
    n = len(r)
    if n < q * 10:
        return np.nan, np.nan
    mu = r.mean()
    d = r - mu
    s1 = np.sum(d ** 2) / n
    rq = np.convolve(r, np.ones(q), "valid")
    sq = np.sum((rq - q * mu) ** 2) / (n * q)
    vr = sq / s1
    theta = 0.0
    for j in range(1, q):
        dj = np.sum((d[j:] ** 2) * (d[:-j] ** 2)) / (np.sum(d ** 2) ** 2) * n
        theta += (2 * (q - j) / q) ** 2 * dj
    z = (vr - 1) / np.sqrt(theta / n) if theta > 0 else np.nan
    return float(vr - 1), float(z)


def hill(r, frac=0.05):
    a = np.sort(np.abs(r))[::-1]
    k = max(int(len(a) * frac), 10)
    if k >= len(a):
        return np.nan
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def rolling_metrics(r, window=365, step=7):
    """Metryki kroczące; wartości między punktami uzupełniane do przodu (bez podglądania)."""
    n = len(r)
    cols = ["hurst", "pent", "vr5", "ac1", "acabs", "hill"]
    out = np.full((n, len(cols)), np.nan)
    for t in range(window, n, step):
        w = r[t - window:t + 1]
        v5, _ = variance_ratio(w, 5)
        ac1 = np.corrcoef(w[:-1], w[1:])[0, 1]
        aa = np.corrcoef(np.abs(w[:-1]), np.abs(w[1:]))[0, 1]
        out[t] = [hurst_dfa(w), perm_entropy(w), v5, ac1, aa, hill(w)]
    df = pd.DataFrame(out, columns=cols)
    return df.ffill(limit=step)


def kalman_trend(logp, q_ratio=None):
    """Lokalny model liniowy (poziom + nachylenie), filtr Kalmana na cenach tygodniowych. Zwraca nachylenie i jego niepewność."""
    y = np.asarray(logp, float)

    def run(par):
        sl, ss, so = np.exp(par)
        x = np.array([y[0], 0.0])
        P = np.diag([1.0, 0.01])
        F = np.array([[1, 1], [0, 1]], float)
        Q = np.diag([sl ** 2, ss ** 2])
        ll, xs, Ps = 0.0, [], []
        for v in y:
            x = F @ x
            P = F @ P @ F.T + Q
            S = P[0, 0] + so ** 2
            K = P[:, 0] / S
            e = v - x[0]
            x = x + K * e
            P = P - np.outer(K, P[0])
            ll += -0.5 * (np.log(S) + e * e / S)
            xs.append(x.copy())
            Ps.append(P[1, 1])
        return -ll, np.array(xs), np.array(Ps)

    res = minimize(lambda p: run(p)[0], np.log([0.05, 0.005, 0.02]), method="Nelder-Mead", options={"maxiter": 200})
    _, xs, Ps = run(res.x)
    return xs[:, 1], np.sqrt(Ps)


def markov(r7, state_cut=(-0.33, 0.33)):
    """Łańcuch Markowa na stanach tygodniowego zwrotu (spadek/neutralnie/wzrost). Test niezależności chi-kwadrat."""
    x = r7[np.isfinite(r7)]
    lo, hi = np.quantile(x, [0.33, 0.67])
    s = np.where(x < lo, 0, np.where(x > hi, 2, 1))
    M = np.zeros((3, 3))
    for a, b in zip(s[:-1], s[1:]):
        M[a, b] += 1
    P = M / M.sum(axis=1, keepdims=True)
    chi = stats.chi2_contingency(M + 0.5)
    return {"P": P.tolist(), "n": int(M.sum()), "p_dep": float(chi[1]), "now": int(s[-1]), "next": P[s[-1]].tolist()}


def evt_tail(r, years=4):
    """Teoria wartości ekstremalnych: GPD dla strat powyżej 95. percentyla z ostatnich lat. Dzienne VaR/ES."""
    x = -r[-365 * years:]
    u = np.quantile(x, 0.95)
    exc = x[x > u] - u
    if len(exc) < 20:
        return None
    c, loc, sc = stats.genpareto.fit(exc, floc=0)
    n, nu = len(x), len(exc)
    out = {}
    for p in (0.99, 0.999):
        var = u + sc / c * (((n / nu) * (1 - p)) ** (-c) - 1) if abs(c) > 1e-6 else u - sc * np.log((n / nu) * (1 - p))
        out[f"var{int(p * 1000)}"] = float(1 - np.exp(-var))
    out.update({"xi": float(c), "thr": float(1 - np.exp(-u)), "gauss99": float(1 - np.exp(-(x.mean() + 2.326 * x.std())))})
    return out


def spectrum(r, top=3):
    """Dominujące okresy w zmienności (periodogram log|r|), pomijając bardzo długie trendy."""
    v = np.log(np.abs(r) + 1e-4)
    v = v - pd.Series(v).rolling(365, min_periods=100, center=False).mean().bfill().to_numpy()
    v = v[~np.isnan(v)]
    f = np.fft.rfftfreq(len(v), 1.0)[1:]
    p = np.abs(np.fft.rfft(v - v.mean())[1:]) ** 2
    per = 1 / f
    ok = (per >= 5) & (per <= 400)
    f, p, per = f[ok], p[ok], per[ok]
    base = np.median(p)
    order = np.argsort(-p)
    picked = []
    for i in order:
        if all(abs(per[i] - q) / q > 0.15 for q, _ in picked):
            picked.append((float(per[i]), float(p[i] / base)))
        if len(picked) == top:
            break
    return picked


def run(df):
    c = df["close"].to_numpy(float)
    r = np.diff(np.log(c), prepend=np.nan)
    r = np.nan_to_num(r)
    rm = rolling_metrics(r)
    rm.index = df.index
    wk = df["close"].resample("W").last()
    slope, sd = kalman_trend(np.log(wk.to_numpy()))
    r7 = np.log(df["close"]).diff(7).to_numpy()[7::7]
    now = rm.dropna().iloc[-1]
    hist = rm.dropna()
    pct = {k: float((hist[k] < now[k]).mean()) for k in rm.columns}
    return {
        "rolling": rm, "now": {k: float(v) for k, v in now.items()}, "pct": pct,
        "kalman": {"slope_w": float(slope[-1]), "sd": float(sd[-1]), "z": float(slope[-1] / max(sd[-1], 1e-9)),
                   "series": pd.Series(slope / np.maximum(sd, 1e-9), index=wk.index)},
        "markov": markov(r7), "evt": evt_tail(r), "cycles": spectrum(r),
    }
