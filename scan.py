"""Skaner zależności: setki szeregów ze świata vs przyszłe zwroty BTC, z kontrolą wielokrotnych testów.

Problem: przy ~1900 testach ok. 5% (czyli ~95) wyjdzie "istotnych" z samego przypadku. Dlatego:
  1. p-wartości liczone konserwatywnie (efektywna liczba obserwacji = n / horyzont, bo zwroty nakładają się),
  2. korekta Benjaminiego-Hochberga (FDR, q = 0,10),
  3. stabilność: ten sam znak w obu połowach próby,
  4. test poza próbą: wybór na pierwszych 70% historii, ocena na ostatnich 30%.
Za "odkrycie" uznajemy tylko to, co przejdzie wszystko naraz.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as tdist

import world as W

HORIZONS = (7, 30)
LAGS = (0, 7, 30)


def _rank(a):
    out = np.full(len(a), np.nan)
    m = np.isfinite(a)
    out[m] = rankdata(a[m])
    return out


def _corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 120:
        return np.nan, int(m.sum())
    a, b = rankdata(x[m]), rankdata(y[m])
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / d) if d > 0 else np.nan), int(m.sum())


def _bh(p, q=0.10):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    flag = np.zeros(n, bool)
    flag[o[:k]] = True
    return flag


def label(col):
    base, _, tr = col.partition("|")
    nm = W.LABELS.get(base, base)
    if tr in ("d7", "dev"):
        return f"{nm} ({'zmiana 7 dni' if tr == 'd7' else 'odchylenie od średniej rocznej'})"
    if base.startswith("we_"):
        desc = {"a7": "natężenie 7 dni vs 90 dni", "a30": "natężenie 30 dni vs rok"}.get(tr, tr)
    else:
        desc = {"r1": "zwrot 1d", "r7": "zwrot 7d", "r30": "zwrot 30d", "a7": "uwaga 7d vs 90d", "a30": "uwaga 30d vs rok",
                "w7": "anomalia 7d", "w30": "anomalia 30d", "s": ""}.get(tr, tr)
    return f"{nm} ({desc})" if desc else nm


def world_analogs(df, top=8, gap=60):
    """Okresy, w których świat (profil wydarzeń z ostatnich 30 dni) był najbardziej podobny do dzisiejszego, i co potem robił BTC.
    Opis, nie prognoza: to kilka epizodów, a nie niezależne próby."""
    import worldlog
    try:
        wl = pd.read_csv(worldlog.DAILY, index_col=0, parse_dates=True)
    except Exception:
        return None
    cols = [c for c in wl.columns if c.startswith("k_") or c in ("c_armed", "c_biz", "c_pol", "c_disaster", "c_intl")]
    x = np.log1p(wl[cols]).reindex(df.index)
    x = x.rolling(30, min_periods=20).mean()
    z = (x - x.expanding(365).mean()) / x.expanding(365).std()
    z = z.dropna(how="any")
    if len(z) < 800:
        return None
    cur = z.iloc[-1].to_numpy()
    c = df["close"]
    f30 = (c.shift(-30) / c - 1) * 100
    f90 = (c.shift(-90) / c - 1) * 100
    Z = z.to_numpy()
    d = np.sqrt(((Z - cur) ** 2).mean(axis=1))
    cand = [(d[i], z.index[i]) for i in range(len(z) - 120)]     # tylko dni z poznanym wynikiem 90 dni
    cand.sort()
    picked = []
    for dist, day in cand:
        if all(abs((day - p[1]).days) > gap for p in picked):
            picked.append((dist, day))
        if len(picked) == top:
            break
    rows = [{"date": day.strftime("%Y-%m-%d"), "dist": float(dist), "f30": float(f30.loc[day]), "f90": float(f90.loc[day])} for dist, day in picked]
    return {"rows": rows, "med30": float(np.median([r["f30"] for r in rows])), "med90": float(np.median([r["f90"] for r in rows])),
            "up30": float(np.mean([r["f30"] > 0 for r in rows])), "n_feat": len(cols), "asof": z.index[-1].strftime("%Y-%m-%d")}


def run_scan(df, T):
    lr = np.log(df["close"])
    n = len(df)
    rows = []
    split70 = int(n * 0.7)
    mid = n // 2
    for h in HORIZONS:
        y = (lr.shift(-h) - lr).to_numpy(float)
        for lag in LAGS:
            for col in T.columns:
                if col.startswith(("s_", "g_")) and (lag != 0 or h != 30):      # ~500 spółek: tylko jeden wariant, żeby nie mnożyć testów
                    continue
                x = T[col].shift(lag).to_numpy(float)
                r, m = _corr(x, y)
                if not np.isfinite(r):
                    continue
                n_eff = max(m / h, 5)
                tt = r * np.sqrt((n_eff - 2) / max(1 - r * r, 1e-9))
                p = 2 * (1 - tdist.cdf(abs(tt), n_eff - 2))
                r1, _ = _corr(x[:mid], y[:mid])
                r2, _ = _corr(x[mid:], y[mid:])
                rtr, _ = _corr(x[:split70], y[:split70])
                rte, _ = _corr(x[split70:], y[split70:])
                rows.append({"col": col, "h": h, "lag": lag, "r": r, "n": m, "p": float(p), "r_a": r1, "r_b": r2, "r_train": rtr, "r_test": rte})
    R = pd.DataFrame(rows)
    R["fdr"] = _bh(R["p"].to_numpy())
    R["stable"] = (np.sign(R["r_a"]) == np.sign(R["r_b"])) & (R["r_a"].abs() > 0.03) & (R["r_b"].abs() > 0.03)
    R["oos"] = (np.sign(R["r_train"]) == np.sign(R["r_test"])) & (R["r_test"].abs() > 0.05)
    R["discovery"] = R["fdr"] & R["stable"] & R["oos"]
    R["label"] = R["col"].map(label)
    R["score"] = R["r"].abs() * R["stable"].astype(float) * R["oos"].astype(float)

    # współruch tygodniowy z BTC (opis, nie predykcja): zwroty 7d, próbkowanie co 7 dni
    co = []
    btc7 = lr.diff(7).to_numpy(float)
    for col in [c for c in T.columns if c.endswith("|r7") and not c.startswith(("m_", "mk_"))]:
        x = T[col].to_numpy(float)
        sl = slice(7, None, 7)
        r, m = _corr(x[sl], btc7[sl])
        if np.isfinite(r):
            co.append({"label": W.LABELS.get(col.split("|")[0], col), "r": r, "n": m})
    co = pd.DataFrame(co).sort_values("r") if co else pd.DataFrame(columns=["label", "r", "n"])

    summary = {"n_tests": int(len(R)), "raw_p05": int((R["p"] < 0.05).sum()), "expected_fp": float(0.05 * len(R)),
               "n_fdr": int(R["fdr"].sum()), "n_stable": int((R["fdr"] & R["stable"]).sum()), "n_disc": int(R["discovery"].sum()),
               "n_series": int(T.columns.map(lambda c: c.split("|")[0]).nunique())}
    disc = R[R["discovery"]].sort_values("p").head(15)
    cand = R[~R["discovery"]].sort_values(["score", "p"], ascending=[False, True]).head(12)
    return {"summary": summary, "discoveries": disc, "candidates": cand, "comove_pos": co.tail(10).iloc[::-1], "comove_neg": co.head(8), "all": R,
            "analogs": world_analogs(df)}
