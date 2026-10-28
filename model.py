"""Cechy, sygnały, wskaźnik złożony i prognozy rozkładu ceny BTC.

Podejście (zgodne z wnioskami opracowań):
 1. Wskaźniki -> oceny sygnałów w skali -1..+1 (progi z opracowań), ważone wg wiarygodności.
 2. Złożony wskaźnik (composite) opisuje fazę rynku (dno/akumulacja <-> przegrzanie).
 3. Prognoza = rozkład ceny. Skala = zmienność (EWMA), kształt = empiryczne reszty z historii,
    przesunięcie środka = regresja composite -> przyszły zwrot, włączana TYLKO gdy walk-forward
    (test na danych "z przyszłości") wykaże przewagę nad prognozą "bez zmian".
"""
import numpy as np
import pandas as pd

import config as C

QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
FFILL = {"m2": 21, "fng": 3, "funding": 3, "oi": 3, "dom": 3}


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def build_frame(raw):
    """Ciągła oś dzienna; braki uzupełniane do przodu w rozsądnych limitach."""
    df = raw.sort_index()
    df = df[df["close"].notna()]
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df["close"] = df["close"].ffill()
    for c in df.columns:
        if c != "close":
            df[c] = df[c].ffill(limit=FFILL.get(c, 7))
    return df


def features(df):
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["price"] = c
    f["ma200w"] = c / c.rolling(1400).mean()
    f["mayer"] = c / c.rolling(200).mean()
    f["pi"] = c.rolling(111).mean() / (2 * c.rolling(350).mean())
    mvrv = _col(df, "mvrv")
    f["mvrv"] = mvrv
    f["realprice"] = c / mvrv                       # Realized Price = cena / MVRV
    mr = _col(df, "minerrev")
    f["puell"] = mr.rolling(7).mean() / mr.rolling(365, min_periods=300).mean()
    h = _col(df, "hash")
    h30, h60 = h.rolling(30).mean(), h.rolling(60).mean()
    f["hashcap"] = (h30 < h60).astype(float).where(h60.notna())
    f["fng"] = _col(df, "fng")
    f["funding"] = _col(df, "funding").rolling(7, min_periods=2).mean()
    st = _col(df, "stable")
    f["stable30"] = st / st.shift(30) - 1
    m2 = _col(df, "m2")
    f["m2yoy"] = (m2 / m2.shift(365) - 1).shift(70)
    for k, n in (("dxy", 30), ("ndx", 90)):
        s = _col(df, k)
        f[f"{k}{n}"] = s / s.shift(n) - 1
    f["vix"] = _col(df, "vix")
    irx = _col(df, "irx")
    f["rates180"] = irx - irx.shift(180)
    hv = pd.to_datetime(C.HALVINGS)
    days = pd.Series(np.nan, index=df.index)
    for d in hv:
        days[df.index >= d] = (df.index[df.index >= d] - d).days
    f["halv_days"] = days
    return f


def _clip(x):
    return x.clip(-1, 1)


def halving_score(d):
    d = d.astype(float)
    conds = [d < 180, d < 365, d < 560, d < 770, d < 1000, d >= 1000]
    vals = [0.4, 0.2, -0.4, -0.5, 0.6, 0.4]
    return pd.Series(np.select(conds, vals, default=np.nan), index=d.index).where(d.notna())


def scores(f):
    s = pd.DataFrame(index=f.index)
    s["mvrv"] = _clip((1.8 - f["mvrv"]) / 0.8)
    s["ma200w"] = _clip((1.7 - f["ma200w"]) / 0.7)
    s["mayer"] = _clip((1.6 - f["mayer"]) / 0.8)
    s["puell"] = _clip((1.2 - f["puell"]) / 0.9)
    s["hashribbon"] = f["hashcap"] * 0.6
    s["fng"] = _clip((45 - f["fng"]) / 35)
    s["funding"] = _clip((0.0001 - f["funding"]) / 0.0003)
    s["stable"] = _clip(f["stable30"] / 0.03)
    s["m2"] = _clip((f["m2yoy"] - 0.04) / 0.05)
    s["dxy"] = _clip(-f["dxy30"] / 0.025)
    s["ndx"] = _clip(f["ndx90"] / 0.12)
    s["vix"] = _clip((22 - f["vix"]) / 10)
    s["rates"] = _clip(-f["rates180"] / 1.0)
    s["pi"] = (-(f["pi"] - 0.85) / 0.15).clip(-1, 0)
    s["halving"] = halving_score(f["halv_days"])
    return s


def composite(s):
    w = pd.Series(C.WEIGHTS).reindex(s.columns).fillna(0.0)
    avail = s.notna().mul(w, axis=1)
    wsum = avail.sum(axis=1)
    comp = s.fillna(0.0).mul(w, axis=1).sum(axis=1) / wsum.replace(0, np.nan)
    return comp.where(wsum >= C.MIN_WEIGHT), wsum


def regime(c):
    if c is None or np.isnan(c):
        return "brak danych"
    if c >= 0.45:
        return "silna strefa akumulacji (historycznie okolice dna)"
    if c >= 0.15:
        return "umiarkowanie tanio / wczesna faza odbicia"
    if c > -0.15:
        return "neutralnie"
    if c > -0.45:
        return "umiarkowanie drogo / rosnące ryzyko"
    return "strefa przegrzania (historycznie okolice szczytu)"


def ewma_vol(c):
    r = np.log(c).diff()
    return np.sqrt((r ** 2).ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean())


def _targets(P, sig, H):
    n = len(P)
    z = np.full(n, np.nan)
    z[: n - H] = np.log(P[H:] / P[: n - H]) / (sig[: n - H] * np.sqrt(H))
    return z


def _fit(x, z):
    mx, my = x.mean(), z.mean()
    vx = x.var()
    b = ((x - mx) * (z - my)).mean() / (vx * (1 + C.SLOPE_RIDGE)) if vx > 0 else 0.0
    return mx, my, b


def walk_forward(x, z, H):
    """Test out-of-sample: model dopasowany tylko na próbkach znanych w chwili t."""
    n = len(x)
    ok = ~(np.isnan(x) | np.isnan(z))
    xx, zz = np.where(ok, x, 0.0), np.where(ok, z, 0.0)
    cn, cx, cz = np.cumsum(ok), np.cumsum(xx), np.cumsum(zz)
    cxx, cxz = np.cumsum(xx * xx), np.cumsum(xx * zz)
    sp, act = [], []
    for t in range(C.WF_START, n - H, C.WF_STEP):
        m = t - H
        k = cn[m]
        if k < C.MIN_TRAIN or np.isnan(x[t]) or np.isnan(z[t]):
            continue
        mx, my = cx[m] / k, cz[m] / k
        vx = cxx[m] / k - mx * mx
        cov = cxz[m] / k - mx * my
        b = cov / (vx * (1 + C.SLOPE_RIDGE)) if vx > 1e-9 else 0.0
        sp.append(b * (x[t] - mx) + C.DRIFT_SHRINK * my)
        act.append(z[t])
    sp, act = np.array(sp), np.array(act)
    if len(sp) < 50:
        return {"n": int(len(sp)), "ic": np.nan, "skill": np.nan, "hit": np.nan, "active": False}
    ic = float(np.corrcoef(sp, act)[0, 1])
    skill = float(1 - np.mean((act - sp) ** 2) / np.mean(act ** 2))
    hit = float(np.mean(np.sign(sp) == np.sign(act)))
    active = bool(ic >= C.MIN_IC and skill > C.MIN_SKILL)
    return {"n": int(len(sp)), "ic": ic, "skill": skill, "hit": hit, "active": active}


def analyze(df):
    f = features(df)
    s = scores(f)
    comp, wsum = composite(s)
    P = df["close"].to_numpy(float)
    sig = ewma_vol(df["close"]).to_numpy(float)
    x = comp.to_numpy(float)
    last = len(df) - 1
    x_now = comp.dropna().iloc[-1]
    comp_date = comp.dropna().index[-1]
    p0, sig_now = P[last], sig[last]

    forecasts, quality = [], {}
    for H in C.HORIZONS:
        z = _targets(P, sig, H)
        ok = ~(np.isnan(x) | np.isnan(z))
        mx, my, b = _fit(x[ok], z[ok])
        wf = walk_forward(x, z, H)
        active = wf["active"]
        if active:
            e = z[ok] - (b * (x[ok] - mx) + C.DRIFT_SHRINK * my)
            pred = b * (x_now - mx) + C.DRIFT_SHRINK * my
        else:
            zz = z[~np.isnan(z)]
            e = zz - np.median(zz)
            pred = 0.0
        qv = np.quantile(e, QS)
        scale = sig_now * np.sqrt(H)
        prices = {f"p{int(q * 100)}": float(p0 * np.exp(scale * (pred + v))) for q, v in zip(QS, qv)}
        forecasts.append({
            "horizon": H, "target": (df.index[last] + pd.Timedelta(days=H)).strftime("%Y-%m-%d"),
            "pred_z": float(pred), "prob_up": float(np.mean(e > -pred)), **prices,
        })
        quality[H] = {**wf, "slope": float(b), "samples": int(ok.sum())}

    cur = s.loc[comp_date]
    parts = []
    for k in s.columns:
        w = C.WEIGHTS.get(k, 0.0)
        parts.append({"key": k, "score": None if np.isnan(cur[k]) else float(cur[k]), "weight": w})

    r = np.log(df["close"]).diff()
    corr = {}
    for k in ("ndx", "spx", "gold", "dxy"):
        if k in df.columns:
            rk = np.log(df[k].where(df[k] > 0)).diff()
            pair = pd.concat([r, rk], axis=1).tail(120)
            pair = pair[pair.iloc[:, 1] != 0].dropna().tail(90)      # bez dni bez notowań (weekendy)
            if len(pair) > 30 and pair.iloc[:, 1].std() > 0:
                corr[k] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))

    return {
        "asof": df.index[last].strftime("%Y-%m-%d"), "price": float(p0), "daily_vol": float(sig_now),
        "composite": float(x_now), "composite_date": comp_date.strftime("%Y-%m-%d"),
        "regime": regime(x_now), "forecasts": forecasts, "quality": quality, "signals": parts,
        "features": {k: (None if pd.isna(v) else float(v)) for k, v in f.iloc[last].items()},
        "corr90": corr, "frame": {"features": f, "scores": s, "composite": comp, "sig": pd.Series(sig, index=df.index)},
    }
