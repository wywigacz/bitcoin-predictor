"""Zewnętrzny sędzia dla szans dotknięcia poziomów: rynki zakładów, na których ludzie stawiają prawdziwe pieniądze.

Polymarket (wstecz, od 2024): rozstrzygnięte miesięczne rynki „czy BTC dotknie X w danym miesiącu” (rozliczane po 1-minutowych
świecach Coinbase). Dla każdego rynku bierzemy cenę z pierwszego dnia po otwarciu i liczymy naszą szansę dotknięcia tego samego
poziomu do końca miesiąca, używając wyłącznie danych sprzed tego dnia. Porównanie wynikiem Briera z tym, co się stało.
Kalshi (na żywo, od 2026-10-30): przy każdej aktualizacji zapisujemy cenę rynków „czy BTC przekroczy / spadnie poniżej X do końca
miesiąca” i naszą szansę; po rozstrzygnięciu dopisujemy wynik. Dziennik tylko dopisywany.
To raport uczciwości, a nie test zaliczony/niezaliczony: rynki w jednym miesiącu dzielą tę samą ścieżkę ceny, więc
niezależnych obserwacji jest tyle, ile miesięcy."""
import json
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import requests

import config as C
import touch as TO

POLY = C.DATA_DIR / "poly_markets.csv"
KLIVE = C.DATA_DIR / "judge_live.csv"
UA = {"User-Agent": "Mozilla/5.0 (btc-predictor research)"}
SLUG = re.compile(r"what-price-will-(btc|bitcoin)-(reach|hit|dip)")
HOLDOUT = pd.Timestamp("2024-10-29")


def _get(host, path):
    """Zwykłe zapytanie; jeśli lokalny DNS blokuje adres, rozwiązanie przez DNS Google i curl z --resolve."""
    try:
        r = requests.get(f"https://{host}{path}", headers=UA, timeout=60)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    try:
        ip = requests.get("https://dns.google/resolve", params={"name": host, "type": "A"}, timeout=20).json()["Answer"][-1]["data"]
        curl = shutil.which("curl") or shutil.which("curl.exe") or "curl"
        out = subprocess.run([curl, "-s", "--max-time", "60", "--resolve", f"{host}:443:{ip}", f"https://{host}{path}"],
                             capture_output=True, text=True, encoding="utf-8")
        return json.loads(out.stdout) if out.stdout else None
    except Exception:
        return None


def _is_up(slug, q):
    s = (str(slug) + " " + str(q)).lower()
    return not any(w in s for w in ("dip", "below", "fall to", "drop to", "drop below", "crash"))


def _strike(q):
    m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?", q)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * 1000 if m.group(2) else v


def poly_update(verbose=True):
    old = pd.read_csv(POLY) if POLY.exists() else pd.DataFrame()
    seen = set(old["key"]) if len(old) else set()
    rows, off = [], 0
    while off < 5000:
        ev = _get("gamma-api.polymarket.com", f"/events?tag_slug=bitcoin&closed=true&limit=100&offset={off}&order=startDate&ascending=true")
        if not isinstance(ev, list) or not ev:
            break
        for e in ev:
            if not isinstance(e, dict) or not SLUG.search(e.get("slug", "")):
                continue
            for m in e.get("markets", []) or []:
                if not isinstance(m, dict):
                    continue
                q = m.get("question", "")
                key = f'{e["slug"]}|{q}'
                if key in seen:
                    continue
                L = _strike(q)
                try:
                    out = json.loads(m.get("outcomePrices") or "[]")
                    tok = json.loads(m.get("clobTokenIds") or "[]")[0]
                except (ValueError, IndexError):
                    continue
                if L is None or len(out) < 2 or out[0] not in ("0", "1"):
                    continue
                start = pd.Timestamp(m.get("startDate") or e.get("startDate")).tz_localize(None) if (m.get("startDate") or e.get("startDate")) else None
                end = pd.Timestamp(m.get("endDate")).tz_localize(None) if m.get("endDate") else None
                if start is None or end is None:
                    continue
                h = _get("clob.polymarket.com", f"/prices-history?market={tok}&interval=max&fidelity=1440")
                h = h if isinstance(h, dict) else {}
                pts = [(pd.Timestamp(int(p["t"]), unit="s"), float(p["p"])) for p in h.get("history", []) if isinstance(p, dict)]
                pts = [(t, p) for t, p in pts if t >= start + pd.Timedelta(days=1)]
                if not pts:
                    continue
                t0, p0 = pts[0]
                d0 = t0.normalize()
                if (end.normalize() - d0).days < 5:
                    continue
                up = _is_up(e["slug"], q)
                rows.append({"key": key, "slug": e["slug"], "question": q, "strike": L, "up": up, "d0": d0.strftime("%Y-%m-%d"),
                             "end": end.normalize().strftime("%Y-%m-%d"), "p_market": p0, "outcome": float(out[0] == "1")})
                time.sleep(0.2)
        off += 100
        if rows:                                                 # zapis po każdej stronie: przerwanie nie gubi pobranego
            part = pd.DataFrame(rows)
            (pd.concat([old, part], ignore_index=True) if len(old) else part).to_csv(POLY, index=False)
    new = pd.DataFrame(rows)
    allm = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(allm):
        allm.to_csv(POLY, index=False)
    if verbose:
        print(f"  [OK   ] Polymarket (rozstrzygnięte rynki dotknięcia): {len(new)} nowych, razem {len(allm)}", flush=True)
    return allm


def _model_probs(df, sig, D, day, H, levels_up, mode, n=4000):
    """Nasza szansa dotknięcia poziomów w ciągu H dni, liczona tylko z danych do dnia `day` włącznie."""
    idx = df.index
    t = int(np.searchsorted(idx, day, side="right")) - 1
    if t < 500:
        return None
    ok = np.isfinite(D[: t + 1]).all(axis=1)
    pool = D[: t + 1][ok]
    s_lr = float(pd.Series(sig[: t + 1], idx[: t + 1]).loc[TO.START:].median())
    lr = np.log(df["close"].to_numpy(float))
    mu = TO._mu(np.r_[np.nan, np.diff(lr)], t)
    gp = None
    if mode in ("fhs", "fhs_drift"):
        gp = TO._garch(np.nan_to_num(np.r_[np.nan, np.diff(lr)] * 100)[int(np.searchsorted(idx, pd.Timestamp(TO.START))):t + 1])
    ext = TO._paths(float(sig[t]), s_lr, pool, H, n, mode, gp, mu)
    P = float(df["close"].iloc[t])
    return [TO.probs(ext, [L / P - 1], True)[0] if (L / P - 1 > 0) == up else 1.0 for L, up in levels_up]


def poly_eval(df, sig, mode):
    M = pd.read_csv(POLY) if POLY.exists() else pd.DataFrame()
    if M.empty:
        return None
    M["up"] = [_is_up(s, q) for s, q in zip(M["slug"], M["question"])]
    D = TO._days(df, sig)
    out = []
    for (d0, end), g in M.groupby(["d0", "end"]):
        day = pd.Timestamp(d0) - pd.Timedelta(days=1)      # cena rynku z początku dnia d0 = nasze dane do zamknięcia dnia d0−1
        H = (pd.Timestamp(end) - pd.Timestamp(d0)).days + 1
        pr = _model_probs(df, sig, D, day, H, list(zip(g["strike"], g["up"])), mode)
        if pr is None:
            continue
        for (_, r), pm in zip(g.iterrows(), pr):
            out.append({**r.to_dict(), "p_model": pm})
    R = pd.DataFrame(out)
    if R.empty:
        return None
    R["blend"] = 0.5 * R["p_model"] + 0.5 * R["p_market"]
    b = lambda c, m: float(((R[c][m] - R["outcome"][m]) ** 2).mean())
    res = {}
    for nm, m in (("przed", pd.to_datetime(R["end"]) < HOLDOUT), ("po", pd.to_datetime(R["d0"]) >= HOLDOUT), ("calosc", pd.Series(True, index=R.index))):
        m = m.to_numpy()
        if m.sum() < 5:
            continue
        res[nm] = {"n": int(m.sum()), "months": int(R.loc[m, "end"].nunique()), "brier_model": b("p_model", m), "brier_market": b("p_market", m),
                   "brier_blend": b("blend", m)}
    examples = R.sort_values("d0").tail(6)[["question", "d0", "p_market", "p_model", "outcome"]].to_dict("records")
    return {"eval": res, "examples": examples, "n": int(len(R))}


def kalshi_update(df, sig, mode, asof):
    """Dziennik na żywo: ceny rynków Kalshi i nasza szansa dla tych samych poziomów; po rozstrzygnięciu wynik."""
    old = pd.read_csv(KLIVE) if KLIVE.exists() else pd.DataFrame()
    rows = []
    D = TO._days(df, sig)
    for series, up in (("KXBTCMAXMON", True), ("KXBTCMINMON", False)):
        try:
            ms = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets", params={"series_ticker": series, "status": "open", "limit": 200},
                              headers=UA, timeout=30).json().get("markets", [])
        except Exception:
            ms = []
        lv, keep = [], []
        for m in ms:
            L = m.get("floor_strike") if up else m.get("cap_strike")
            try:
                bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
            except ValueError:
                continue
            if L is None or ask <= 0 or ask - bid > 0.2:
                continue
            close = pd.Timestamp(m["close_time"]).tz_localize(None)
            H = max((close.normalize() - pd.Timestamp(asof)).days, 1)
            lv.append((float(L), up)); keep.append((m["ticker"], float(L), close, H, (bid + ask) / 2))
        if not keep:
            continue
        for H in sorted({k[3] for k in keep}):
            sub = [(k, l) for k, l in zip(keep, lv) if k[3] == H]
            pr = _model_probs(df, sig, D, pd.Timestamp(asof), H, [l for _, l in sub], mode)
            for (k, _), pm in zip(sub, pr or [np.nan] * len(sub)):
                rows.append({"asof": str(pd.Timestamp(asof).date()), "ticker": k[0], "strike": k[1], "up": up, "close": str(k[2].date()),
                             "H": H, "p_market": k[4], "p_model": pm, "outcome": np.nan})
    new = pd.DataFrame(rows)
    if len(old) and len(new):
        new = new[~(new["asof"] + "|" + new["ticker"]).isin(old["asof"].astype(str) + "|" + old["ticker"])]
    L = pd.concat([old, new], ignore_index=True) if len(old) else new
    if L.empty:
        return None
    # wynik po rozstrzygnięciu (dopisujemy tylko wynik, prognoz nie ruszamy)
    pend = L[L["outcome"].isna() & (pd.to_datetime(L["close"]) < pd.Timestamp(asof))]["ticker"].unique()
    for tk in pend[:200]:
        try:
            m = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{tk}", headers=UA, timeout=30).json().get("market", {})
            if m.get("result") in ("yes", "no"):
                L.loc[L["ticker"] == tk, "outcome"] = float(m["result"] == "yes")
        except Exception:
            continue
    L.to_csv(KLIVE, index=False)
    d = L.dropna(subset=["outcome", "p_model"])
    return {"rows": int(len(L)), "resolved": int(len(d)),
            "brier_model": float(((d["p_model"] - d["outcome"]) ** 2).mean()) if len(d) else None,
            "brier_market": float(((d["p_market"] - d["outcome"]) ** 2).mean()) if len(d) else None,
            "now_asof": str(L["asof"].astype(str).max()),
            "now": L[(L["asof"].astype(str) == L["asof"].astype(str).max()) & (pd.to_datetime(L["close"]) >= pd.Timestamp(asof))]
            .sort_values(["up", "strike"]).head(14)[["strike", "up", "close", "p_market", "p_model"]].to_dict("records")}


def run(df, sig, mode, asof):
    out = {}
    try:
        poly_update()
        out["poly"] = poly_eval(df, sig, mode)
    except Exception as e:
        print("  [uwaga] Polymarket:", str(e)[:120])
    try:
        out["kalshi"] = kalshi_update(df, sig, mode, asof)
    except Exception as e:
        print("  [uwaga] Kalshi:", str(e)[:120])
    return out
