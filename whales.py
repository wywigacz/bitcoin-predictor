"""Ruchy dużych graczy: przepływy do funduszy ETF na bitcoina, zakupy i sprzedaże firmy Strategy (MicroStrategy),
pozycje najlepszych i najgorszych traderów na giełdzie Hyperliquid (wszystkie pozycje są tam publiczne).

Źródła (bez kluczy):
  - TFTC Bitcoin ETF Tracker, licencja CC BY 4.0 (dane dzienne od SoSoValue): https://www.tftc.io/bitcoin-etf-flows
  - Bitbo, tabela transakcji Strategy (zgodna z raportami 8-K w SEC): https://bitbo.io/treasuries/microstrategy/
  - Hyperliquid, publiczne API /info i ranking traderów.

Wynik: data/whales.csv (dzienny) oraz data/hl_snapshots.csv (zdjęcia pozycji co 6 godzin; historii nie da się pobrać wstecz,
więc nagrywamy ją od 2026-10-29). Te szeregi NIE trafiają do dużego skanera; testuje je osobno confirm.py."""
import re
import time

import numpy as np
import pandas as pd
import requests

import config as C

OUT = C.DATA_DIR / "whales.csv"
SNAP = C.DATA_DIR / "hl_snapshots.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
HL = "https://api.hyperliquid.xyz/info"
TOP = 150          # tylu najlepszych i tylu najgorszych traderów śledzimy w każdym zdjęciu


def _get(url, **kw):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"brak odpowiedzi: {url[:90]}")


def _hl(body):
    for i in range(3):
        try:
            r = requests.post(HL, json=body, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * i)
    raise RuntimeError(f"Hyperliquid: {body.get('type')}")


def etf():
    j = _get("https://www.tftc.io/bitcoin-etf-flows/data.json").json()
    rows = next(v for v in j.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "netFlowUsd" in v[0])
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["date"])
    out = pd.DataFrame(index=d.index)
    out["w_etf_flow"] = d["netFlowUsd"].astype(float)
    out["w_etf_aum"] = pd.to_numeric(d["totalNetAssetsUsd"], errors="coerce")
    per = pd.DataFrame([x if isinstance(x, dict) else {} for x in d["perEtfUsd"]], index=d.index)
    for tic, name in (("IBIT", "w_etf_ibit"), ("FBTC", "w_etf_fbtc"), ("GBTC", "w_etf_gbtc")):
        if tic in per:
            out[name] = pd.to_numeric(per[tic], errors="coerce")
    return out


def strategy():
    t = _get("https://bitbo.io/treasuries/microstrategy/").text
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
        c = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        c = [x for x in c if x]
        if len(c) >= 4 and re.match(r"\d+/\d+/\d\d(\d\d)?$", c[0]):
            n = c[1].replace(",", "")
            neg = n.startswith("(")
            try:
                btc = float(n.strip("()")) * (-1 if neg else 1)
                total = float(c[3].replace(",", ""))
            except ValueError:
                continue
            fmt = "%m/%d/%Y" if len(c[0].split("/")[2]) == 4 else "%m/%d/%y"
            rows.append((pd.to_datetime(c[0], format=fmt), btc, total))
    if len(rows) < 50:
        raise RuntimeError(f"tabela Strategy: tylko {len(rows)} wierszy")
    d = pd.DataFrame(rows, columns=["date", "btc", "total"]).groupby("date").agg({"btc": "sum", "total": "last"}).sort_index()
    # tabela pomija część drobnych transakcji, ale stan po każdym raporcie jest pełny: zmianę liczymy z różnicy stanów
    chg = d["total"].diff().fillna(d["total"].iloc[0])
    return pd.DataFrame({"w_mstr_btc": chg, "w_mstr_total": d["total"]})


def hl_funding(since):
    """Stawka finansowania BTC na Hyperliquid (co godzinę, od maja 2023), uśredniana do dnia."""
    start = int(pd.Timestamp(since).timestamp() * 1000)
    acc = []
    for _ in range(200):
        page = _hl({"type": "fundingHistory", "coin": "BTC", "startTime": start})
        if not page:
            break
        acc += page
        nxt = int(page[-1]["time"]) + 1
        if len(page) < 500 or nxt <= start:
            break
        start = nxt
        time.sleep(0.2)
    if not acc:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(r["time"]), unit="ms"): float(r["fundingRate"]) for r in acc})
    return s.groupby(s.index.normalize()).mean() * 24 * 365 * 100     # w % rocznie


def hl_snapshot():
    """Zdjęcie pozycji na BTC: najlepsi traderzy (największy zysk w całej historii konta) kontra najgorsi."""
    lb = _get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard").json()["leaderboardRows"]
    rows = []
    for r in lb:
        try:
            perf = dict(r["windowPerformances"])
            rows.append((r["ethAddress"], float(r["accountValue"]), float(perf["allTime"]["pnl"])))
        except (KeyError, ValueError, TypeError):
            continue
    L = pd.DataFrame(rows, columns=["addr", "value", "pnl"])
    smart = L[L["value"] >= 100_000].nlargest(TOP, "pnl")
    dumb = L[L["value"] >= 10_000].nsmallest(TOP, "pnl")
    ctx = _hl({"type": "metaAndAssetCtxs"})
    i = [u["name"] for u in ctx[0]["universe"]].index("BTC")
    px, oi = float(ctx[1][i]["markPx"]), float(ctx[1][i]["openInterest"])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    snap = {"time": now, "px": px, "oi_usd": oi * px}
    for grp, G in (("smart", smart), ("dumb", dumb)):
        net, nl, ns = 0.0, 0, 0
        for a in G["addr"]:
            try:
                st = _hl({"type": "clearinghouseState", "user": a})
            except RuntimeError:
                continue
            for p in st.get("assetPositions", []):
                q = p.get("position", {})
                if q.get("coin") == "BTC":
                    sz = float(q["szi"])
                    net += sz * px
                    nl += sz > 0
                    ns += sz < 0
            time.sleep(0.06)
        snap.update({f"{grp}_net_usd": net, f"{grp}_long": nl, f"{grp}_short": ns, f"{grp}_n": len(G)})
    old = pd.read_csv(SNAP, parse_dates=["time"]) if SNAP.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([snap])], ignore_index=True).to_csv(SNAP, index=False, float_format="%.6g")
    return snap


def hl_daily():
    if not SNAP.exists():
        return pd.DataFrame()
    s = pd.read_csv(SNAP, parse_dates=["time"])
    s["day"] = s["time"].dt.normalize()
    g = s.groupby("day").mean(numeric_only=True)
    out = pd.DataFrame(index=g.index)
    for grp in ("smart", "dumb"):
        out[f"w_hl_{grp}"] = g[f"{grp}_net_usd"] / g["oi_usd"] * 100            # pozycja netto grupy jako % całego rynku BTC na Hyperliquid
        out[f"w_hl_{grp}_longpct"] = g[f"{grp}_long"] / (g[f"{grp}_long"] + g[f"{grp}_short"]).replace(0, np.nan) * 100
    out["w_hl_oi"] = g["oi_usd"]
    return out


def update(verbose=True):
    old = pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else pd.DataFrame()
    parts, status = [], []
    jobs = [("ETF na bitcoina (TFTC)", etf), ("Strategy: zakupy i sprzedaże BTC (Bitbo)", strategy),
            ("Hyperliquid: stawka finansowania", lambda: hl_funding(old["w_hl_fund"].dropna().index.max() - pd.Timedelta(days=3)
                                                                    if "w_hl_fund" in old and old["w_hl_fund"].notna().any() else "2023-05-01").rename("w_hl_fund").to_frame()),
            ("Hyperliquid: pozycje najlepszych i najgorszych traderów", lambda: (hl_snapshot(), hl_daily())[1])]
    for name, fn in jobs:
        t0 = time.time()
        try:
            r = fn()
            parts.append(r)
            status.append((name, "OK", f"{r.shape[1]} szeregów, {r.notna().any(axis=1).sum()} dni, {time.time() - t0:.0f}s"))
        except Exception as e:
            status.append((name, "BŁĄD", str(e)[:120]))
    if verbose:
        for s in status:
            print(f"  [{s[1]:5}] {s[0]}: {s[2]}", flush=True)
    new = pd.concat([p for p in parts if len(p)], axis=1) if any(len(p) for p in parts) else pd.DataFrame()
    merged = new.combine_first(old) if len(old) else new
    merged = merged.sort_index()
    merged.index.name = "date"
    C.DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(OUT, float_format="%.8g")
    return merged, status
