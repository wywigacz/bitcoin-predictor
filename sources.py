"""Pobieranie danych z darmowych, publicznych API (bez kluczy)."""
import io
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (btc-predictor)"}


def _get(url, timeout=30, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = str(e)[:120]
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{last} <- {url}")


def _ser(d):
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def _day(ts_sec):
    return pd.Timestamp(int(ts_sec), unit="s").normalize()


def _today():
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def fetch_price(since):
    """Bitstamp: dzienne świece BTC/USD od 2011."""
    start = int(pd.Timestamp(since).timestamp()) if since is not None else 1313625600
    close, vol = {}, {}
    for _ in range(25):
        j = _get(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={start}").json()
        arr = j.get("data", {}).get("ohlc", [])
        if not arr:
            break
        for c in arr:
            d = _day(c["timestamp"])
            close[d], vol[d] = float(c["close"]), float(c["volume"])
        if len(arr) < 1000:
            break
        start = int(arr[-1]["timestamp"]) + 86400
    if not close:
        raise RuntimeError("Bitstamp: brak danych")
    return {"close": _ser(close), "volume": _ser(vol)}


def fetch_coinmetrics(since):
    """CoinMetrics community: MVRV, aktywne adresy, podaż."""
    metrics = {"mvrv": "CapMVRVCur", "adr": "AdrActCnt", "supply": "SplyCur"}
    start = pd.Timestamp(since).strftime("%Y-%m-%d") if since is not None else "2011-01-01"
    out, errs = {}, []
    for field, m in metrics.items():
        try:
            url = (f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc"
                   f"&metrics={m}&frequency=1d&start_time={start}&page_size=10000")
            d = {}
            while url:
                j = _get(url).json()
                for row in j.get("data", []):
                    if row.get(m) is not None:
                        d[row["time"][:10]] = float(row[m])
                url = j.get("next_page_url")
            if d:
                out[field] = _ser(d)
        except Exception as e:
            errs.append(f"{field}: {e}")
    if not out:
        raise RuntimeError("; ".join(errs))
    return out


def fetch_blockchain(since):
    """blockchain.com: hash rate i przychody górników (USD)."""
    if since is None:
        span = "all"
    else:
        span = f"{max((_today() - pd.Timestamp(since)).days + 5, 10)}days"
    out = {}
    for field, chart in (("hash", "hash-rate"), ("minerrev", "miners-revenue")):
        j = _get(f"https://api.blockchain.info/charts/{chart}?timespan={span}&sampled=false&format=json", timeout=60).json()
        out[field] = _ser({_day(v["x"]): v["y"] for v in j["values"]})
    return out


def fetch_fng(since):
    j = _get(f"https://api.alternative.me/fng/?limit={0 if since is None else 15}&format=json").json()
    return {"fng": _ser({_day(r["timestamp"]): r["value"] for r in j["data"]})}


YAHOO = {"spx": "^GSPC", "ndx": "^IXIC", "dxy": "DX-Y.NYB", "vix": "^VIX",
         "tnx": "^TNX", "irx": "^IRX", "gold": "GC=F"}


def fetch_yahoo(since):
    p1 = 1313625600 if since is None else int(pd.Timestamp(since).timestamp())
    p2 = int(time.time()) + 86400
    out, errs = {}, []
    for field, sym in YAHOO.items():
        try:
            j = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}"
                     f"?period1={p1}&period2={p2}&interval=1d").json()
            res = j["chart"]["result"][0]
            ts, cl = res["timestamp"], res["indicators"]["quote"][0]["close"]
            out[field] = _ser({_day(t): c for t, c in zip(ts, cl) if c is not None})
        except Exception as e:
            errs.append(f"{field}: {e}")
    if not out:
        raise RuntimeError("; ".join(errs))
    return out


def fetch_stable(since):
    j = _get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=60).json()
    return {"stable": _ser({_day(r["date"]): r["totalCirculatingUSD"]["peggedUSD"] for r in j})}


def fetch_okx(since):
    """Funding rate (dzienna średnia z ~30 dni) i open interest (migawka z dziś)."""
    out = {}
    j = _get("https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100").json()
    df = pd.DataFrame(j["data"])
    df["d"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms").dt.normalize()
    df["r"] = df["fundingRate"].astype(float)
    out["funding"] = df.groupby("d")["r"].mean()
    j = _get("https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP").json()
    out["oi"] = pd.Series([float(j["data"][0]["oiCcy"])], index=[_today()])
    return out


def fetch_global(since):
    j = _get("https://api.coingecko.com/api/v3/global").json()
    return {"dom": pd.Series([float(j["data"]["market_cap_percentage"]["btc"])], index=[_today()])}


def fetch_m2(since):
    """Opcjonalnie: M2 USA z FRED (bywa niedostępny, wtedy sygnał jest pomijany)."""
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=WM2NS", timeout=15, tries=1)
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
    return {"m2": s.dropna()}


JOBS = [("Bitstamp (cena)", fetch_price), ("CoinMetrics (MVRV, adresy)", fetch_coinmetrics),
        ("Blockchain.com (hash rate, przychody)", fetch_blockchain), ("Fear & Greed", fetch_fng),
        ("Yahoo (makro, złoto)", fetch_yahoo), ("DefiLlama (stablecoiny)", fetch_stable),
        ("OKX (funding, OI)", fetch_okx), ("CoinGecko (dominacja)", fetch_global),
        ("FRED (M2, opcjonalnie)", fetch_m2)]


def fetch_all(since=None):
    data, status = {}, []
    for name, fn in JOBS:
        try:
            r = fn(since)
            data.update(r)
            status.append((name, "OK", ", ".join(f"{k}→{v.index.max():%Y-%m-%d}" for k, v in r.items())))
        except Exception as e:
            status.append((name, "BŁĄD", str(e)[:160]))
    return data, status
