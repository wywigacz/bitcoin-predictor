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
    """Bitstamp: dzienne świece BTC/USD od 2011. Tylko zamknięte dni UTC: bez tego parametru giełda zwraca też
    bieżącą, niedokończoną świecę, a wtedy prognoza i dziennik liczyłyby się od ceny z środka dnia."""
    start = int(pd.Timestamp(since).timestamp()) if since is not None else 1313625600
    close, vol, hi, lo = {}, {}, {}, {}
    for _ in range(25):
        j = _get(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={start}&exclude_current_candle=true").json()
        arr = j.get("data", {}).get("ohlc", [])
        if not arr:
            break
        for c in arr:
            d = _day(c["timestamp"])
            close[d], vol[d] = float(c["close"]), float(c["volume"])
            hi[d], lo[d] = float(c["high"]), float(c["low"])
        if len(arr) < 1000:
            break
        start = int(arr[-1]["timestamp"]) + 86400
    if not close:
        raise RuntimeError("Bitstamp: brak danych")
    return {"close": _ser(close), "volume": _ser(vol), "high": _ser(hi), "low": _ser(lo)}


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
    for field, chart in (("hash", "hash-rate"), ("minerrev", "miners-revenue"), ("ntx", "n-transactions"), ("fees", "transaction-fees-usd"),
                         ("txvol", "estimated-transaction-volume-usd"), ("uaddr", "n-unique-addresses"), ("difficulty", "difficulty")):
        j = _get(f"https://api.blockchain.info/charts/{chart}?timespan={span}&sampled=false&format=json", timeout=60).json()
        out[field] = _ser({_day(v["x"]): v["y"] for v in j["values"]})
    return out


def fetch_fng(since):
    j = _get(f"https://api.alternative.me/fng/?limit={0 if since is None else 15}&format=json").json()
    return {"fng": _ser({_day(r["timestamp"]): r["value"] for r in j["data"]})}


YAHOO = {"spx": "^GSPC", "ndx": "^IXIC", "dxy": "DX-Y.NYB", "vix": "^VIX",
         "tnx": "^TNX", "irx": "^IRX", "gold": "GC=F", "oil": "CL=F", "hyg": "HYG",
         "usdjpy": "JPY=X", "tlt": "TLT", "eth": "ETH-USD", "usdkrw": "KRW=X", "move": "^MOVE"}


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


WIKI = {"en": "Bitcoin", "de": "Bitcoin", "fr": "Bitcoin", "es": "Bitcoin", "ru": "Биткойн", "ja": "ビットコイン",
        "ko": "비트코인", "zh": "比特币", "tr": "Bitcoin", "pl": "Bitcoin", "pt": "Bitcoin", "it": "Bitcoin"}


def fetch_wiki(since):
    """Uwaga na Wikipedii (artykuł o Bitcoinie) w wielu językach, jako przybliżenie nastrojów regionów."""
    start = "20150701" if since is None else (pd.Timestamp(since) - pd.Timedelta(days=5)).strftime("%Y%m%d")
    end = _today().strftime("%Y%m%d")
    out = {}
    for lang, title in WIKI.items():
        try:
            u = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/"
                 f"{requests.utils.quote(title, safe='')}/daily/{start}/{end}")
            items = _get(u, timeout=40, tries=2).json()["items"]
            out[f"wiki_{lang}"] = _ser({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items})
        except Exception:
            pass
    if not out:
        raise RuntimeError("Wikipedia: brak danych")
    return out


def fetch_kimchi(since):
    """BTC/KRW z Upbit (Korea): do liczenia premii kimchi względem ceny w USD i kursu USD/KRW."""
    cutoff = pd.Timestamp("2017-10-01") if since is None else pd.Timestamp(since) - pd.Timedelta(days=5)
    d, to = {}, None
    for _ in range(40):
        url = "https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=200" + (f"&to={to}" if to else "")
        arr = _get(url).json()
        if not arr:
            break
        for c in arr:
            d[pd.Timestamp(c["candle_date_time_utc"][:10])] = float(c["trade_price"])
        oldest = pd.Timestamp(arr[-1]["candle_date_time_utc"])
        if oldest <= cutoff:
            break
        to = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")
        time.sleep(0.15)
    return {"btckrw": _ser(d)}


def fetch_deribit(since):
    """DVOL (implikowana zmienność, od 2021) i stosunek put/call z otwartych pozycji opcji (migawka)."""
    out = {}
    end = int(time.time() * 1000)
    start = 1616000000000 if since is None else int(pd.Timestamp(since).timestamp() * 1000) - 5 * 86400000
    d = {}
    while start < end:
        stop = min(start + 400 * 86400000, end)
        j = _get("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC"
                 f"&start_timestamp={start}&end_timestamp={stop}&resolution=86400").json()
        for t, o, h, l, c in j["result"]["data"]:
            d[pd.Timestamp(int(t), unit="ms").normalize()] = c
        start = stop
    if d:
        out["dvol"] = _ser(d)
    j = _get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=40).json()
    put = sum(x["open_interest"] for x in j["result"] if x["instrument_name"].endswith("-P"))
    call = sum(x["open_interest"] for x in j["result"] if x["instrument_name"].endswith("-C"))
    if call > 0:
        out["pcr"] = pd.Series([put / call], index=[_today()])
    return out


def fetch_cftc(since):
    """CFTC (tygodniowo): pozycjonowanie funduszy lewarowanych na kontraktach BTC na CME."""
    url = ("https://publicreporting.cftc.gov/resource/gpe5-46if.json?$limit=2000&$order=report_date_as_yyyy_mm_dd%20DESC"
           "&$where=market_and_exchange_names%20like%20%27BITCOIN%20-%20CHICAGO%25%27")
    rows = _get(url, timeout=40).json()
    d = {}
    for r in rows:
        try:
            net = float(r["lev_money_positions_long"]) - float(r["lev_money_positions_short"])
            d[pd.Timestamp(r["report_date_as_yyyy_mm_dd"][:10])] = net / float(r["open_interest_all"])
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    if not d:
        raise RuntimeError("CFTC: brak danych")
    s = _ser(d)
    s.index = s.index + pd.Timedelta(days=3)          # stan z wtorku CFTC publikuje w piątek: bez podglądania przyszłości
    return {"cot_lev": s}


def fetch_coinbase(since):
    """Dzienne zamknięcia Coinbase (USD): z nich premia Coinbase względem ceny z Bitstamp (popyt z USA)."""
    end = _today() + pd.Timedelta(days=1)
    start = pd.Timestamp("2015-07-20") if since is None else pd.Timestamp(since) - pd.Timedelta(days=5)
    d = {}
    while start < end:
        stop = min(start + pd.Timedelta(days=290), end)
        arr = _get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400&start={start:%Y-%m-%dT00:00:00Z}&end={stop:%Y-%m-%dT00:00:00Z}").json()
        for t, lo, hi, op, cl, vol in arr:
            d[_day(t)] = float(cl)
        start = stop
        time.sleep(0.4)
    return {"cbclose": _ser(d)}


def fetch_bybit(since):
    """Historia stawek za lewar na kontraktach wieczystych Bybit (od 2020), średnia dzienna."""
    end = int(time.time() * 1000)
    stop_at = int(pd.Timestamp("2020-03-26" if since is None else since).timestamp() * 1000)
    rows = []
    for _ in range(80):
        j = _get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=200&endTime={end}").json()
        lst = j.get("result", {}).get("list", [])
        if not lst:
            break
        rows += [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst]
        end = min(int(x["fundingRateTimestamp"]) for x in lst) - 1
        if end < stop_at:
            break
        time.sleep(0.2)
    if not rows:
        raise RuntimeError("Bybit: brak danych")
    s = pd.Series({pd.Timestamp(t, unit="ms"): v for t, v in rows}).sort_index()
    return {"bfund": s.groupby(s.index.normalize()).mean()}


JOBS = [("Bitstamp (cena)", fetch_price), ("CoinMetrics (MVRV, adresy)", fetch_coinmetrics),
        ("Blockchain.com (hash rate, przychody)", fetch_blockchain), ("Fear & Greed", fetch_fng),
        ("Yahoo (makro, złoto)", fetch_yahoo), ("DefiLlama (stablecoiny)", fetch_stable),
        ("OKX (funding, OI)", fetch_okx), ("CoinGecko (dominacja)", fetch_global),
        ("Wikipedia (uwaga wg języków)", fetch_wiki), ("Upbit (premia kimchi)", fetch_kimchi),
        ("Deribit (DVOL, put/call)", fetch_deribit), ("CFTC (pozycje CME)", fetch_cftc),
        ("Coinbase (premia USA)", fetch_coinbase), ("Bybit (stawki za lewar)", fetch_bybit),
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
