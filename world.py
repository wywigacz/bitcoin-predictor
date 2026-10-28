"""Szeroki zbiór szeregów ze świata (rynki, towary, waluty, uwaga polityczna, pogoda, Słońce, ziemia, księżyc)
do skanera zależności. Wszystko darmowe i bez kluczy. Awaria pojedynczego szeregu jest pomijana."""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C
import markets as MK

WORLD_CSV = C.DATA_DIR / "world.csv"
UA = {"User-Agent": "btc-predictor/1.0 (research; wywigacz@users.noreply.github.com)"}

# nazwa -> (symbol Yahoo, etykieta)
YAHOO = {
    # giełdy świata
    "nikkei": ("^N225", "Nikkei (Japonia)"), "hsi": ("^HSI", "Hang Seng (Hongkong)"), "shanghai": ("000001.SS", "Szanghaj (Chiny)"),
    "kospi": ("^KS11", "KOSPI (Korea Płd.)"), "ftse": ("^FTSE", "FTSE 100 (W. Brytania)"), "dax": ("^GDAXI", "DAX (Niemcy)"),
    "cac": ("^FCHI", "CAC 40 (Francja)"), "bovespa": ("^BVSP", "Bovespa (Brazylia)"), "nifty": ("^NSEI", "Nifty (Indie)"),
    "taiex": ("^TWII", "TAIEX (Tajwan)"), "asx": ("^AXJO", "ASX 200 (Australia)"), "eem": ("EEM", "Rynki wschodzące"),
    "tur": ("TUR", "Turcja (ETF)"), "argt": ("ARGT", "Argentyna (ETF)"), "eza": ("EZA", "RPA (ETF)"), "ewz": ("EWZ", "Brazylia (ETF)"),
    "ewj": ("EWJ", "Japonia (ETF)"), "fxi": ("FXI", "Chiny large cap (ETF)"), "inda": ("INDA", "Indie (ETF)"),
    # waluty
    "try": ("TRY=X", "USD/TRY (lira)"), "ars": ("ARS=X", "USD/ARS (peso arg.)"), "ngn": ("NGN=X", "USD/NGN (naira)"),
    "pln": ("PLN=X", "USD/PLN (złoty)"), "brl": ("BRL=X", "USD/BRL"), "eur": ("EURUSD=X", "EUR/USD"), "gbp": ("GBPUSD=X", "GBP/USD"), "aud": ("AUDUSD=X", "AUD/USD"),
    "cad": ("CAD=X", "USD/CAD"), "chf": ("CHF=X", "USD/CHF"), "rub": ("RUB=X", "USD/RUB"), "inr": ("INR=X", "USD/INR"),
    "zar": ("ZAR=X", "USD/ZAR"), "mxn": ("MXN=X", "USD/MXN"), "cnh": ("CNY=X", "USD/CNY"), "idr": ("IDR=X", "USD/IDR"),
    # surowce i energia
    "natgas": ("NG=F", "Gaz ziemny"), "copper": ("HG=F", "Miedź"), "silver": ("SI=F", "Srebro"), "platinum": ("PL=F", "Platyna"),
    "wheat": ("ZW=F", "Pszenica"), "corn": ("ZC=F", "Kukurydza"), "sugar": ("SB=F", "Cukier"), "coffee": ("KC=F", "Kawa"),
    "uranium": ("URA", "Uran (ETF)"), "brent": ("BZ=F", "Ropa Brent"),
    # sektory, obligacje, kredyt, zmienność
    "xle": ("XLE", "Energetyka USA"), "xlf": ("XLF", "Finanse USA"), "xlk": ("XLK", "Technologia USA"), "xlu": ("XLU", "Użyteczności USA"),
    "xlv": ("XLV", "Zdrowie USA"), "defense": ("ITA", "Obronność USA"), "clean": ("ICLN", "Czysta energia"), "regbank": ("KRE", "Banki regionalne USA"),
    "semis": ("SMH", "Półprzewodniki"), "arkk": ("ARKK", "ARK Innovation"), "gdx": ("GDX", "Górnicy złota"), "vnq": ("VNQ", "Nieruchomości USA"),
    "tip": ("TIP", "Obligacje indeksowane inflacją"), "lqd": ("LQD", "Obligacje korp. inwest."), "jnk": ("JNK", "Obligacje śmieciowe"),
    "tyx": ("^TYX", "Rentowność 30-latek"), "fvx": ("^FVX", "Rentowność 5-latek"), "vvix": ("^VVIX", "VVIX (zmienność zmienności)"),
    "skew": ("^SKEW", "CBOE SKEW"), "russell": ("^RUT", "Russell 2000"), "nvda": ("NVDA", "Nvidia"),
    # spółki powiązane z krypto
    "coin": ("COIN", "Coinbase"), "mstr": ("MSTR", "Strategy (MSTR)"), "mara": ("MARA", "Marathon (górnik)"), "riot": ("RIOT", "Riot (górnik)"),
}
# nazwa -> (artykuł Wikipedii EN, etykieta)
WIKI = {
    "trump": ("Donald_Trump", "Uwaga: Donald Trump"), "biden": ("Joe_Biden", "Uwaga: Joe Biden"), "recession": ("Recession", "Uwaga: recesja"),
    "inflation": ("Inflation", "Uwaga: inflacja"), "fed": ("Federal_Reserve", "Uwaga: Fed"), "crash": ("Stock_market_crash", "Uwaga: krach giełdowy"),
    "bankrun": ("Bank_run", "Uwaga: run na banki"), "russia": ("Russia", "Uwaga: Rosja"), "ukraine": ("Ukraine", "Uwaga: Ukraina"),
    "china": ("China", "Uwaga: Chiny"), "iran": ("Iran", "Uwaga: Iran"), "israel": ("Israel", "Uwaga: Izrael"), "taiwan": ("Taiwan", "Uwaga: Tajwan"),
    "musk": ("Elon_Musk", "Uwaga: Elon Musk"), "crypto": ("Cryptocurrency", "Uwaga: kryptowaluty"), "ethereum": ("Ethereum", "Uwaga: Ethereum"),
    "gold": ("Gold", "Uwaga: złoto"), "covid": ("COVID-19", "Uwaga: COVID-19"), "tariff": ("Tariff", "Uwaga: cła"),
    "oilprice": ("Price_of_oil", "Uwaga: ceny ropy"), "heatwave": ("Heat_wave", "Uwaga: fale upałów"), "hurricane": ("Tropical_cyclone", "Uwaga: cyklony"),
    "earthquake": ("Earthquake", "Uwaga: trzęsienia ziemi"), "war": ("War", "Uwaga: wojna"), "sanctions": ("International_sanctions", "Uwaga: sankcje"),
    "ai": ("Artificial_intelligence", "Uwaga: sztuczna inteligencja"), "nft": ("Non-fungible_token", "Uwaga: NFT"), "scam": ("Ponzi_scheme", "Uwaga: piramidy finansowe"),
    "unemployment": ("Unemployment", "Uwaga: bezrobocie"), "debt": ("National_debt_of_the_United_States", "Uwaga: dług USA"),
}
# nazwa -> (szer., dł., etykieta) regiony związane z kopaniem i popytem na energię
WEATHER = {
    "texas": (32.45, -99.7, "Teksas (kopanie)"), "sichuan": (30.6, 104.0, "Syczuan (Chiny)"), "xinjiang": (43.8, 87.6, "Sinciang (Chiny)"),
    "kazakh": (51.2, 71.4, "Kazachstan"), "iceland": (64.1, -21.9, "Islandia"), "siberia": (52.3, 104.3, "Syberia (Irkuck)"),
    "nyc": (40.7, -74.0, "Nowy Jork"), "tokyo": (35.7, 139.7, "Tokio"), "seoul": (37.6, 127.0, "Seul"),
}
LABELS = {f"y_{k}": v[1] for k, v in YAHOO.items()}
LABELS.update({f"wk_{k}": v[1] for k, v in WIKI.items()})
LABELS.update({f"wx_{k}_t": f"Temperatura: {v[2]}" for k, v in WEATHER.items()})
LABELS.update({f"wx_{k}_p": f"Opady: {v[2]}" for k, v in WEATHER.items()})
LABELS.update({"sunspots": "Plamy słoneczne", "quakes": "Trzęsienia ziemi M6+ (liczba)", "moon_sin": "Faza księżyca (sin)", "moon_cos": "Faza księżyca (cos)",
               "season_sin": "Sezonowość roczna (sin)", "season_cos": "Sezonowość roczna (cos)"})
LABELS.update({f"dow_{i}": f"Dzień tygodnia: {d}" for i, d in enumerate(["pon", "wt", "śr", "czw", "pt", "sob", "niedz"])})
WL = {"n_total": "wszystkie wydarzenia", "c_armed": "konflikty zbrojne i ataki", "c_biz": "biznes i gospodarka", "c_disaster": "klęski i wypadki",
      "c_intl": "stosunki międzynarodowe", "c_law": "prawo i przestępczość", "c_pol": "polityka i wybory", "c_health": "zdrowie i środowisko",
      "c_sci": "nauka i technologia", "k_war": "wojna", "k_terror": "terroryzm", "k_deaths": "ofiary śmiertelne", "k_economy": "gospodarka (inflacja, stopy)",
      "k_markets": "rynki i giełdy", "k_finstress": "stres finansowy (banki, długi)", "k_sanctions": "sankcje i cła", "k_elections": "wybory",
      "k_power": "zmiany władzy", "k_naturaldis": "klęski żywiołowe", "k_pandemic": "pandemie i choroby", "k_protest": "protesty i strajki",
      "k_energy": "energia i surowce", "k_tech": "technologie i big tech", "k_crypto": "kryptowaluty", "k_china": "Chiny", "k_ukraine": "Rosja i Ukraina",
      "k_mideast": "Bliski Wschód", "k_usa": "USA", "k_europe": "Europa"}
LABELS.update({f"we_{k}": f"Świat: {v}" for k, v in WL.items()})
LABELS.update({f"m_{k}": f"{v} (fundusz)" for k, v in MK.ETF.items()})
LABELS.update({f"mk_ratio_{k}": v[2] for k, v in MK.RATIOS.items()})
LABELS.update({f"mk_{k}": v for k, v in MK.LEVELS.items()})
try:
    _sp = pd.read_csv(MK.SP500)
    LABELS.update({f"s_{a}": f"{b} (akcja)" for a, b in zip(_sp["Symbol"], _sp["Security"])})
except Exception:
    pass
ATTN = {f"wk_{k}" for k in WIKI} | {"quakes", "sunspots"} | {f"we_{k}" for k in WL}
WX = {f"wx_{k}_t" for k in WEATHER} | {f"wx_{k}_p" for k in WEATHER}
STATIC = {"moon_sin", "moon_cos", "season_sin", "season_cos"} | {f"dow_{i}" for i in range(7)}


def _get(url, timeout=30, tries=2):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.0 * (i + 1))
    return None


def _day(ts):
    return pd.Timestamp(int(ts), unit="s").normalize()


def _yahoo(item, p1, p2):
    name, (sym, _) = item
    r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}?period1={p1}&period2={p2}&interval=1d")
    if r is None:
        return name, None
    try:
        res = r.json()["chart"]["result"][0]
        s = pd.Series({_day(t): c for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]) if c is not None}, dtype=float)
        return f"y_{name}", s.sort_index()
    except Exception:
        return name, None


def _wiki(item, start, end):
    name, (art, _) = item
    r = _get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{art}/daily/{start}/{end}", timeout=40)
    if r is None:
        return name, None
    try:
        return f"wk_{name}", pd.Series({pd.Timestamp(i["timestamp"][:8]): float(i["views"]) for i in r.json()["items"]}).sort_index()
    except Exception:
        return name, None


def _weather(item, start, end):
    name, (la, lo, _) = item
    r = _get(f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&start_date={start}&end_date={end}"
             "&daily=temperature_2m_mean,precipitation_sum&timezone=UTC", timeout=60)
    if r is None:
        return []
    try:
        d = r.json()["daily"]
        t = pd.to_datetime(d["time"])
        return [(f"wx_{name}_t", pd.Series(d["temperature_2m_mean"], index=t, dtype=float)),
                (f"wx_{name}_p", pd.Series(d["precipitation_sum"], index=t, dtype=float))]
    except Exception:
        return []


def fetch_world(since=None):
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = pd.Timestamp("2011-08-01") if since is None else pd.Timestamp(since) - pd.Timedelta(days=10)
    p1, p2 = int(start.timestamp()), int(time.time()) + 86400
    out, fails = {}, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, s in ex.map(lambda it: _yahoo(it, p1, p2), YAHOO.items()):
            (out.__setitem__(k, s) if s is not None else fails.append(k))
    ws = max(start, pd.Timestamp("2015-07-01")).strftime("%Y%m%d")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, s in ex.map(lambda it: _wiki(it, ws, today.strftime("%Y%m%d")), WIKI.items()):
            (out.__setitem__(k, s) if s is not None else fails.append(k))
    for item in WEATHER.items():
        for k, s in _weather(item, start.strftime("%Y-%m-%d"), (today - pd.Timedelta(days=6)).strftime("%Y-%m-%d")):
            out[k] = s
        time.sleep(0.3)
    r = _get("https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv", timeout=60)
    if r is not None:
        try:
            df = pd.read_csv(io.StringIO(r.text), sep=";", header=None, usecols=[0, 1, 2, 4], names=["y", "m", "d", "sn"])
            df = df[df["sn"] >= 0]
            out["sunspots"] = pd.Series(df["sn"].values, index=pd.to_datetime(dict(year=df["y"], month=df["m"], day=df["d"])), dtype=float)
        except Exception:
            fails.append("sunspots")
    r = _get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start:%Y-%m-%d}&endtime={today:%Y-%m-%d}&minmagnitude=6&orderby=time-asc", timeout=60)
    if r is not None:
        try:
            q = pd.read_csv(io.StringIO(r.text), usecols=["time"])
            cnt = pd.to_datetime(q["time"]).dt.tz_localize(None).dt.normalize().value_counts()
            idx = pd.date_range(start, today)
            out["quakes"] = cnt.reindex(idx, fill_value=0).astype(float)
        except Exception:
            fails.append("quakes")
    return out, fails


def update_world(full=False):
    old = pd.read_csv(WORLD_CSV, index_col=0, parse_dates=True) if WORLD_CSV.exists() else pd.DataFrame()
    missing = [f"y_{k}" for k in YAHOO if f"y_{k}" not in old.columns]       # nowy szereg: pobierz całą historię
    since = None if (full or old.empty or missing) else old.index.max() - pd.Timedelta(days=30)
    data, fails = fetch_world(since)
    new = pd.DataFrame(data)
    merged = new.combine_first(old) if len(old) else new
    merged = merged.sort_index().loc["2011-08-01":]
    C.DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(WORLD_CSV, float_format="%.8g")
    return merged, fails


def static_series(index):
    d = pd.DataFrame(index=index)
    days = (index - pd.Timestamp("2000-01-06 18:14")).days.to_numpy(float)
    ph = (days / 29.530588853) % 1
    d["moon_sin"], d["moon_cos"] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
    doy = index.dayofyear.to_numpy(float)
    d["season_sin"], d["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for i in range(7):
        d[f"dow_{i}"] = (index.dayofweek == i).astype(float)
    return d


def transforms(raw, index):
    """Stacjonarne transformacje (zwroty, log-ilorazy uwagi, anomalie pogody) wyrównane do osi BTC. Bez podglądania przyszłości."""
    F = {}
    try:                                                   # dziennik świata (Wikipedia) jako dodatkowe szeregi natężenia
        wl = pd.read_csv(C.DATA_DIR / "worldlog_daily.csv", index_col=0, parse_dates=True)
        raw = raw.join(wl[[c for c in WL if c in wl.columns]].rename(columns=lambda c: f"we_{c}"), how="outer")
    except Exception:
        pass
    try:                                                   # akcje S&P 500, fundusze ETF i miary rynku
        px = MK.load()
        if px is not None:
            sp = set(pd.read_csv(MK.SP500)["Symbol"]) if MK.SP500.exists() else set()
            ren = {c: (f"s_{c}" if c in sp else f"m_{c}") for c in px.columns}
            raw = raw.join(px.rename(columns=ren), how="outer")
        if MK.FEAT.exists():
            mf = pd.read_csv(MK.FEAT, index_col=0, parse_dates=True)
            raw = raw.join(mf.rename(columns=lambda c: f"mk_{c}"), how="outer")
    except Exception:
        pass
    raw = raw.reindex(index.union(raw.index)).sort_index()
    for c in raw.columns:
        s = raw[c]
        if c.startswith(("y_", "m_", "mk_ratio_")):
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            ls = np.log(s)
            if not c.startswith("mk_"):
                F[f"{c}|r1"] = ls.diff(1)
            F[f"{c}|r7"], F[f"{c}|r30"] = ls.diff(7), ls.diff(30)
        elif c.startswith("s_"):
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            F[f"{c}|r30"] = np.log(s).diff(30)
        elif c.startswith("mk_"):
            s = s.ffill(limit=5).reindex(index)
            F[f"{c}|d7"] = s - s.shift(7)
            F[f"{c}|dev"] = s - s.rolling(365, min_periods=200).mean()
        elif c in ATTN:
            s = s.reindex(index)
            if c == "sunspots":
                s = s.ffill(limit=5)
            ls = np.log1p(s.clip(lower=0))
            F[f"{c}|a7"] = ls.rolling(7).mean() - ls.rolling(90, min_periods=60).mean()
            F[f"{c}|a30"] = ls.rolling(30).mean() - ls.rolling(365, min_periods=200).mean()
        elif c in WX:
            s = s.reindex(index)
            doy = s.groupby(s.index.dayofyear).transform(lambda x: x.expanding(min_periods=2).mean().shift(1))
            an = (s - doy)
            F[f"{c}|w7"] = an.rolling(7).mean()
            F[f"{c}|w30"] = an.rolling(30).mean()
    st = static_series(index)
    for c in st.columns:
        F[f"{c}|s"] = st[c]
    return pd.DataFrame(F, index=index)
