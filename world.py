"""Szeroki zbiór szeregów ze świata (rynki, towary, waluty, uwaga polityczna, pogoda, Słońce, ziemia, księżyc)
do skanera zależności. Wszystko darmowe i bez kluczy. Awaria pojedynczego szeregu jest pomijana."""
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C
import extra as XT
import markets as MK

WORLD_CSV = C.DATA_DIR / "world.csv"
FAILS = C.DATA_DIR / "world_fails.json"
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
    # indeksy giełd świata
    "i_dji": ("^DJI", "Dow Jones"), "i_tsx": ("^GSPTSE", "TSX (Kanada)"), "i_mxx": ("^MXX", "IPC (Meksyk)"), "i_merv": ("^MERV", "Merval (Argentyna)"),
    "i_stoxx": ("^STOXX50E", "Euro Stoxx 50"), "i_aex": ("^AEX", "AEX (Holandia)"), "i_ibex": ("^IBEX", "IBEX 35 (Hiszpania)"), "i_mib": ("FTSEMIB.MI", "FTSE MIB (Włochy)"),
    "i_smi": ("^SSMI", "SMI (Szwajcaria)"), "i_omx": ("^OMX", "OMX (Szwecja)"), "i_bfx": ("^BFX", "BEL 20 (Belgia)"), "i_psi": ("PSI20.LS", "PSI (Portugalia)"),
    "i_atx": ("^ATX", "ATX (Austria)"), "i_wig": ("WIG.WA", "WIG (Polska)"), "i_szse": ("399001.SZ", "Shenzhen (Chiny)"), "i_sti": ("^STI", "STI (Singapur)"),
    "i_klse": ("^KLSE", "KLCI (Malezja)"), "i_jkse": ("^JKSE", "Jakarta (Indonezja)"), "i_bse": ("^BSESN", "Sensex (Indie)"), "i_nz": ("^NZ50", "NZX 50 (Nowa Zelandia)"),
    "i_ta": ("^TA125.TA", "TA-125 (Izrael)"), "i_xu": ("XU100.IS", "BIST 100 (Turcja)"), "i_jse": ("^J203.JO", "JSE (RPA)"), "i_omxc": ("^OMXC25", "OMX C25 (Dania)"),
    "i_omxh": ("^OMXH25", "OMX H25 (Finlandia)"),
    # strach na różnych rynkach
    "v_vix9d": ("^VIX9D", "Strach na giełdzie 9 dni (VIX9D)"), "v_vix3m": ("^VIX3M", "Strach na giełdzie 3 miesiące (VIX3M)"), "v_vxn": ("^VXN", "Strach na Nasdaq (VXN)"),
    "v_ovx": ("^OVX", "Strach na rynku ropy (OVX)"), "v_gvz": ("^GVZ", "Strach na rynku złota (GVZ)"),
    # stopy i surowce
    "zq": ("ZQ=F", "Oczekiwania stóp Fed (kontrakty)"), "zt": ("ZT=F", "Obligacje 2-letnie (kontrakty)"), "palladium": ("PA=F", "Pallad"),
    "aluminium": ("ALI=F", "Aluminium"), "lumber": ("LBR=F", "Drewno"), "cocoa": ("CC=F", "Kakao"), "cotton": ("CT=F", "Bawełna"), "ojuice": ("OJ=F", "Sok pomarańczowy"),
    "cattle": ("LE=F", "Bydło"), "freight": ("BDRY", "Fracht morski (suchy)"),
    # waluty rynków wschodzących
    "krw": ("KRW=X", "USD/KRW (Korea)"), "twd": ("TWD=X", "USD/TWD (Tajwan)"), "thb": ("THB=X", "USD/THB (Tajlandia)"), "php": ("PHP=X", "USD/PHP (Filipiny)"),
    "egp": ("EGP=X", "USD/EGP (Egipt)"), "pkr": ("PKR=X", "USD/PKR (Pakistan)"), "cop": ("COP=X", "USD/COP (Kolumbia)"), "clp": ("CLP=X", "USD/CLP (Chile)"),
    "pen": ("PEN=X", "USD/PEN (Peru)"), "ils": ("ILS=X", "USD/ILS (Izrael)"), "hkd": ("HKD=X", "USD/HKD (Hongkong)"), "uah": ("UAH=X", "USD/UAH (Ukraina)"), "vnd": ("VND=X", "USD/VND (Wietnam)"),
    # inne kryptowaluty i kontrakty CME
    "btcf": ("BTC=F", "Kontrakty CME na Bitcoina"), "c_eth": ("ETH-USD", "Ethereum"), "c_sol": ("SOL-USD", "Solana"), "c_xrp": ("XRP-USD", "XRP"), "c_bnb": ("BNB-USD", "BNB"),
    "c_doge": ("DOGE-USD", "Dogecoin"), "c_ada": ("ADA-USD", "Cardano"), "c_ltc": ("LTC-USD", "Litecoin"), "c_trx": ("TRX-USD", "Tron"), "c_avax": ("AVAX-USD", "Avalanche"),
    "c_link": ("LINK-USD", "Chainlink"), "c_dot": ("DOT-USD", "Polkadot"), "c_xmr": ("XMR-USD", "Monero"), "c_bch": ("BCH-USD", "Bitcoin Cash"),
    # spółki powiązane z krypto
    "coin": ("COIN", "Coinbase"), "mstr": ("MSTR", "Strategy (MSTR)"), "mara": ("MARA", "Marathon (górnik)"), "riot": ("RIOT", "Riot (górnik)"),
    "clsk": ("CLSK", "CleanSpark (górnik)"), "hut": ("HUT", "Hut 8 (górnik)"), "cifr": ("CIFR", "Cipher (górnik)"), "iren": ("IREN", "IREN (górnik)"),
    "wulf": ("WULF", "TeraWulf (górnik)"), "xyz": ("XYZ", "Block (dawniej Square)"),
    # energia: koszt prądu dla górników i presja inflacyjna
    "ttf": ("TTF=F", "Gaz w Europie (TTF)"), "coal": ("MTF=F", "Węgiel (Rotterdam)"), "gasoline": ("RB=F", "Benzyna (USA)"), "heatoil": ("HO=F", "Olej opałowy (USA)"),
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
    # ludzie i instytucje
    "powell": ("Jerome_Powell", "Uwaga: Jerome Powell (Fed)"), "yellen": ("Janet_Yellen", "Uwaga: Janet Yellen"), "saylor": ("Michael_Saylor", "Uwaga: Michael Saylor"),
    "sbf": ("Sam_Bankman-Fried", "Uwaga: Sam Bankman-Fried"), "cz": ("Changpeng_Zhao", "Uwaga: Changpeng Zhao (Binance)"), "satoshi": ("Satoshi_Nakamoto", "Uwaga: Satoshi Nakamoto"),
    "buterin": ("Vitalik_Buterin", "Uwaga: Vitalik Buterin"), "gensler": ("Gary_Gensler", "Uwaga: Gary Gensler (SEC)"), "xi": ("Xi_Jinping", "Uwaga: Xi Jinping"),
    "putin": ("Vladimir_Putin", "Uwaga: Władimir Putin"), "bukele": ("Nayib_Bukele", "Uwaga: Nayib Bukele (Salwador)"), "milei": ("Javier_Milei", "Uwaga: Javier Milei"),
    "blackrock": ("BlackRock", "Uwaga: BlackRock"), "binance": ("Binance", "Uwaga: Binance"), "coinbase_w": ("Coinbase", "Uwaga: Coinbase"), "tether_w": ("Tether_(cryptocurrency)", "Uwaga: Tether"),
    "sec": ("U.S._Securities_and_Exchange_Commission", "Uwaga: SEC"), "imf": ("International_Monetary_Fund", "Uwaga: MFW"), "ecb": ("European_Central_Bank", "Uwaga: EBC"),
    "nvidia": ("Nvidia", "Uwaga: Nvidia"), "openai": ("OpenAI", "Uwaga: OpenAI"), "chatgpt": ("ChatGPT", "Uwaga: ChatGPT"), "tesla": ("Tesla,_Inc.", "Uwaga: Tesla"),
    # pojęcia finansowe i bańki
    "bubble": ("Economic_bubble", "Uwaga: bańki spekulacyjne"), "tulip": ("Tulip_mania", "Uwaga: tulipomania"), "dotcom": ("Dot-com_bubble", "Uwaga: bańka internetowa"),
    "gfc": ("2008_financial_crisis", "Uwaga: kryzys 2008"), "depression": ("Great_Depression", "Uwaga: Wielki Kryzys"), "hyperinf": ("Hyperinflation", "Uwaga: hiperinflacja"),
    "stagflation": ("Stagflation", "Uwaga: stagflacja"), "qe": ("Quantitative_easing", "Uwaga: luzowanie ilościowe"), "goldstd": ("Gold_standard", "Uwaga: standard złota"),
    "cbdc": ("Central_bank_digital_currency", "Uwaga: cyfrowe waluty banków centralnych"), "stablecoin": ("Stablecoin", "Uwaga: stablecoiny"),
    "halving": ("Bitcoin_halving", "Uwaga: halving"), "blockchain": ("Blockchain", "Uwaga: blockchain"), "defi": ("Decentralized_finance", "Uwaga: DeFi"),
    "mining": ("Cryptocurrency_mining", "Uwaga: kopanie krypto"), "etf": ("Exchange-traded_fund", "Uwaga: fundusze ETF"), "shortsq": ("Short_squeeze", "Uwaga: short squeeze"),
    "gamestop": ("GameStop_short_squeeze", "Uwaga: GameStop"), "ftx": ("FTX", "Uwaga: FTX"), "mtgox": ("Mt._Gox", "Uwaga: Mt. Gox"), "silk": ("Silk_Road_(marketplace)", "Uwaga: Silk Road"),
    "ransomware": ("Ransomware", "Uwaga: ransomware"), "hack": ("Security_hacker", "Uwaga: hakerzy"),
    # technologie, które mogą zagrozić lub pomóc BTC
    "quantum": ("Quantum_computing", "Uwaga: komputery kwantowe"), "metaverse": ("Metaverse", "Uwaga: metawersum"), "web3": ("Web3", "Uwaga: Web3"),
    # świat, klimat, katastrofy
    "climate": ("Climate_change", "Uwaga: zmiany klimatu"), "elnino": ("El_Niño–Southern_Oscillation", "Uwaga: El Niño"), "wildfire": ("Wildfire", "Uwaga: pożary"),
    "flood": ("Flood", "Uwaga: powodzie"), "drought": ("Drought", "Uwaga: susze"), "pandemic": ("Pandemic", "Uwaga: pandemie"), "nuclear": ("Nuclear_weapon", "Uwaga: broń jądrowa"),
    "ww3": ("World_War_III", "Uwaga: III wojna światowa"), "brexit": ("Brexit", "Uwaga: Brexit"), "venezuela": ("Venezuela", "Uwaga: Wenezuela"),
    "argentina": ("Argentina", "Uwaga: Argentyna"), "turkey": ("Turkey", "Uwaga: Turcja"), "nkorea": ("North_Korea", "Uwaga: Korea Północna"),
    "elsalvador": ("El_Salvador", "Uwaga: Salwador"), "india": ("India", "Uwaga: Indie"), "japan": ("Japan", "Uwaga: Japonia"),
    "solar": ("Solar_flare", "Uwaga: rozbłyski słoneczne"), "eclipse": ("Solar_eclipse", "Uwaga: zaćmienia Słońca"), "ufo": ("Unidentified_flying_object", "Uwaga: UFO"),
    "astrology": ("Astrology", "Uwaga: astrologia"), "mercury_r": ("Apparent_retrograde_motion", "Uwaga: retrogradacja Merkurego"),
    "worldcup": ("FIFA_World_Cup", "Uwaga: mundial"), "olympics": ("Olympic_Games", "Uwaga: igrzyska"), "superbowl": ("Super_Bowl", "Uwaga: Super Bowl"),
    "lottery": ("Lottery", "Uwaga: loterie"), "casino": ("Casino", "Uwaga: kasyna"), "poverty": ("Poverty", "Uwaga: ubóstwo"),
    "layoff": ("Layoff", "Uwaga: zwolnienia"), "housing": ("Real_estate_bubble", "Uwaga: bańka na nieruchomościach"), "bankrupt": ("Bankruptcy", "Uwaga: bankructwa"),
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
LABELS.update({"cal_monthend": "Kalendarz: koniec miesiąca", "cal_quarterend": "Kalendarz: koniec kwartału", "cal_lunarny": "Kalendarz: chiński Nowy Rok (±7 dni)",
               "cal_taxday": "Kalendarz: amerykański dzień podatkowy", "cal_xmas": "Kalendarz: okres świąteczny",
               "cal_mercury": "Dzika karta: retrogradacja Merkurego", "cal_fri13": "Dzika karta: piątek 13.", "cal_goldenweek": "Kalendarz: chiński Złoty Tydzień",
               "cal_thanksgiving": "Kalendarz: Święto Dziękczynienia (USA)", "cal_ramadan": "Kalendarz: ramadan", "cal_sport": "Dzika karta: mundial, igrzyska, Super Bowl",
               "cal_dst": "Kalendarz: zmiana czasu w USA"})
LABELS.update({k: v[0] for k, v in XT.KIND.items()})
DSER = {  # szeregi z tabeli dziennej BTC dorzucane do skanera: (etykieta, rodzaj)
    "d_ntx": ("Sieć BTC: liczba transakcji", "attn"), "d_fees": ("Sieć BTC: opłaty transakcyjne (USD)", "attn"), "d_mempool": ("Sieć BTC: kolejka transakcji (mempool)", "attn"),
    "d_txvol": ("Sieć BTC: wartość przesyłana w sieci", "attn"), "d_uaddr": ("Sieć BTC: unikalne adresy", "attn"), "d_difficulty": ("Sieć BTC: trudność kopania", "attn"),
    "d_hash": ("Sieć BTC: moc obliczeniowa", "attn"), "d_minerrev": ("Sieć BTC: przychody górników", "attn"), "d_adr": ("Sieć BTC: aktywne adresy", "attn"),
    "d_volume": ("Wolumen handlu BTC (Bitstamp)", "attn"), "d_cbprem": ("Premia Coinbase (popyt z USA)", "lvl"), "d_basis": ("Różnica ceny CME vs bieżąca", "lvl"),
    "d_bfund": ("Stawka za lewar (Bybit, 7 dni)", "lvl"), "d_mvrv": ("Zysk posiadaczy (MVRV)", "lvl"), "d_fng": ("Strach i chciwość (krypto)", "lvl"),
    "d_ethbtc": ("Ethereum względem Bitcoina", "ratio"), "d_altbtc": ("Koszyk altcoinów względem Bitcoina", "ratio"),
}
LABELS.update({k: v[0] for k, v in DSER.items()})
STATIC = {"moon_sin", "moon_cos", "season_sin", "season_cos", "cal_monthend", "cal_quarterend", "cal_lunarny", "cal_taxday", "cal_xmas"} | {f"dow_{i}" for i in range(7)}
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
    _it = pd.read_csv(MK.INTL_CSV)
    LABELS.update({f"g_{a}": f"{b} ({c})" for a, b, c in zip(_it["Symbol"], _it["Name"], _it["Index"])})
except Exception:
    pass
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
    tried = set(json.loads(FAILS.read_text())) if FAILS.exists() else set()   # szeregi, których źródło nie ma: nie ponawiamy pełnego pobierania w kółko
    missing = [c for c in [f"y_{k}" for k in YAHOO] + [f"wk_{k}" for k in WIKI] if c not in old.columns and c.split("_", 1)[1] not in tried]
    since = None if (full or old.empty or missing) else old.index.max() - pd.Timedelta(days=30)
    data, fails = fetch_world(since)
    if since is None:
        FAILS.write_text(json.dumps(sorted(fails)))
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
    # kalendarz: koniec miesiąca/kwartału, chiński Nowy Rok, amerykański dzień podatkowy, okres świąteczny
    me = index.to_series().dt.is_month_end
    d["cal_monthend"] = (index.to_series().shift(-3).dt.month != index.month).astype(float).to_numpy()
    d["cal_quarterend"] = ((index.month % 3 == 0) & (d["cal_monthend"] > 0)).astype(float)
    lny = pd.to_datetime(["2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25",
                          "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"])
    d["cal_lunarny"] = np.array([float(any(-7 <= (t - x).days <= 7 for x in lny)) for t in index])
    d["cal_taxday"] = ((index.month == 4) & (index.day >= 8) & (index.day <= 18)).astype(float)
    d["cal_xmas"] = (((index.month == 12) & (index.day >= 20)) | ((index.month == 1) & (index.day <= 3))).astype(float)
    # „dzikie karty”: rzeczy zapewne bez związku, ale ludzie w nie wierzą lub zmieniają wtedy zachowanie; sprawdzamy, żeby to wykluczyć albo zauważyć
    d["cal_mercury"] = mercury_retrograde(index)
    d["cal_fri13"] = ((index.day == 13) & (index.dayofweek == 4)).astype(float)
    d["cal_goldenweek"] = ((index.month == 10) & (index.day <= 7)).astype(float)                  # chiński Złoty Tydzień
    thx = {y: pd.Timestamp(f"{y}-11-01") + pd.offsets.Week(weekday=3) * 4 - pd.Timedelta(days=7 if pd.Timestamp(f"{y}-11-01").dayofweek == 3 else 0)
           for y in range(index.year.min(), index.year.max() + 1)}
    d["cal_thanksgiving"] = np.array([float(0 <= (t - thx[t.year]).days <= 4) for t in index])
    d["cal_ramadan"] = _in_windows(index, RAMADAN, 29)
    d["cal_sport"] = np.maximum(_in_ranges(index, WORLDCUP + OLYMPICS), _in_windows(index, SUPERBOWL, 1, before=3))
    dst = []
    for y in range(index.year.min(), index.year.max() + 1):                                          # zmiana czasu w USA (2. niedziela marca, 1. niedziela listopada)
        dst += [pd.Timestamp(f"{y}-03-01") + pd.offsets.Week(weekday=6) * 2 - pd.Timedelta(days=7 if pd.Timestamp(f"{y}-03-01").dayofweek == 6 else 0),
                pd.Timestamp(f"{y}-11-01") + pd.offsets.Week(weekday=6) - pd.Timedelta(days=7 if pd.Timestamp(f"{y}-11-01").dayofweek == 6 else 0)]
    d["cal_dst"] = _in_windows(index, dst, 2)
    return d


RAMADAN = ["2012-07-20", "2013-07-09", "2014-06-28", "2015-06-18", "2016-06-06", "2017-05-27", "2018-05-16", "2019-05-06", "2020-04-24",
           "2021-04-13", "2022-04-02", "2023-03-23", "2024-03-11", "2025-03-01", "2026-02-18", "2027-02-08"]
SUPERBOWL = ["2012-02-05", "2013-02-03", "2014-02-02", "2015-02-01", "2016-02-07", "2017-02-05", "2018-02-04", "2019-02-03", "2020-02-02",
             "2021-02-07", "2022-02-13", "2023-02-12", "2024-02-11", "2025-02-09", "2026-02-08", "2027-02-14"]
WORLDCUP = [("2014-06-12", "2014-07-13"), ("2018-06-14", "2018-07-15"), ("2022-11-20", "2022-12-18"), ("2026-06-11", "2026-07-19")]
OLYMPICS = [("2012-07-27", "2012-08-12"), ("2016-08-05", "2016-08-21"), ("2021-07-23", "2021-08-08"), ("2024-07-26", "2024-08-11"),
            ("2014-02-07", "2014-02-23"), ("2018-02-09", "2018-02-25"), ("2022-02-04", "2022-02-20"), ("2026-02-06", "2026-02-22")]


def _in_windows(index, starts, days, before=0):
    st = pd.to_datetime(pd.Index(starts))
    return np.array([float(any(-before <= (t - s).days < days for s in st)) for t in index])


def _in_ranges(index, ranges):
    return np.array([float(any(pd.Timestamp(a) <= t <= pd.Timestamp(b) for a, b in ranges)) for t in index])


def mercury_retrograde(index):
    """Pozorny ruch wsteczny Merkurego widziany z Ziemi: liczony z orbit Keplera (elementy J2000, dokładność ~1 dnia)."""
    def helio(t, a, e, i, L, w, O):
        M = np.radians((L - w) % 360)
        E = M.copy()
        for _ in range(8):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
        wp, Op, ip = np.radians(w - O), np.radians(O), np.radians(i)
        X = (np.cos(wp) * np.cos(Op) - np.sin(wp) * np.sin(Op) * np.cos(ip)) * x + (-np.sin(wp) * np.cos(Op) - np.cos(wp) * np.sin(Op) * np.cos(ip)) * y
        Y = (np.cos(wp) * np.sin(Op) + np.sin(wp) * np.cos(Op) * np.cos(ip)) * x + (-np.sin(wp) * np.sin(Op) + np.cos(wp) * np.cos(Op) * np.cos(ip)) * y
        return X, Y
    T = ((index - pd.Timestamp("2000-01-01 12:00")).days.to_numpy(float) + 0.5) / 36525
    mx, my = helio(T, 0.38709927, 0.20563593, 7.00497902, 252.25032350 + 149472.67411175 * T, 77.45779628 + 0.16047689 * T, 48.33076593 - 0.12534081 * T)
    ex, ey = helio(T, 1.00000261, 0.01671123, -0.00001531, 100.46457166 + 35999.37244981 * T, 102.93768193 + 0.32327364 * T, 0.0)
    lon = np.unwrap(np.arctan2(my - ey, mx - ex))
    return (np.gradient(lon) < 0).astype(float)


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
            etf = set(MK.ETF)
            ren = {c: (f"m_{c}" if c in etf else f"s_{c}" if c in sp else f"g_{c}") for c in px.columns}
            raw = raw.join(px.rename(columns=ren), how="outer")
        if MK.FEAT.exists():
            mf = pd.read_csv(MK.FEAT, index_col=0, parse_dates=True)
            raw = raw.join(mf.rename(columns=lambda c: f"mk_{c}"), how="outer")
    except Exception:
        pass
    try:                                                   # płynność, CFTC, DeFi, Lightning, Hacker News, pogoda kosmiczna
        if XT.OUT.exists():
            raw = raw.join(pd.read_csv(XT.OUT, index_col=0, parse_dates=True), how="outer")
    except Exception:
        pass
    raw = raw.reindex(index.union(raw.index)).sort_index()
    for c in raw.columns:
        s = raw[c]
        kind = DSER.get(c, XT.KIND.get(c, (None, None)))[1]
        if kind == "chg":
            ls = np.log(s.where(s > 0).ffill(limit=10).reindex(index))
            F[f"{c}|r7"], F[f"{c}|r30"] = ls.diff(7), ls.diff(30)
            continue
        if kind == "attn":
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            ls = np.log(s)
            F[f"{c}|a7"] = ls.rolling(7).mean() - ls.rolling(90, min_periods=60).mean()
            F[f"{c}|a30"] = ls.rolling(30).mean() - ls.rolling(365, min_periods=200).mean()
            continue
        if kind == "lvl":
            s = s.ffill(limit=5).reindex(index)
            F[f"{c}|d7"] = s - s.shift(7)
            F[f"{c}|dev"] = s - s.rolling(365, min_periods=120).mean()
            continue
        if kind == "ratio":
            ls = np.log(s.where(s > 0).ffill(limit=5).reindex(index))
            F[f"{c}|r7"], F[f"{c}|r30"] = ls.diff(7), ls.diff(30)
            continue
        if c.startswith("g_"):
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            F[f"{c}|r30"] = np.log(s).diff(30)
            continue
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
