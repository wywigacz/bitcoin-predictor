"""Szerokie śledzenie akcji i obligacji: wszystkie spółki S&P 500 + ~110 funduszy ETF (kraje, sektory, style, obligacje, surowce).

Z tego liczymy:
  * miary "pod maską" rynku (szerokość, rozproszenie, średnia korelacja, krzywa rentowności, spread kredytowy, relacje ETF),
  * sieć powiązań z BTC (z czym porusza się w tych samych tygodniach),
  * analogie struktury powiązań: kiedy zestaw korelacji BTC z rynkami wyglądał jak dziś.
Surowe ceny (duży plik) nie trafiają do repozytorium; w chmurze są pobierane od nowa przy każdej aktualizacji."""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C

RAW = C.DATA_DIR / "markets.csv.gz"
FEAT = C.DATA_DIR / "market_features.csv"
SP500 = C.DATA_DIR / "sp500.csv"
UA = {"User-Agent": "Mozilla/5.0 (btc-predictor)"}

ETF = {
    # obligacje
    "SHY": "Obligacje USA 1-3 lata", "IEI": "Obligacje USA 3-7 lat", "IEF": "Obligacje USA 7-10 lat", "TLT": "Obligacje USA 20+ lat",
    "TIP": "Obligacje USA indeksowane inflacją", "AGG": "Obligacje USA (cały rynek)", "LQD": "Obligacje korporacyjne (inwestycyjne)",
    "HYG": "Obligacje śmieciowe (wysokie ryzyko)", "JNK": "Obligacje śmieciowe (SPDR)", "EMB": "Obligacje rynków wschodzących (USD)",
    "BNDX": "Obligacje spoza USA", "BWX": "Obligacje rządów świata", "MUB": "Obligacje komunalne USA", "MBB": "Obligacje hipoteczne USA",
    "BKLN": "Kredyty lewarowane", "FLOT": "Obligacje o zmiennym oprocentowaniu",
    # szeroki rynek i style
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Małe spółki USA", "DIA": "Dow Jones", "RSP": "S&P 500 równoważony", "MDY": "Średnie spółki USA",
    "IWD": "Spółki wartościowe", "IWF": "Spółki wzrostowe", "MTUM": "Momentum", "QUAL": "Jakość", "USMV": "Niska zmienność",
    "SPHB": "Wysoka beta", "SPLV": "Niska zmienność S&P",
    # sektory
    "XLB": "Surowce (sektor)", "XLC": "Komunikacja", "XLE": "Energetyka", "XLF": "Finanse", "XLI": "Przemysł", "XLK": "Technologia",
    "XLP": "Dobra podstawowe", "XLRE": "Nieruchomości", "XLU": "Użyteczność publiczna", "XLV": "Ochrona zdrowia", "XLY": "Dobra luksusowe",
    # tematy
    "SMH": "Półprzewodniki", "SOXX": "Półprzewodniki (iShares)", "IGV": "Oprogramowanie", "ARKK": "Innowacje (ARK)", "KWEB": "Chiński internet",
    "IBB": "Biotechnologia", "XBI": "Biotechnologia (małe)", "ITB": "Budownictwo mieszkaniowe", "KRE": "Banki regionalne", "KBE": "Banki",
    "XRT": "Handel detaliczny", "JETS": "Linie lotnicze", "ICLN": "Czysta energia", "TAN": "Energia słoneczna", "URA": "Uran",
    "LIT": "Lit i baterie", "GDX": "Kopalnie złota", "SIL": "Kopalnie srebra", "BLOK": "Spółki blockchain", "WGMI": "Kopalnie bitcoina",
    # kraje
    "EWJ": "Japonia", "EWZ": "Brazylia", "EWG": "Niemcy", "EWU": "Wielka Brytania", "EWQ": "Francja", "EWI": "Włochy", "EWP": "Hiszpania",
    "EWC": "Kanada", "EWA": "Australia", "EWY": "Korea Płd.", "EWT": "Tajwan", "EWH": "Hongkong", "FXI": "Chiny (duże)", "MCHI": "Chiny (MSCI)",
    "INDA": "Indie", "EWW": "Meksyk", "TUR": "Turcja", "EZA": "RPA", "EIS": "Izrael", "KSA": "Arabia Saudyjska", "ARGT": "Argentyna",
    "VNM": "Wietnam", "EPOL": "Polska", "EWS": "Singapur", "EIDO": "Indonezja", "THD": "Tajlandia", "EEM": "Rynki wschodzące", "VGK": "Europa",
    # surowce i waluty
    "GLD": "Złoto", "SLV": "Srebro", "USO": "Ropa", "UNG": "Gaz ziemny", "DBC": "Koszyk surowców", "DBA": "Rolnictwo", "CPER": "Miedź",
    "UUP": "Dolar amerykański", "FXE": "Euro", "FXY": "Jen", "FXF": "Frank szwajcarski",
}
RATIOS = {  # relacje, które często opisują nastroje (rynek ryzyka vs bezpieczne przystanie)
    "credit": ("HYG", "IEF", "Apetyt na ryzyko w obligacjach (śmieciowe vs skarbowe)"),
    "term": ("TLT", "SHY", "Długie vs krótkie obligacje"),
    "infl": ("TIP", "IEF", "Oczekiwania inflacyjne (TIPS vs skarbowe)"),
    "small": ("IWM", "SPY", "Małe vs duże spółki"),
    "growth": ("IWF", "IWD", "Wzrostowe vs wartościowe"),
    "beta": ("SPHB", "SPLV", "Wysoka beta vs niska zmienność"),
    "em": ("EEM", "SPY", "Rynki wschodzące vs USA"),
    "semis": ("SMH", "SPY", "Półprzewodniki vs rynek"),
    "banks": ("KRE", "SPY", "Banki regionalne vs rynek"),
    "homes": ("ITB", "SPY", "Budownictwo vs rynek"),
    "cugold": ("CPER", "GLD", "Miedź vs złoto (koniunktura)"),
    "miners": ("GDX", "GLD", "Kopalnie złota vs złoto"),
    "discr": ("XLY", "XLP", "Dobra luksusowe vs podstawowe"),
    "china": ("KWEB", "SPY", "Chiński internet vs USA"),
}
LEVELS = {"breadth200": "Odsetek spółek S&P 500 powyżej średniej 200 dni", "breadth50": "Odsetek spółek powyżej średniej 50 dni",
          "dispersion": "Rozproszenie zwrotów spółek (30 dni)", "avgcorr": "Jak mocno akcje poruszają się razem (średnia korelacja 60 dni)",
          "updown": "Odsetek spółek na plusie w tygodniu", "curve": "Krzywa rentowności (10 lat minus 3 miesiące)"}


def sp500_list():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers={"User-Agent": "btc-predictor/1.0 (research; wywigacz@users.noreply.github.com)"}, timeout=40)
        t = pd.read_html(io.StringIO(r.text))[0][["Symbol", "Security", "GICS Sector"]]
        t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
        t.to_csv(SP500, index=False)
        return t
    except Exception:
        return pd.read_csv(SP500) if SP500.exists() else pd.DataFrame(columns=["Symbol", "Security", "GICS Sector"])


def _yahoo(sym, p1, p2):
    for i in range(3):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}?period1={p1}&period2={p2}&interval=1d",
                             headers=UA, timeout=30)
            if r.status_code == 200:
                res = r.json()["chart"]["result"][0]
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
                s = pd.Series({pd.Timestamp(int(t), unit="s").normalize(): c for t, c in zip(res["timestamp"], adj) if c is not None}, dtype=float)
                return sym, s[~s.index.duplicated(keep="last")].sort_index()
            if r.status_code == 404:
                return sym, None
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return sym, None


def update(workers=8, verbose=True):
    sp = sp500_list()
    syms = list(ETF) + sp["Symbol"].tolist()
    p1, p2 = 1313625600, int(time.time()) + 86400
    t0 = time.time()
    out, fails = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, s in ex.map(lambda x: _yahoo(x, p1, p2), syms):
            if s is not None and len(s) > 200:
                out[sym] = s
            else:
                fails.append(sym)
    px = pd.DataFrame(out).sort_index()
    px = px.loc["2011-08-01":]
    C.DATA_DIR.mkdir(exist_ok=True)
    px.to_csv(RAW, float_format="%.6g", compression="gzip")
    if verbose:
        print(f"  rynki: {px.shape[1]} szeregów ({len(fails)} braków), {time.time() - t0:.0f}s", flush=True)
    return px, sp


def load():
    if not RAW.exists():
        return None
    return pd.read_csv(RAW, index_col=0, parse_dates=True, compression="gzip")


def features(px, sp):
    """Miary "pod maską" rynku, dzień po dniu (tylko z przeszłości)."""
    stocks = [s for s in sp["Symbol"] if s in px.columns]
    S = px[stocks]
    f = pd.DataFrame(index=px.index)
    f["breadth200"] = (S > S.rolling(200, min_periods=180).mean()).where(S.rolling(200, min_periods=180).mean().notna()).mean(axis=1) * 100
    f["breadth50"] = (S > S.rolling(50, min_periods=45).mean()).where(S.rolling(50, min_periods=45).mean().notna()).mean(axis=1) * 100
    lr = np.log(S).diff()
    f["dispersion"] = np.log(S / S.shift(21)).std(axis=1)
    f["updown"] = (S / S.shift(5) > 1).where(S.shift(5).notna()).mean(axis=1) * 100
    # średnia korelacja z wariancji portfela równoważonego: rho = (n^2 Var(EW) - sum var) / ((sum sd)^2 - sum var)
    w = 60
    v_i = lr.rolling(w, min_periods=40).var()
    sd_i = np.sqrt(v_i)
    ew = lr.mean(axis=1)
    n = lr.notna().sum(axis=1)
    v_ew = ew.rolling(w, min_periods=40).var()
    num = n ** 2 * v_ew - v_i.sum(axis=1)
    den = sd_i.sum(axis=1) ** 2 - v_i.sum(axis=1)
    f["avgcorr"] = (num / den).clip(-1, 1)
    for k, (a, b, _) in RATIOS.items():
        if a in px and b in px:
            f[f"ratio_{k}"] = px[a] / px[b]
    return f


def comove(df, px, sp, weeks=104):
    """Z czym BTC porusza się w tych samych tygodniach (korelacja zwrotów tygodniowych, ostatnie 2 lata)."""
    btc = df["close"].resample("W-FRI").last()
    P = px.resample("W-FRI").last()
    r_b = np.log(btc).diff().tail(weeks)
    R = np.log(P).diff().reindex(r_b.index)
    corr = R.apply(lambda s: s.corr(r_b) if s.notna().sum() >= weeks * 0.8 else np.nan).dropna()
    names = {**ETF, **dict(zip(sp["Symbol"], sp["Security"]))}
    sector = dict(zip(sp["Symbol"], sp["GICS Sector"]))
    tab = pd.DataFrame({"sym": corr.index, "r": corr.values})
    tab["name"] = tab["sym"].map(names)
    tab["kind"] = np.where(tab["sym"].isin(list(ETF)), "fundusz", "spółka")
    sec = tab[tab["kind"] == "spółka"].assign(sector=lambda t: t["sym"].map(sector)).groupby("sector")["r"].mean().sort_values(ascending=False)
    bonds = [k for k in ("SHY", "IEI", "IEF", "TLT", "TIP", "AGG", "LQD", "HYG", "JNK", "EMB", "BNDX", "MUB", "MBB", "BKLN") if k in corr.index]
    return {"top": tab.sort_values("r", ascending=False).head(15), "bottom": tab.sort_values("r").head(8),
            "sectors": sec, "bonds": tab[tab["sym"].isin(bonds)].sort_values("r", ascending=False), "n": int(len(tab))}


KEY = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "LQD", "GLD", "UUP", "EEM", "XLE", "XLF", "XLK", "SMH", "ARKK", "USO", "DBC", "FXY",
       "TIP", "EMB", "KRE", "XLU", "XLP", "EWJ", "FXI"]


def corr_analogs(df, px, win=90, top=8, gap=90):
    """'Analogie analogii': okresy, gdy zestaw korelacji BTC z 25 kluczowymi rynkami wyglądał jak dziś."""
    keys = [k for k in KEY if k in px.columns]
    P = px[keys].reindex(df.index).ffill(limit=4)
    rb = np.log(df["close"]).diff()
    R = np.log(P).diff()
    mask = R.abs().sum(axis=1) > 0                            # dni sesyjne
    rb, R = rb[mask], R[mask]
    cor = pd.DataFrame({k: rb.rolling(win, min_periods=int(win * 0.8)).corr(R[k]) for k in keys}).dropna(how="any")
    if len(cor) < 500:
        return None
    cur = cor.iloc[-1].to_numpy()
    c = df["close"]
    f30, f90 = (c.shift(-30) / c - 1) * 100, (c.shift(-90) / c - 1) * 100
    X = cor.to_numpy()
    dist = np.sqrt(((X - cur) ** 2).mean(axis=1))
    order = np.argsort(dist)
    picked = []
    last_known = c.index[-91]
    for i in order:
        d = cor.index[i]
        if d > last_known:
            continue
        if all(abs((d - p).days) > gap for p in picked):
            picked.append(d)
        if len(picked) == top:
            break
    rows = [{"date": d.strftime("%Y-%m-%d"), "f30": float(f30.loc[d]), "f90": float(f90.loc[d])} for d in picked]
    now = {k: float(v) for k, v in cor.iloc[-1].items()}
    return {"rows": rows, "now": now, "med30": float(np.median([r["f30"] for r in rows])), "med90": float(np.median([r["f90"] for r in rows])),
            "up30": float(np.mean([r["f30"] > 0 for r in rows])), "n_keys": len(keys)}
