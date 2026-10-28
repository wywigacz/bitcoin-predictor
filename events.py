"""Baza wydarzeń: ręczne + kalendarz (FOMC, opcje, rynek pracy) + automatycznie wykrywane szoki rynkowe.
Do tego atrybucja dużych ruchów BTC do wydarzeń i event study (typowa reakcja wg kategorii)."""
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

import config as C
import events_data as E

FOMC_CSV = C.DATA_DIR / "fomc.csv"
MONTHS = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                           "August", "September", "October", "November", "December"])}
WINDOWS = {1: "dzień", 3: "3 dni", 7: "7 dni", 14: "14 dni", 30: "30 dni"}


# ---------------------------------------------------------------- kalendarz
def _mon(s):
    s = s.strip()
    for name, n in MONTHS.items():
        if name.lower().startswith(s[:3].lower()):
            return n
    return None


def _fetch_fomc(year_hist=None):
    h = {"User-Agent": "Mozilla/5.0 btc-predictor"}
    out = []
    if year_hist is None:
        t = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=h, timeout=30).text
        year = None
        pat = re.compile(r"(\d{4}) FOMC Meetings|fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", re.S)
        for m in pat.finditer(t):
            if m.group(1):
                year = int(m.group(1))
                continue
            mon, days = m.group(2), m.group(3).replace("*", "").strip()
            nums = re.findall(r"\d+", days)
            if not nums or year is None:
                continue
            last = int(nums[-1])
            months = [x for x in re.split(r"[/\-]", mon) if x.strip()]
            mo = _mon(months[-1]) if len(months) > 1 and len(nums) > 1 and int(nums[-1]) < int(nums[0]) else _mon(months[0])
            if mo:
                out.append(date(year, mo, last))
    else:
        t = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year_hist}.htm", headers=h, timeout=30).text
        pat = re.compile(r"<h5[^>]*>\s*(\w+)\s+(\d{1,2})(?:-(?:(\w+)\s+)?(\d{1,2}))?\*?\s+Meeting\s*-\s*(\d{4})", re.S)
        for m in pat.finditer(t):
            mo = _mon(m.group(3) or m.group(1))
            if mo:
                out.append(date(int(m.group(5)), mo, int(m.group(4) or m.group(2))))
    return out


def fomc_dates():
    """Dni decyzji FOMC (cache w data/fomc.csv). Awaria scrapowania nie zatrzymuje systemu."""
    have = set()
    if FOMC_CSV.exists():
        have = set(pd.to_datetime(pd.read_csv(FOMC_CSV)["date"]).dt.date)
    try:
        have |= set(_fetch_fomc())
        years = {d.year for d in have}
        for y in range(2011, 2021):
            if y not in years:
                have |= set(_fetch_fomc(y))
    except Exception as e:
        print("  [uwaga] FOMC:", str(e)[:100])
    if have:
        pd.DataFrame({"date": sorted(have)}).to_csv(FOMC_CSV, index=False)
    return sorted(have)


def _last_friday(y, m):
    d = date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1)
    while d.weekday() != 4:
        d -= timedelta(days=1)
    return d


def _first_friday(y, m):
    d = date(y, m, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def calendar_events(last_day):
    rows = []
    for d in fomc_dates():
        rows.append((pd.Timestamp(d), "fomc", "Decyzja FOMC", 0, 2, "calendar"))
    for y in range(2016, last_day.year + 2):
        for m in (3, 6, 9, 12):
            rows.append((pd.Timestamp(_last_friday(y, m)), "opex", "Kwartalne wygasanie opcji i futures", 0, 1, "calendar"))
    for y in range(2012, last_day.year + 2):
        for m in range(1, 13):
            rows.append((pd.Timestamp(_first_friday(y, m)), "nfp", "Raport z rynku pracy USA (pierwszy piątek)", 0, 1, "calendar"))
    for y in (2012, 2014, 2018, 2022):
        rows.append((pd.Timestamp({2012: "2012-11-06", 2014: "2014-11-04", 2018: "2018-11-06", 2022: "2022-11-08"}[y]),
                     "politics", "Wybory w USA" + (" (prezydenckie)" if y == 2012 else " (połówkowe)"), 0, 1, "calendar"))
    return rows


# ---------------------------------------------------------------- auto-wykrywanie szoków
def _declutter(mask, gap):
    out, last = [], None
    for d in mask.index[mask.fillna(False).to_numpy(bool)]:
        if last is None or (d - last).days >= gap:
            out.append(d)
            last = d
    return out


def detect_shocks(df):
    ev = []

    def col(n):
        return df[n].where(df[n] > 0) if n in df.columns else None

    def add(mask, cat, title, gap=10):
        for d in _declutter(mask, gap):
            ev.append((d, cat, title, 0, 1, "auto"))

    vix = col("vix")
    if vix is not None:
        add((vix >= 30) & (vix.shift(1).rolling(20).max() < 30), "auto_vix", "VIX przekracza 30", 20)
        add((vix / vix.shift(1) - 1) >= 0.35, "auto_vix", "Skok VIX o ponad 35% w dzień", 10)
    spx = col("spx")
    if spx is not None:
        d1, d5 = spx / spx.shift(1) - 1, spx / spx.shift(5) - 1
        add((d1 <= -0.03) | (d5 <= -0.07), "auto_eq_dn", "Gwałtowny spadek S&P 500", 10)
        add((d1 >= 0.04) | (d5 >= 0.07), "auto_eq_up", "Gwałtowny wzrost S&P 500", 10)
    dxy = col("dxy")
    if dxy is not None:
        d5 = dxy / dxy.shift(5) - 1
        add(d5 >= 0.022, "auto_usd_up", "Gwałtowne umocnienie dolara", 10)
        add(d5 <= -0.022, "auto_usd_dn", "Gwałtowne osłabienie dolara", 10)
    tnx = col("tnx")
    if tnx is not None:
        d5 = tnx / tnx.shift(5) - 1
        add(d5 >= 0.08, "auto_yld_up", "Skok rentowności 10-latek", 10)
        add(d5 <= -0.08, "auto_yld_dn", "Spadek rentowności 10-latek", 10)
    oil = col("oil")
    if oil is not None:
        d5 = oil / oil.shift(5) - 1
        add(d5 >= 0.12, "auto_oil_up", "Skok cen ropy", 10)
        add(d5 <= -0.12, "auto_oil_dn", "Spadek cen ropy", 10)
    gold = col("gold")
    if gold is not None:
        add((gold / gold.shift(5) - 1) >= 0.05, "auto_gold_up", "Skok cen złota", 10)
    hyg = col("hyg")
    if hyg is not None:
        add((hyg / hyg.shift(5) - 1) <= -0.025, "auto_credit", "Stres na rynku obligacji korporacyjnych", 10)
    jpy = col("usdjpy")
    if jpy is not None:
        add((jpy / jpy.shift(5) - 1) <= -0.03, "auto_yen", "Gwałtowne umocnienie jena", 10)
    return ev


HACKS_CSV = C.DATA_DIR / "hacks.csv"


def hack_events(min_usd=1e8):
    """Duże włamania i kradzieże (DefiLlama /hacks, keyless) jako zdarzenia kategorii 'hack'."""
    try:
        h = requests.get("https://api.llama.fi/hacks", headers={"User-Agent": "btc-predictor"}, timeout=40).json()
        df = pd.DataFrame([{"date": pd.Timestamp(int(x["date"]), unit="s").normalize(), "name": x["name"], "usd": x.get("amount") or 0}
                           for x in h if x.get("date")])
        df.to_csv(HACKS_CSV, index=False)
    except Exception as e:
        print("  [uwaga] hacks:", str(e)[:100])
        if not HACKS_CSV.exists():
            return []
        df = pd.read_csv(HACKS_CSV, parse_dates=["date"])
    df = df[df["usd"] >= min_usd]
    return [(r.date, "hack", f"Włamanie: {r.name} (~{r.usd/1e6:,.0f} mln USD)", -1, 2 if r.usd >= 3e8 else 1, "auto")
            for r in df.itertuples()]


# ---------------------------------------------------------------- budowa bazy
def build_events(df):
    rows = []
    for d, cat, title, sign, imp in E.CURATED:
        rows.append((pd.Timestamp(d), cat, title, sign, imp, "curated"))
    first, last = df.index.min(), df.index.max()
    rows += hack_events()
    rows += calendar_events(last)
    rows += detect_shocks(df)
    rows += [(pd.Timestamp(d), c, t, s, i, "upcoming") for d, c, t, s, i in E.UPCOMING]
    ev = pd.DataFrame(rows, columns=["date", "cat", "title", "sign", "imp", "kind"]).drop_duplicates(["date", "cat", "title"])
    return ev.sort_values("date").reset_index(drop=True)


def _sigma(c):
    r = np.log(c).diff()
    return np.sqrt((r ** 2).ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean())


def study(df, ev):
    """Event study: zwrot BTC po wydarzeniu, standaryzowany zmiennością sprzed zdarzenia."""
    c, sig = df["close"], _sigma(df["close"])
    pos_idx = df.index.get_indexer(ev["date"], method="bfill")
    ev = ev.copy()
    n = len(df)
    lc = np.log(c.to_numpy())
    for k in WINDOWS:
        z = np.full(len(ev), np.nan)
        r = np.full(len(ev), np.nan)
        for j, p in enumerate(pos_idx):
            if p < 31 or p + k - 1 >= n or ev["date"].iloc[j] > df.index[-1]:
                continue
            base = lc[p - 1]
            ret = lc[p + k - 1] - base
            s = sig.iloc[p - 1]
            if s > 0:
                z[j] = ret / (s * np.sqrt(k))
                r[j] = (np.exp(ret) - 1) * 100
        ev[f"z{k}"], ev[f"r{k}"] = z, r
    ev["pos"] = pos_idx
    return ev


def category_table(ev):
    rows = []
    for cat, g in ev.groupby("cat"):
        for k in WINDOWS:
            z = g[f"z{k}"].dropna()
            if len(z) < 3:
                continue
            t = z.mean() / (z.std(ddof=1) / np.sqrt(len(z))) if z.std(ddof=1) > 0 else 0.0
            rows.append({"cat": cat, "window": k, "n": len(z), "mean_z": float(z.mean()), "median_z": float(z.median()),
                         "pos": float((z > 0).mean()), "t": float(t), "mean_ret": float(g[f"r{k}"].dropna().mean())})
    return pd.DataFrame(rows)


def attribute_moves(df, ev, cat_tab, thr=2.0, limit=60):
    c = df["close"]
    sig = _sigma(c)
    r = np.log(c).diff()
    z = (r / sig.shift(1)).dropna()
    big = z[z.abs() >= thr]
    clusters, cur = [], []
    for d in big.index:
        if cur and (d - cur[-1]).days <= 1:
            cur.append(d)
        else:
            if cur:
                clusters.append(cur)
            cur = [d]
    if cur:
        clusters.append(cur)

    prior = {}
    if len(cat_tab):
        w3 = cat_tab[cat_tab["window"] == 3]
        prior = {row.cat: abs(row.t) for row in w3.itertuples()}

    def candidates(d):
        out = []
        for e in ev.itertuples():
            lag = (d - e.date).days
            if e.kind == "upcoming" or abs(lag) > (2 if e.kind == "auto" else 1):
                continue
            score = e.imp + (2.0 if e.kind == "curated" else 0.0) + (1.0 if lag == 0 else 0.0) + min(prior.get(e.cat, 0), 3) * 0.3
            out.append((score, e))
        out.sort(key=lambda x: -x[0])
        return out

    moves = []
    for cl in clusters[-limit:] if limit else clusters:
        d0 = cl[0]
        ret = (c.loc[cl[-1]] / c.shift(1).loc[d0] - 1) * 100
        cand = candidates(d0)
        moves.append({"date": d0.strftime("%Y-%m-%d"), "ret": float(ret), "z": float(z.loc[cl].sum() / np.sqrt(len(cl))),
                      "causes": [{"cat": e.cat, "title": e.title, "kind": e.kind, "date": e.date.strftime("%Y-%m-%d")} for _, e in cand[:3]]})
    # placebo: jaki odsetek losowych dni ma jakiegokolwiek kandydata (miara napompowania "wyjaśnień")
    rng = np.random.default_rng(7)
    pool = z.index[z.index > "2013-01-01"]
    pick = rng.choice(len(pool), size=min(400, len(pool)), replace=False)
    base = np.mean([len(candidates(pool[i])) > 0 for i in pick])
    allc = [cl[0] for cl in clusters]
    expl = np.mean([any(e.kind == "curated" for _, e in candidates(d)) for d in allc]) if allc else 0
    anyc = np.mean([len(candidates(d)) > 0 for d in allc]) if allc else 0
    return moves[::-1], {"n_moves": len(clusters), "explained_curated": float(expl), "explained_any": float(anyc), "placebo_any": float(base)}


def upcoming(ev, cat_tab, last, days=120):
    out = []
    horizon = last + pd.Timedelta(days=days)
    fut = ev[(ev["date"] > last) & (ev["date"] <= horizon) & (ev["cat"] != "nfp")]
    for e in fut.itertuples():
        g = cat_tab[(cat_tab["cat"] == e.cat) & (cat_tab["window"] == 3)]
        stat = None
        if len(g):
            row = g.iloc[0]
            stat = {"n": int(row["n"]), "mean_ret": float(row["mean_ret"]), "pos": float(row["pos"]), "t": float(row["t"])}
        out.append({"date": e.date.strftime("%Y-%m-%d"), "cat": e.cat, "title": e.title, "hist": stat})
    return out


def analyze_events(df):
    ev = build_events(df)
    ev = study(df, ev)
    tab = category_table(ev)
    moves, cover = attribute_moves(df, ev, tab)
    return {"events": ev, "table": tab, "moves": moves, "cover": cover, "upcoming": upcoming(ev, tab, df.index[-1])}
