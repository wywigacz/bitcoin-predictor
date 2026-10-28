"""Nagłówki prasowe z dni dużych ruchów ceny: GDELT (wiadomości o Bitcoinie) i dziennik wydarzeń światowych z Wikipedii.

Uwaga metodologiczna: nagłówki dobieramy PO fakcie, dla dni z dużym ruchem, więc służą do opisu ("co się wtedy działo"),
a nie do dowodzenia przyczyny ani do prognoz (byłby to błąd selekcji)."""
import json
import re
import time

import numpy as np
import pandas as pd
import requests

import config as C

CACHE = C.DATA_DIR / "news_cache.json"
UA = {"User-Agent": "btc-predictor/1.0 (research; wywigacz@users.noreply.github.com)"}
GDELT_FROM = pd.Timestamp("2017-01-01")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

CATS = {
    "reg_neg": r"\b(ban|bans|banned|crackdown|lawsuit|sues|sued|probe|investigat\w+|charges?|charged|illegal|outlaw\w*|restrict\w*|warns?)\b",
    "reg_pos": r"\b(approv\w+|legal tender|regulatory clarity|framework|green light|adopts?|recogni[sz]\w+|legali[sz]\w+)\b",
    "etf": r"\b(etf|etfs|futures|spot bitcoin|blackrock|fidelity|grayscale|bakkt|cme)\b",
    "hack": r"\b(hack\w*|stolen|exploit\w*|breach\w*|heist|theft|drained)\b",
    "contagion": r"\b(bankrupt\w*|insolven\w*|collapse\w*|halts? withdrawals?|freezes? withdrawals?|ftx|celsius|terra|luna|3ac|contagion)\b",
    "leverage": r"\b(liquidat\w+|margin call\w*|squeeze|leverag\w+|wiped out)\b",
    "fed": r"\b(fed|federal reserve|interest rates?|rate (hike|cut)s?|powell|fomc|monetary policy|qe|quantitative)\b",
    "macro": r"\b(inflation|cpi|recession|tariffs?|trade war|bank run|svb|silicon valley|stocks?|nasdaq|wall street|dollar|bond yields?|yen|banking)\b",
    "geo": r"\b(war|invasion|invades?|missiles?|attack\w*|strikes?|iran|russia|ukraine|israel|hamas|sanctions?|military)\b",
    "china": r"\b(china|chinese|pboc|beijing|hong kong)\b",
    "inst": r"\b(tesla|musk|microstrategy|strategy|saylor|paypal|coinbase|square|buys? bitcoin|adds? bitcoin|treasury|corporate)\b",
    "supply": r"\b(sells?|sold|dump\w*|selling|mt\.? gox|repayments?|government (sells|moves))\b",
    "politics": r"\b(trump|biden|election|senate|congress|white house|president|lawmakers?)\b",
    "tech": r"\b(fork|upgrade|halving|taproot|segwit|lightning|ethereum merge)\b",
    "adoption": r"\b(el salvador|adoption|accepts? bitcoin|payments?)\b",
}
CATRX = {k: re.compile(v, re.I) for k, v in CATS.items()}
WIKI_KW = re.compile(r"\b(bitcoin|crypto\w*|bank\w*|market\w*|stocks?|price|tariff\w*|sanction\w*|war|attack\w*|election\w*|president|inflation|oil|rate|fed|econom\w*|trade|currency|dollar|debt|crisis|default|protest\w*|coup|ceasefire|invasion|strike)\b", re.I)


def load():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save(c):
    C.DATA_DIR.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")


def _norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()[:70]


def fetch_gdelt(day, tries=4):
    d = day.strftime("%Y%m%d")
    url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin%20sourcelang:english&mode=artlist&maxrecords=40&format=json&sort=hybridrel"
           f"&startdatetime={d}000000&enddatetime={d}235959")
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                arts = r.json().get("articles", [])
                out, seen = [], set()
                for a in arts:
                    t = (a.get("title") or "").strip()
                    k = _norm(t)
                    if len(t) < 25 or k in seen:
                        continue
                    seen.add(k)
                    out.append({"t": t[:160], "d": a.get("domain", "")})
                return out
            if r.status_code == 200:      # pusta odpowiedź (brak artykułów)
                return []
        except requests.RequestException:
            pass
        time.sleep(15 * (i + 1))
    return None


def fetch_wiki_day(day):
    page = f"Portal:Current_events/{day.year}_{MONTHS[day.month - 1]}_{day.day}"
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "parse", "page": page, "prop": "text", "format": "json", "formatversion": 2},
                         headers=UA, timeout=40)
        if r.status_code != 200:
            return None
        html_ = r.json()["parse"]["text"]
    except Exception:
        return None
    items = []
    for m in re.finditer(r"<li>(.*?)</li>", html_, flags=re.S):
        raw = m.group(1)
        if "<ul>" in raw or "<li>" in raw:
            continue
        t = re.sub(r"<[^>]+>", "", raw)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"\s*\([^()]*\)\s*$", "", t)
        if len(t) >= 40:
            items.append(t[:200])
    scored = sorted(items, key=lambda x: -len(WIKI_KW.findall(x)))
    return scored[:6]


def classify(titles):
    """Które rodzaje wydarzeń wynikają z nagłówków (liczba nagłówków pasujących do kategorii)."""
    hits = {k: sum(1 for t in titles if rx.search(t)) for k, rx in CATRX.items()}
    need = 2 if len(titles) >= 6 else 1
    out = sorted(((k, v) for k, v in hits.items() if v >= need), key=lambda x: -x[1])[:3]
    return [{"cat": k, "n": v} for k, v in out]


def big_move_days(df, thr=2.0):
    """Pierwszy dzień każdej serii dużych ruchów (|z| >= thr) wraz z ruchem i z."""
    c = df["close"]
    r = np.log(c).diff()
    sig = np.sqrt((r ** 2).ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean())
    z = (r / sig.shift(1)).dropna()
    big = z[z.abs() >= thr]
    days, cur = [], []
    for d in big.index:
        if cur and (d - cur[-1]).days <= 1:
            cur.append(d)
        else:
            if cur:
                days.append(cur)
            cur = [d]
    if cur:
        days.append(cur)
    out = []
    for cl in days:
        top = max(cl, key=lambda d: abs(z.loc[d]))
        out.append({"day": top, "z": float(z.loc[top]), "first": cl[0]})
    return out


def update_news(df, budget_s=600, thr=2.0, pause=7.0, verbose=True):
    """Dociąga brakujące nagłówki (najpierw największe ruchy, potem nowsze). Zatrzymuje się po budget_s sekundach."""
    cache = load()
    moves = big_move_days(df, thr)
    todo = sorted(moves, key=lambda m: (-abs(m["z"]),))
    t0 = time.time()
    done = 0
    for m in todo:
        day = m["day"]
        key = day.strftime("%Y-%m-%d")
        ent = cache.get(key, {})
        need_g = day >= GDELT_FROM and ent.get("g") is None
        need_w = ent.get("w") is None
        if not (need_g or need_w):
            continue
        if time.time() - t0 > budget_s:
            break
        if need_w:
            w = fetch_wiki_day(day)
            if w is not None:
                ent["w"] = w
        if need_g:
            g = fetch_gdelt(day)
            if g is not None:
                ent["g"] = g
            time.sleep(pause)
        cache[key] = ent
        done += 1
        if done % 5 == 0:
            save(cache)
            if verbose:
                print(f"  nagłówki: {done} dni pobrano, {time.time() - t0:.0f}s", flush=True)
    save(cache)
    return cache


def for_day(cache, day):
    e = cache.get(pd.Timestamp(day).strftime("%Y-%m-%d"))
    if not e:
        return None
    g = e.get("g") or []
    return {"g": g[:6], "w": (e.get("w") or [])[:4], "cats": classify([x["t"] for x in g]) if g else []}
