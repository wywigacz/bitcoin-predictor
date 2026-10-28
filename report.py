"""Dashboard HTML: jeden samodzielny plik (CSS + JS + dane), działa offline."""
import html
import re
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

# ----------------------------------------------------------------------------------------------
# Dane pomocnicze
# ----------------------------------------------------------------------------------------------
GROUPS = [
    ("Wycena: cena na tle historii", "Porównujemy dzisiejszą cenę z tym, ile średnio zapłacili posiadacze i jaka była cena w ostatnich latach. To najlepiej sprawdzające się wskaźniki.", ["mvrv", "ma200w", "mayer"]),
    ("Kopalnie Bitcoina", "Kiedy kopanie przestaje się opłacać, część kopalni się poddaje. Historycznie bywało to okolicą dna cen.", ["puell", "hashribbon"]),
    ("Nastroje i pożyczone pieniądze", "Strach i chciwość inwestorów, zakłady na pożyczone pieniądze oraz ilość gotówki gotowej do kupowania.", ["fng", "funding", "stable"]),
    ("Gospodarka i pieniądze na świecie", "Od 2020 roku Bitcoin zachowuje się często jak ryzykowna akcja technologiczna, ale to się zmienia.", ["m2", "ndx", "dxy", "vix", "rates"]),
    ("Cykl 4-letni", "Co około 4 lata następuje „halving” (nagroda za kopanie spada o połowę). Historycznie ceny rosły po nim przez ok. 1,5 roku.", ["halving", "pi"]),
]
LABELS = {
    "mvrv": "Zysk posiadaczy (MVRV)", "ma200w": "Cena vs średnia z 4 lat", "mayer": "Cena vs średnia z 200 dni",
    "puell": "Opłacalność kopania", "hashribbon": "Poddawanie się kopalni", "fng": "Strach i chciwość",
    "funding": "Zakłady na wzrost (pożyczone)", "stable": "Gotówka w kryptowalutach", "m2": "Ilość pieniędzy w USA",
    "dxy": "Siła dolara", "ndx": "Giełda technologiczna", "vix": "Strach na giełdzie (VIX)", "rates": "Oprocentowanie pieniądza",
    "pi": "Ostrzeżenie o szczycie (Pi Cycle)", "halving": "Miejsce w cyklu 4-letnim",
}
SRC_NOTE = {
    "mvrv": "powyżej 1 = posiadacze średnio na plusie", "ma200w": "1,0 = równo ze średnią z 4 lat", "mayer": "1,0 = równo ze średnią z 200 dni",
    "puell": "niżej = kopanie mniej zarabia niż zwykle", "hashribbon": "czy moc kopalni ostatnio spada", "fng": "0 = paniczny strach, 100 = euforia",
    "funding": "dodatnie = więcej zakładów na wzrost", "stable": "wzrost = więcej gotówki do wydania", "m2": "więcej pieniędzy = zwykle lepiej dla ceny",
    "dxy": "mocniejszy dolar zwykle szkodzi", "ndx": "zmiana z ostatnich 3 miesięcy", "vix": "wyżej = więcej strachu", "rates": "zmiana z pół roku, punkty proc.",
    "pi": "poniżej 1 = brak ostrzeżenia", "halving": "dni od ostatniego halvingu",
}


def n(v, d=0, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{v:,.{d}f}".replace(",", " ").replace(".", ",")
    return s + suffix


def raw_value(k, fe):
    g = fe.get
    try:
        return {
            "mvrv": n(g("mvrv"), 2), "ma200w": n(g("ma200w"), 2, "×"), "mayer": n(g("mayer"), 2, "×"),
            "puell": n(g("puell"), 2), "hashribbon": "poddają się" if g("hashcap") == 1 else ("nie" if g("hashcap") == 0 else "–"),
            "fng": n(g("fng"), 0), "funding": n((g("funding") or float("nan")) * 100, 4, "%"),
            "stable": n((g("stable30") or float("nan")) * 100, 1, "%"), "m2": n((g("m2yoy") or float("nan")) * 100, 1, "% r/r") if g("m2yoy") is not None else "brak danych",
            "dxy": n((g("dxy30") or float("nan")) * 100, 1, "%"), "ndx": n((g("ndx90") or float("nan")) * 100, 1, "%"),
            "vix": n(g("vix"), 1), "rates": n(g("rates180"), 2, " p.p."), "pi": n(g("pi"), 2),
            "halving": n(g("halv_days"), 0, " dni"),
        }[k]
    except Exception:
        return "–"


def ser_list(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


# ----------------------------------------------------------------------------------------------
# Sekcje HTML
# ----------------------------------------------------------------------------------------------
def gauge_svg(v):
    """Półokrągły wskaźnik -1..+1."""
    cx, cy, r = 150, 150, 118
    ang = math.pi * (1 - (v + 1) / 2)              # +1 -> 0 rad (prawo), -1 -> pi (lewo)
    nx, ny = cx + (r - 22) * math.cos(ang), cy - (r - 22) * math.sin(ang)
    ticks = ""
    for i in range(0, 21):
        a = math.pi * (1 - i / 20)
        r1, r2 = (r + 8, r + 16) if i % 5 == 0 else (r + 8, r + 12)
        ticks += (f'<line x1="{cx + r1 * math.cos(a):.1f}" y1="{cy - r1 * math.sin(a):.1f}" '
                  f'x2="{cx + r2 * math.cos(a):.1f}" y2="{cy - r2 * math.sin(a):.1f}" class="gt"/>')
    return f"""<svg viewBox="0 0 300 185" class="gauge" role="img" aria-label="Wskaźnik złożony {v:+.2f}">
<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" style="stop-color:var(--red)"/><stop offset=".5" style="stop-color:var(--amber)"/><stop offset="1" style="stop-color:var(--green)"/></linearGradient></defs>
<path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="var(--track)" stroke-width="16" stroke-linecap="round"/>
<path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="url(#gg)" stroke-width="16" stroke-linecap="round" opacity=".92"/>
{ticks}
<g class="needle" style="--rot:{(v) * 90:.1f}deg"><line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-(r-22)}" stroke="var(--fg)" stroke-width="3" stroke-linecap="round"/></g>
<circle cx="{cx}" cy="{cy}" r="8" fill="var(--fg)"/><circle cx="{cx}" cy="{cy}" r="3" fill="var(--bg2)"/>
<text x="{cx-r}" y="{cy+22}" text-anchor="middle" class="gl">drogo</text>
<text x="{cx+r}" y="{cy+22}" text-anchor="middle" class="gl">tanio</text></svg>"""


def cycle_html(days):
    total = 1460
    zones = [(0, 365, "Wzrosty", "var(--blue)"), (365, 560, "Szczyt", "var(--amber)"),
             (560, 770, "Spadki", "var(--red)"), (770, 1000, "Dno", "var(--green)"), (1000, 1460, "Odbudowa", "var(--violet)")]
    seg = "".join(f'<div class="zone" style="flex:{b-a};--c:{c}" title="{nm}: dni {a}–{b} po halvingu"><span>{nm}</span></div>' for a, b, nm, c in zones)
    d = days if days is not None and not math.isnan(days) else 0
    cur = next((nm for a, b, nm, _ in zones if a <= d < b), "Odbudowa")
    pos = min(max(d / total, 0), 1) * 100
    return (f'<div class="cycle"><div class="zones">{seg}<div class="marker" style="left:{pos:.2f}%"><i></i></div></div>'
            f'<div class="cyc-foot">Minęło <b>{n(d)} dni</b> od ostatniego halvingu ({C.HALVINGS[-1]}). Etap cyklu: <b>{cur}</b>. '
            f'Następny halving ok. {C.NEXT_HALVING}.</div></div>')


def forecast_cards(res):
    cards = ""
    p0 = res["price"]
    for f in res["forecasts"]:
        q = res["quality"][f["horizon"]]
        lo, hi = math.log(f["p5"]), math.log(f["p95"])
        pos = lambda v: (math.log(v) - lo) / (hi - lo) * 100
        chg = (f["p50"] / p0 - 1) * 100
        up = f["prob_up"] * 100
        badge = ('<span class="badge on">wsparte modelami</span>' if q["active"]
                 else '<span class="badge">typowe wahania</span>')
        v = f.get("view")
        view = (f'<div class="view"><span class="muted sm">Co sugerują wskaźniki <i>(testy nie potwierdzają, że to działa)</i></span>'
                f'<b>{n(v["p50"])}</b><span class="muted sm"> · {(v["p50"]/p0-1)*100:+.1f}% · szansa wzrostu {v["prob_up"]*100:.0f}%</span></div>') if v else ""
        cards += f"""<article class="card fc">
<header><h3>Za {f['horizon']} dni</h3>{badge}</header>
<div class="muted sm">do {f['target']}</div>
<div class="fc-med">{n(f['p50'])}<small> USD</small></div>
<div class="fc-chg {'pos' if chg >= 0 else 'neg'}">{chg:+.1f}% względem dziś</div>
<div class="range"><div class="r90" style="left:{pos(f['p10']):.1f}%;width:{pos(f['p90'])-pos(f['p10']):.1f}%"></div>
<div class="r50" style="left:{pos(f['p25']):.1f}%;width:{pos(f['p75'])-pos(f['p25']):.1f}%"></div>
<div class="tick" style="left:{pos(f['p50']):.1f}%"></div><div class="now" style="left:{pos(p0):.1f}%" title="cena dziś"></div></div>
<div class="range-l"><span>{n(f['p10'])}</span><span class="muted">typowy zakres</span><span>{n(f['p90'])}</span></div>
<div class="pup"><div class="ring" style="--p:{up:.0f}"><b>{up:.0f}%</b></div><span class="muted sm">szansa, że cena<br>będzie wyższa niż dziś</span></div>
{view}</article>"""
    return cards


MODEL_LAB = {"naive": "Zgadywanie: cena zostaje taka sama", "comp": "Ocena „tani czy drogi”", "ridge": "Połączenie wielu wskaźników",
             "knn": "Szukanie podobnych dni z przeszłości", "gbq": "Samouczące się drzewa decyzji", "scan": "Szukanie powiązań na świecie"}
LANG = {"en": "Kraje anglojęzyczne", "de": "Niemcy i okolice", "fr": "Francja", "es": "Hiszpania i Ameryka Łac.", "ru": "Rosja",
        "ja": "Japonia", "ko": "Korea Południowa", "zh": "Chiny", "tr": "Turcja", "pl": "Polska", "pt": "Brazylia i Portugalia"}


def tys(v):
    return f"{v / 1000:.0f} tys."


def _stars(t, nobs):
    a = abs(t)
    if nobs < 8:
        return "mało danych"
    return "wysoka" if a >= 3 else "średnia" if a >= 2 else "niska"


def summary_html(res, ens, E):
    p0 = res["price"]
    fc = {f["horizon"]: f for f in res["forecasts"]}
    names = {7: "za tydzień", 30: "za miesiąc", 90: "za 3 miesiące", 180: "za pół roku"}
    items = "".join(f'<li><b>{names.get(H, f"za {H} dni")}:</b> od {tys(fc[H]["p10"])} do {tys(fc[H]["p90"])} USD</li>' for H in fc)
    none = all(h["alpha"] == 0 for h in ens["horizons"].values())
    if none:
        direction = ('<b>Czy cena pójdzie w górę, czy w dół? Tego nie umiemy wiarygodnie przewidzieć.</b> Sprawdziliśmy to na historii: '
                     'żadna z naszych metod nie była lepsza od prostego założenia „cena zostanie taka sama”. Szansa na wzrost to więc około 50 na 50.')
    else:
        act = ", ".join(str(H) for H, h in ens["horizons"].items() if h["alpha"] > 0)
        direction = (f'<b>Czy cena pójdzie w górę, czy w dół?</b> Na okresie {act} dni nasze metody okazały się lepsze od zgadywania i lekko przesuwają prognozę, '
                     'ale to wciąż bardzo niepewne. Na pozostałych okresach nie umiemy tego wiarygodnie przewidzieć.')
    nxt = next((u for u in E["upcoming"] if u["cat"] != "nfp"), None)
    ntxt = f'<li><b>Najbliższe ważne wydarzenie:</b> {html.escape(nxt["title"])} ({nxt["date"]}).</li>' if nxt else ""
    return (f'<section class="card summary reveal"><h3>W skrócie</h3><ul>'
            f'<li><b>Bitcoin kosztuje dziś {n(p0)} USD.</b> {res["regime"]}. <span class="muted">(Ocena rynku {res["composite"]:+.2f} w skali od −1 „drogo” do +1 „tanio”.)</span></li>'
            f'<li><b>Gdzie może być cena?</b> Podane zakresy obejmują prawdziwą cenę w około 4 przypadkach na 5:<ul class="inner">{items}</ul></li>'
            f'<li>{direction}</li>'
            f'<li><b>Co potrafimy dobrze:</b> oceniać, jak mocno cena może się wahać. W testach na historii nasze zakresy trafiały w około 8 na 10 przypadków.</li>'
            f'{ntxt}</ul><p class="muted sm" style="margin:10px 0 0">Poniżej jest szczegółowe uzasadnienie. Trudne słowa wyjaśniamy na dole strony w słowniczku. To nie jest porada inwestycyjna.</p></section>')


def glossary_html():
    g = [
        ("Halving", "Co około 4 lata nagroda dla ludzi „kopiących” Bitcoina spada o połowę. Historycznie po halvingu ceny rosły przez ok. 1,5 roku, choć każdy kolejny wzrost był mniejszy."),
        ("Typowy zakres („4 na 5”)", "Przedział cen, w którym cena zmieści się w około 80% przypadków. Jedna na 10 sytuacji będzie poniżej, jedna na 10 powyżej."),
        ("Zmienność", "Jak mocno cena skacze w górę i w dół. Dla Bitcoina typowo około 2% dziennie."),
        ("MVRV (zysk posiadaczy)", "Dzisiejsza cena podzielona przez średnią cenę, po jakiej posiadacze kupili swoje monety. Poniżej 1 większość jest na stracie, co historycznie bywało okolicą dołków."),
        ("Średnia cena zakupu", "Średnia cena, po jakiej ostatnio „przeniesiono” wszystkie istniejące monety. Cena rzadko spada długo poniżej tej linii."),
        ("Test na historii", "Sprawdzamy metodę tak, jakby była używana na bieżąco: w każdym dniu dostaje tylko dane, które były wtedy znane, i zgaduje, co będzie dalej."),
        ("„Cena zostanie taka sama”", "Prosty punkt odniesienia. Każda metoda musi go pokonać, inaczej nie ma sensu jej używać."),
        ("Szansa, że to przypadek", "Im niższa, tym pewniej przewaga metody jest prawdziwa. Wymagamy, żeby była mniejsza niż 10%."),
        ("Korelacja", "Miara tego, jak mocno dwie rzeczy poruszają się razem: +1 zawsze razem, 0 niezależnie, −1 w przeciwne strony."),
        ("FOMC", "Posiedzenie amerykańskiego banku centralnego (Fed), na którym decyduje o stopach procentowych."),
        ("Opcje i DVOL", "Opcje to „ubezpieczenia” od zmian ceny. DVOL mówi, ile inwestorzy za to ubezpieczenie płacą, czyli jak duże wahania oczekują."),
        ("Premia kimchi", "O ile Bitcoin w Korei Południowej jest droższy niż w reszcie świata. Dodatnia premia oznacza duży zapał koreańskich inwestorów."),
        ("Put/Call", "Stosunek zakładów na spadek ceny do zakładów na wzrost. Poniżej 1 przeważają zakłady na wzrost."),
        ("CME i CFTC", "CME to amerykańska giełda kontraktów terminowych, a CFTC urząd, który co tydzień publikuje, jak ustawiają się duże fundusze."),
        ("Reżim rynku", "Nasz model rozpoznaje, czy rynek jest teraz w trybie „spokojnym”, czy „nerwowym” (duże wahania)."),
        ("Trend potęgowy", "Długoterminowa linia, wokół której cena Bitcoina układała się przez 15 lat. Cena poniżej linii bywała okazją, powyżej przegrzaniem."),
        ("Wiele testów naraz", "Gdy sprawdza się tysiące rzeczy, część wygląda na powiązaną tylko przez przypadek, jak trafienie szóstki w loterii przy tysiącach zakładów. Dlatego stosujemy surowsze kryteria."),
    ]
    return "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in g)


def models_html(ens):
    out = ""
    for H, h in ens["horizons"].items():
        rows = ""
        for r in h["table"]:
            p = r["dm_p"]
            ptxt = "–" if p != p else f"{p*100:.0f}%"
            lab = MODEL_LAB.get(r["model"], "Zespół metod (wynik końcowy)" if r["model"] == "ens" else r["label"])
            if r["model"] == "naive":
                sk = '<td class="muted">punkt odniesienia</td>'
            else:
                sk = (f'<td class="muted">0,0%</td>' if abs(r["skill"]) < 0.0005 else f'<td class="{"pos" if r["skill"] > 0 else "neg"}">{r["skill"]*100:+.1f}%</td>')
            rows += (f'<tr class="{"ens" if r["model"] == "ens" else ""}"><td>{lab}</td>{sk}<td>{ptxt}</td>'
                     f'<td>{r["cover80"]*10:.1f} z 10</td></tr>')
        used = ", ".join(f'{MODEL_LAB[m].lower()} {v*100:.0f}%' for m, v in h["weights"].items() if v > 0.005 and m != "naive")
        state = (f'<span class="badge on">używamy: {used}</span>' if h["alpha"] > 0
                 else '<span class="badge">żadna metoda nie pokonała zgadywania</span>')
        out += (f'<div class="card q"><header><h3>Za {H} dni</h3>{state}</header>'
                f'<table class="tbl"><tr><th>Metoda</th><th title="O ile mniejsze błędy niż przy założeniu, że cena zostanie taka sama (plus = lepiej)">Lepsza od zgadywania?</th>'
                f'<th title="Szansa, że przewaga to tylko przypadek. Im niżej, tym pewniej.">Szansa, że to przypadek</th><th title="W ilu na 10 przypadków cena zmieściła się w typowym zakresie (cel: 8)">Trafienia w zakres</th></tr>{rows}</table>'
                f'<div class="muted sm" style="margin-top:8px">Sprawdzono na {h["n"]} sytuacjach z historii.</div></div>')
    return out


def regions_html(ens, df, res):
    last = lambda c: df[c].dropna().iloc[-1] if c in df and df[c].notna().any() else None
    tiles = []
    kim = df["btckrw"] / df["usdkrw"] / df["close"] - 1 if {"btckrw", "usdkrw"} <= set(df.columns) else None
    if kim is not None and kim.notna().any():
        k = kim.dropna()
        tiles.append(("Czy w Korei Bitcoin jest droższy niż na świecie?", f"{k.iloc[-1]*100:+.1f}%".replace(".", ","), f"średnio z ostatnich 90 dni: {k.tail(90).mean()*100:+.1f}%".replace(".", ","),
                      "Dodatnia wartość to tzw. premia kimchi: koreańscy inwestorzy kupują bardzo chętnie."))
    if last("dvol") is not None:
        vrp = ens["features_now"].get("dvol_vrp")
        tiles.append(("Jak duże wahania ceny oczekują inwestorzy?", f"{last('dvol'):.0f}", "indeks DVOL z giełdy opcji, w skali roku",
                      ("Inwestorzy płacą więcej niż zwykle za ubezpieczenie od wahań." if vrp is not None and vrp > 0.15 else "Ubezpieczenie od wahań kosztuje tyle, ile wynika z ostatnich ruchów." if vrp is not None else "")))
    if last("pcr") is not None:
        tiles.append(("Zakłady na spadek kontra na wzrost", f"{last('pcr'):.2f}".replace(".", ","), "stosunek zakładów na spadek do zakładów na wzrost",
                      "Poniżej 1 = przeważają zakłady na wzrost, powyżej 1 = na spadek."))
    if last("cot_lev") is not None:
        tiles.append(("Duże fundusze na giełdzie CME", f"{last('cot_lev')*100:+.0f}%", "ich pozycja netto w stosunku do całego rynku",
                      "Ujemna wartość często oznacza grę na różnicy cen, a nie zakład na spadek."))
    tiles.append(("Czy rynek jest teraz „nerwowy”?", f"{ens['hmm']['p_hi']*100:.0f}%", "szansa, że trwa okres dużych wahań", "Model rozpoznaje to wyłącznie na podstawie przeszłości."))
    pz = ens["powerlaw_z"]
    tiles.append(("Cena na tle długoterminowego trendu", ("poniżej" if pz < -0.3 else "powyżej" if pz > 0.3 else "w okolicy"), f"odchylenie: {pz:+.1f}".replace(".", ",") + " (w typowych jednostkach)",
                  "Poniżej linii trendu cena bywała okazją, a daleko powyżej przegrzaniem."))
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b><span class="muted sm">{c}</span><span class="muted sm" style="opacity:.85">{d}</span></div>' for a, b, c, d in tiles)
    rows = ""
    for k, name in LANG.items():
        c = f"wiki_{k}"
        if c not in df:
            continue
        s = df[c].dropna()
        if len(s) < 400:
            continue
        v = float(np.log(s.tail(30).mean() / s.tail(365).mean()))
        w = min(abs(v), 1.0) * 50
        rows += (f'<div class="sig sm-row"><div class="sig-l"><b>{name}</b></div><div class="dv"><i class="mid"></i>'
                 f'<span class="{"g" if v >= 0 else "r"}" style="left:{50 if v >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                 f'<div class="sig-v"><b>{(math.exp(v) - 1) * 100:+.0f}%</b></div></div>')
    return tl, rows


def _catlab(c):
    import events_data
    return events_data.CATEGORIES.get(c, c)


_WL_CAT = {"biz": "Gospodarka", "pol": "Polityka", "armed": "Konflikty", "intl": "Stosunki międzynarodowe", "disaster": "Klęski", "law": "Prawo i przestępczość",
           "health": "Zdrowie", "sci": "Nauka i technologia"}


def _move_row(m, cache):
    import news
    import worldlog
    d = m["date"]
    chips = "".join(f'<span class="cause {x["kind"]}" title="{html.escape(x["title"])} ({x["date"]})">{html.escape(_catlab(x["cat"]))}</span>' for x in m["causes"])
    nd = news.for_day(cache, d)
    wh = worldlog.day_headlines(d)
    detail = ""
    if nd and nd["cats"]:
        chips += "".join(f'<span class="cause news" title="Tyle nagłówków o Bitcoinie z tego dnia dotyczyło tego tematu">{html.escape(_catlab(c["cat"]))} ×{c["n"]}</span>' for c in nd["cats"])
    if nd and nd["g"]:
        detail += '<div class="hl"><b>Nagłówki prasowe o Bitcoinie z tego dnia</b><ul>' + "".join(
            f'<li>{html.escape(x["t"])} <span class="muted">({html.escape(x["d"])})</span></li>' for x in nd["g"][:5]) + "</ul></div>"
    if wh:
        detail += '<div class="hl"><b>Co działo się na świecie tego dnia</b><ul>' + "".join(
            f'<li><span class="muted">{_WL_CAT.get(x["c"], x["c"])}:</span> {html.escape(x["t"])}</li>' for x in wh) + "</ul></div>"
    cnt = (len(nd["g"][:5]) if nd and nd["g"] else 0) + len(wh)
    cell = chips or '<span class="muted sm">brak wydarzeń z naszej listy</span>'
    if detail:
        cell += f'<details class="hl-d"><summary>Pokaż, co się działo ({cnt})</summary>{detail}</details>'
    return f'<tr><td>{d}</td><td class="{"pos" if m["ret"] >= 0 else "neg"}"><b>{m["ret"]:+.1f}%</b></td><td>{cell}</td></tr>'


def moves_html(E):
    import news
    cache = news.load()
    head = '<table class="tbl"><tr><th>Data</th><th>Zmiana ceny</th><th>Co się działo (wydarzenia i nagłówki)</th></tr>'
    first = "".join(_move_row(m, cache) for m in E["moves"][:10])
    rest = "".join(_move_row(m, cache) for m in E["moves"][10:50])
    c = E["cover"]
    note = (f'Przeanalizowaliśmy {c["n_moves"]} dużych skoków ceny w historii. Przy każdym zapisujemy, co działo się wtedy na świecie: wydarzenia z naszej listy, '
            f'nagłówki prasowe o Bitcoinie (z archiwum GDELT) i dziennik wydarzeń światowych z Wikipedii. Nie twierdzimy, że to jest przyczyna skoku. '
            f'Chodzi o to, żeby mieć gęstą oś czasu świata i szukać na niej powtarzających się wzorców (patrz „Szukanie powiązań”). '
            f'Dla porównania: nawet dla losowo wybranych dni jakieś wydarzenie z listy znajduje się w {c["placebo_any"]*100:.0f}% przypadków.')
    more = f'<details class="more"><summary>Pokaż kolejne skoki</summary>{head}{rest}</table></details>' if rest else ""
    return f'{head}{first}</table>{more}<p class="muted sm" style="margin-top:10px">{note}</p>'


def study_html(E):
    t = E["table"]
    t = t[(t["window"] == 3) & (t["n"] >= 5)].copy()
    t["abs"] = t["t"].abs()
    t = t.sort_values("abs", ascending=False).head(16)
    rows = ""
    for r in t.itertuples():
        w = min(abs(r.t) / 5, 1) * 50
        sig = abs(r.t) >= 2 and r.n >= 8
        rows += (f'<tr class="{"sigrow" if sig else ""}"><td>{html.escape(_catlab(r.cat))}</td><td>{r.n}</td>'
                 f'<td class="{"pos" if r.mean_ret >= 0 else "neg"}">{r.mean_ret:+.1f}%</td><td>{r.pos*100:.0f}%</td>'
                 f'<td><div class="dv"><i class="mid"></i><span class="{"g" if r.t >= 0 else "r"}" style="left:{50 if r.t >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div></td>'
                 f'<td>{_stars(r.t, r.n)}</td></tr>')
    return (f'<table class="tbl"><tr><th>Rodzaj wydarzenia</th><th>Ile razy</th><th>Średnia zmiana ceny po 3 dniach</th><th>Jak często cena rosła</th><th></th><th>Pewność wyniku</th></tr>{rows}</table>'
            f'<p class="muted sm" style="margin-top:10px">Liczymy zmianę ceny od końca dnia przed wydarzeniem do końca trzeciego dnia po nim. Pogrubione wiersze mają wysoką lub średnią pewność. '
            f'Gdy rodzajów wydarzeń jest dużo, kilka wyników wychodzi „pewnych” przypadkiem. '
            f'Dla wydarzeń wykrywanych automatycznie z rynków (spadek akcji, skok strachu na giełdzie, dolar itd.) to głównie ruchy <b>w tych samych dniach</b>, a nie dowód, że Bitcoin reaguje później.</p>')


FEAT_LAB = {"comp": "Ogólna ocena rynku", "val": "Czy Bitcoin jest tani (wycena)", "macro": "Gospodarka i pieniądze na świecie",
            "sent": "Nastroje i pożyczone pieniądze", "cycle": "Miejsce w cyklu 4-letnim", "mom30": "Zmiana ceny z 30 dni", "mom90": "Zmiana ceny z 90 dni",
            "volpct": "Jak duże są teraz wahania", "powerlaw": "Cena na tle długoterminowego trendu", "hmm_hi": "Czy rynek jest „nerwowy”",
            "hmm_mu": "Kierunek sugerowany przez model reżimów", "ev_eqdn30": "Spadki akcji w ostatnich 30 dniach", "ev_shock30": "Szoki na rynkach (30 dni)",
            "ev_hack30": "Włamania do giełd (30 dni)", "stress": "Strach na giełdzie", "fomc_soon": "Decyzja Fed w ciągu 3 dni",
            "opex_soon": "Wygasanie kontraktów w ciągu 3 dni", "kimchi": "Premia kimchi (Korea)", "wiki_west": "Zainteresowanie na Zachodzie",
            "wiki_asia": "Zainteresowanie w Azji", "dvol_vrp": "Koszt ubezpieczenia od wahań", "cot": "Pozycje dużych funduszy",
            "hurst": "Skłonność ruchów do kontynuacji", "pent": "Wzorce w kolejności ruchów", "vr5": "Cofanie się cen w 5 dni",
            "acabs": "Serie dużych ruchów", "hill": "Częstość ekstremalnych ruchów", "kalman": "Trend ceny"}


def _level(x):
    a = abs(x)
    if a < 0.4:
        return "normalnie"
    word = "wysoko" if x > 0 else "nisko"
    return ("dość " if a < 1.2 else "bardzo ") + word


def why_html(res, ens, E, df):
    fc = {f["horizon"]: f for f in res["forecasts"]}
    p0 = res["price"]
    sig = ens["sigma"]
    cards = []

    rows = "".join(f'<tr><td>{H} dni</td><td>±{h["sigma_H"]*100:.0f}%</td><td>{(fc[H]["p10"]/p0-1)*100:+.0f}% … {(fc[H]["p90"]/p0-1)*100:+.0f}%</td></tr>'
                   for H, h in ens["horizons"].items())
    dv = df["dvol"].dropna().iloc[-1] if "dvol" in df and df["dvol"].notna().any() else None
    dvt = ""
    if dv is not None:
        impl = dv / math.sqrt(365)
        cmpt = ("Inwestorzy spodziewają się więcej ruchów, niż wynika z ostatnich dni." if impl > sig * 100 * 1.08 else
                "Inwestorzy spodziewają się mniej ruchów, niż wynika z ostatnich dni." if impl < sig * 100 * 0.92 else "To mniej więcej to samo, co wynika z ostatnich dni.")
        dvt = (f'<li>Na giełdzie opcji (to rynek „ubezpieczeń” od zmian ceny) inwestorzy wyceniają dzienne wahania na ok. <b>{impl:.1f}%</b>, a nasz model wylicza <b>{sig*100:.1f}%</b>. {cmpt}</li>')
    cards.append(("Skąd bierze się szerokość zakresu?",
                  f'<ul><li>W ostatnim czasie cena Bitcoina zmienia się dziennie o około <b>{sig*100:.1f}%</b>. Liczymy to trzema niezależnymi sposobami i łączymy je, '
                  f'a wagę każdego sposobu wybieramy według tego, który lepiej sprawdzał się w przeszłości.</li>'
                  f'<li>Im dłuższy okres, tym większa niepewność (ale rośnie wolniej niż czas). Kształt rozkładu, czyli to, jak często zdarzają się duże skoki, bierzemy z prawdziwej historii Bitcoina, '
                  f'z większym naciskiem na ostatnie dwa lata.</li>{dvt}'
                  f'<li>Szansa, że rynek jest teraz w „nerwowym” trybie dużych wahań, wynosi wg naszego modelu <b>{ens["hmm"]["p_hi"]*100:.0f}%</b>.</li>'
                  f'<li>Zakresy są skalibrowane na historii tak, żeby cena mieściła się w nich w ok. 8 na 10 przypadków.</li></ul>'
                  f'<table class="tbl"><tr><th>Za ile</th><th>Typowe wahanie</th><th>Zakres ceny względem dziś (8 na 10)</th></tr>{rows}</table>'))

    lines = ""
    for H, h in ens["horizons"].items():
        cand = [r for r in h["table"] if r["model"] not in ("naive", "ens")]
        b = max(cand, key=lambda r: r["skill"])
        p = b["dm_p"]
        verdict = "Przewaga jest prawdopodobnie prawdziwa." if (p == p and p <= 0.05) else ("Może coś w tym jest, ale to niepewne." if (p == p and p <= 0.20) else "Nie ma dowodu, że to działa.")
        mname = MODEL_LAB.get(b["model"], b["label"])
        better = "lepsza" if b["skill"] > 0 else "gorsza"
        lines += (f'<li><b>Za {H} dni:</b> najlepsza metoda ({mname[0].lower() + mname[1:]}) była o {abs(b["skill"])*100:.1f}% {better} od zgadywania; '
                  f'szansa, że to przypadek: {p*100:.0f}%. {verdict}</li>')
    allzero = all(h["alpha"] == 0 for h in ens["horizons"].values())
    cards.append(("Dlaczego nie mówimy „cena wzrośnie” lub „spadnie”?",
                  f'<ul>{lines}</ul><p class="muted sm">Każdą metodę sprawdzaliśmy na historii tak, jakby używano jej na bieżąco: w każdym dniu znała tylko to, co było wtedy wiadome, '
                  f'i zgadywała przyszłość. Porównujemy ją z prostym założeniem „cena zostaje taka sama”. Metodę dopuszczamy do prognozy dopiero, gdy szansa, że jej przewaga to przypadek, jest mniejsza niż 10%. '
                  f'{"Żadna metoda nie spełniła tego warunku, więc środek prognozy to dzisiejsza cena." if allzero else "Tam, gdzie warunek spełniono, środek prognozy jest przesunięty."} '
                  f'To częsty wynik dla Bitcoina: badacze zgadzają się, że kierunek ceny na miesiące jest bardzo trudny do przewidzenia, a wiarygodnie da się oceniać głównie wahania.</p>'))

    H = 90 if 90 in ens["horizons"] else list(ens["horizons"])[0]
    ex = ens["horizons"][H]["explain"]
    drv = ""
    for name, c, x in ex.get("ridge", [])[:6]:
        eff = c * ens["horizons"][H]["sigma_H"] * 100
        drv += (f'<tr><td>{FEAT_LAB.get(name, name)}</td><td>{_level(x)}</td><td class="{"pos" if eff >= 0 else "neg"}">{eff:+.1f}% ({"podnosi" if eff >= 0 else "obniża"})</td></tr>')
    view = fc[H]["view"]
    sg = sorted([p for p in res["signals"] if p["score"] is not None and p["weight"] > 0], key=lambda p: -(p["score"] * p["weight"]))
    low = lambda t: t[0].lower() + t[1:]
    pos = ", ".join(low(LABELS[p["key"]]) for p in sg[:3] if p["score"] > 0) or "brak wyraźnych"
    neg = ", ".join(low(LABELS[p["key"]]) for p in sg[::-1][:3] if p["score"] < 0) or "brak wyraźnych"
    cards.append((f"Co sugerują wskaźniki (scenariusz na {H} dni, nie prognoza)",
                  f'<p>Gdyby ufać wskaźnikom, środek prognozy wyniósłby <b>{n(view["p50"])} USD</b> ({(view["p50"]/p0-1)*100:+.1f}% względem dziś), a szansa wzrostu {view["prob_up"]*100:.0f}%. Najwięcej wnoszą:</p>'
                  f'<table class="tbl"><tr><th>Czynnik</th><th>Jak jest dziś</th><th>Wpływ na prognozę</th></tr>{drv}</table>'
                  f'<p class="muted sm" style="margin-top:8px">Najmocniej wskazują na wzrost: {pos}. Najmocniej na spadek: {neg}. '
                  f'Ponieważ ta metoda nie przeszła testu na historii, traktuj to jako ciekawy scenariusz, a nie prognozę.</p>'))

    kn = ex.get("knn", [])
    ka = ex.get("knn_all")
    if kn:
        rows = "".join(f'<tr><td>{d}</td><td class="{"pos" if r >= 0 else "neg"}">{r:+.0f}%</td></tr>' for d, r, _ in kn)
        kskill = [r for r in ens["horizons"][H]["table"] if r["model"] == "knn"][0]["skill"] * 100
        cards.append((f"Podobne dni z przeszłości (co było {H} dni później)",
                      f'<p>Szukamy dni z historii, w których sytuacja była podobna do dzisiejszej (ocena rynku, zmiany cen, wahania, miejsce w cyklu). '
                      f'Wśród {ka["n"]} najbardziej podobnych dni <b>{ka["up"]*100:.0f}%</b> zakończyło się wzrostem, a typowa zmiana to <b>{ka["median_ret"]:+.0f}%</b>. '
                      f'<b>Uwaga:</b> te dni pochodzą tylko z {ka.get("episodes", "?")} oddzielnych okresów (kolejne dni jednego rajdu to nie są niezależne przykłady), '
                      f'a ta metoda w teście na historii wypadła o {abs(kskill):.0f}% {"lepiej" if kskill > 0 else "gorzej"} niż zgadywanie. Traktuj to jako ilustrację, nie dowód.</p>'
                      f'<table class="tbl"><tr><th>Najbardziej podobne dni (co najmniej 30 dni od siebie)</th><th>Zmiana ceny po {H} dniach</th></tr>{rows}</table>'))

    ev = E["events"]
    last = df.index[-1]
    rec = ev[(ev["date"] > last - pd.Timedelta(days=30)) & (ev["date"] <= last) & (~ev["cat"].isin(["nfp", "opex", "fomc"]))]
    recl = "".join(f'<li>{r.date:%Y-%m-%d}: {html.escape(r.title)}</li>' for r in rec.itertuples()) or "<li>W ostatnich 30 dniach nie było żadnych dużych szoków ani wydarzeń z naszej listy.</li>"
    up = "".join(f'<li>{u["date"]}: {html.escape(u["title"])}' + (f' <span class="muted">(zwykle po 3 dniach cena zmieniała się o {u["hist"]["mean_ret"]:+.1f}%, na podstawie {u["hist"]["n"]} podobnych wydarzeń)</span>' if u["hist"] else "") + "</li>"
                 for u in E["upcoming"][:4])
    cards.append(("Co dzieje się na świecie",
                  f'<p style="margin-bottom:4px"><b>Ostatnie 30 dni</b></p><ul>{recl}</ul><p style="margin:10px 0 4px"><b>Nadchodzące</b></p><ul>{up}</ul>'
                  f'<p class="muted sm">Takie wydarzenia podajemy też naszym metodom jako dodatkową wskazówkę. Zaczną wpływać na prognozę dopiero, gdy test na historii pokaże, że to pomaga.</p>'))

    cov = " · ".join(f'{H} dni: {[r for r in h["table"] if r["model"] == "ens"][0]["cover80"]*10:.1f}' for H, h in ens["horizons"].items())
    cards.append(("Na ile można temu ufać?",
                  f'<ul><li><b>Szerokość zakresu: można ufać.</b> W testach na historii cena mieściła się w podanym zakresie w tylu przypadkach na 10 ({cov}). Cel to 8.</li>'
                  + ('<li><b>Kierunek ruchu: nie można ufać.</b> Nie ma dowodu, że potrafimy przewidzieć, czy cena pójdzie w górę, czy w dół, więc szansa wzrostu to około 50%.</li>'
                     if allzero else
                     '<li><b>Kierunek ruchu: częściowo.</b> Na części okresów nasze metody okazały się lepsze od zgadywania, ale próba jest mała, więc nie traktuj tego jak pewności.</li>') +
                  '<li><b>Nagłe, rzadkie wydarzenia</b> (upadek giełdy, pandemia) bywają silniejsze od każdej prognozy. Mamy dane z tylko czterech pełnych cykli, a rynek się zmienia (fundusze ETF, firmy kupujące Bitcoina).</li>'
                  '<li>System aktualizuje się sam i sam sprawdza swoje dawne prognozy (sekcja niżej).</li></ul>'))
    return "".join(f'<article class="card why"><h3 class="ct">{t}</h3>{b}</article>' for t, b in cards)


def conclusion_text(ens):
    act = [H for H, h in ens["horizons"].items() if h["alpha"] > 0]
    off = [H for H, h in ens["horizons"].items() if h["alpha"] == 0]
    if not act:
        s = ("Wniosek na dziś: żadna z metod nie jest istotnie lepsza od założenia „cena zostaje taka sama”, na żadnym z okresów. "
             "Dlatego główna prognoza opiera się na tym, jak mocno cena zwykle się waha, a scenariusz ze wskaźników pokazujemy osobno jako ciekawostkę.")
    else:
        s = (f"Wniosek na dziś: na okresie {', '.join(map(str, act))} dni nasze metody okazały się istotnie lepsze od zgadywania i przesuwają środek prognozy "
             f"(ostrożnie: przy długich okresach przykłady z historii mocno się nakładają). "
             + (f"Na okresie {', '.join(map(str, off))} dni prognoza opiera się na typowych wahaniach ceny." if off else ""))
    return s + (" Szansa wzrostu około 50% to założenie neutralne: w historii Bitcoin rósł w ok. 60% okien 30-dniowych, ale uwzględnienie tego nie poprawiło wyników testu, "
                "a dawne zyski nie muszą się powtórzyć. System aktualizuje się sam, więc wagi metod zmieniają się wraz z nowymi danymi.")


def _pctbar(p):
    return f'<div class="pbar"><i style="left:{p*100:.0f}%"></i></div><div class="range-l sm muted"><span>rzadko</span><span>częściej niż w {p*100:.0f}% historii</span><span>często</span></div>'


def regularity_html(Rg):
    nw, pc = Rg["now"], Rg["pct"]
    h, pe, vr, aa, hl = nw["hurst"], nw["pent"], nw["vr5"], nw["acabs"], nw["hill"]
    mk, kal, evt = Rg["markov"], Rg["kalman"], Rg["evt"]
    cyc = ", ".join(f'{p:.0f} dni' for p, s in Rg["cycles"])
    tiles = [
        ("Czy ruchy ceny mają tendencję do kontynuacji?", "wykładnik Hursta", f"{h:.2f}".replace(".", ","), pc["hurst"],
         ("Wartość 0,5 to czysty przypadek. Tu: ruchy mają niewielką skłonność do kontynuacji." if h > 0.53 else "Wartość 0,5 to czysty przypadek. Tu: ruchy częściej się cofają." if h < 0.47 else "Wartość 0,5 to czysty przypadek, i tak właśnie jest.")),
        ("Czy w kolejności ruchów są jakieś wzorce?", "entropia permutacyjna", f"{pe:.3f}".replace(".", ","), pc["pent"],
         "Wartość 1 oznacza brak jakichkolwiek wzorców (czysty przypadek). Tak jest teraz." if pe > 0.98 else "Poniżej 1: pojawiają się powtarzalne wzorce w kolejności ruchów."),
        ("Czy po wzroście częściej następuje spadek?", "iloraz wariancji", f"{vr:+.2f}".replace(".", ","), pc["vr5"],
         "Ujemna wartość: ruchy w ciągu 5 dni lekko się cofają. Bliskie zera: czysty przypadek." if vr < 0 else "Dodatnia wartość: ruchy w ciągu 5 dni lekko się kumulują."),
        ("Czy duże ruchy przychodzą seriami?", "grupowanie zmienności", f"{aa:+.2f}".replace(".", ","), pc["acabs"],
         "Tak, wyraźnie: po dużym ruchu zwykle są kolejne. To jest ta część rynku, którą da się prognozować." if aa > 0.15 else "Słabo: duże ruchy zdarzają się dość niezależnie."),
        ("Jak często zdarzają się ekstremalne ruchy?", "indeks ogona", f"{hl:.1f}".replace(".", ","), pc["hill"],
         "Wartość około 3 oznacza, że ekstremalne ruchy są znacznie częstsze niż w „książkowym” rozkładzie." if hl < 4 else "Ekstremalne ruchy zdarzają się rzadziej niż zwykle w ostatnim roku."),
    ]
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b2}</b><span class="muted sm" style="margin-top:-2px">nazwa techniczna: {tech}</span>{_pctbar(c)}<span class="muted sm">{d}</span></div>'
                 for a, tech, b2, c, d in tiles)
    nx = mk["next"]
    names = ["spadek", "brak większej zmiany", "wzrost"]
    ev = (f'<div class="card tile"><span class="muted sm">Jak zły może być pojedynczy dzień?</span><b>−{evt["var990"]*100:.1f}%</b>'
          f'<span class="muted sm">Taki dzienny spadek zdarza się mniej więcej raz na 100 dni (według zwykłego rozkładu byłoby tylko −{evt["gauss99"]*100:.1f}%). '
          f'Raz na 1000 dni: −{evt["var999"]*100:.1f}%.</span></div>') if evt else ""
    extra = (f'<div class="card tile"><span class="muted sm">Co zwykle dzieje się po ostatnim tygodniu?</span><b>{max(nx)*100:.0f}% szans na: {names[int(np.argmax(nx))]}</b>'
             f'<span class="muted sm">Po tygodniu „{names[mk["now"]]}” kolejny tydzień kończył się: spadkiem w {nx[0]*100:.0f}%, bez większej zmiany w {nx[1]*100:.0f}%, wzrostem w {nx[2]*100:.0f}% przypadków. '
             f'{"Zależność jest słaba, ale możliwa." if mk["p_dep"] < 0.1 else "Praktycznie nie ma tu zależności."}</span></div>'
             f'<div class="card tile"><span class="muted sm">Czy cena ma teraz wyraźny trend?</span><b>{kal["slope_w"]*100:+.2f}% na tydzień</b>'
             f'<span class="muted sm">Błąd tej oceny to ±{kal["sd"]*100:.1f}%, więc {"trend nie różni się od zera." if abs(kal["z"]) < 2 else "trend jest wyraźny."}</span></div>'
             f'<div class="card tile"><span class="muted sm">Czy zmienność ma powtarzalny rytm?</span><b>{Rg["cycles"][0][0]:.0f} dni</b>'
             f'<span class="muted sm">Najsilniejsze rytmy: {cyc}. Rytm 7-dniowy to zwykły efekt dni tygodnia. Dłuższe rytmy są niepewne, bo mamy tylko około 15 lat danych.</span></div>')
    dir_random = 0.47 <= h <= 0.57 and pe > 0.97 and abs(vr) < 0.1
    vol_struct = aa > 0.12
    verdict = (("Kierunek ruchów ceny jest bliski czystemu przypadkowi. " if dir_random else
                "W kierunku ruchów widać pewne wzorce, ale wymagają potwierdzenia testem na historii. ")
               + ("Za to wielkość wahań ma wyraźną strukturę (duże ruchy przychodzą seriami, jest rytm tygodniowy, zdarzają się ekstremalne skoki), dlatego system prognozuje przede wszystkim zakres ceny i ryzyko. "
                  if vol_struct else "Serie dużych ruchów są teraz słabe. ")
               + "Wszystkie miary liczymy z ostatniego roku, więc zmieniają się z czasem.")
    return tl + ev + extra, verdict


def analogs_html(A):
    if not A or not A.get("rows"):
        return ""
    rows = "".join(f'<tr><td>{r["date"]}</td><td class="{"pos" if r["f30"] >= 0 else "neg"}">{r["f30"]:+.0f}%</td><td class="{"pos" if r["f90"] >= 0 else "neg"}">{r["f90"]:+.0f}%</td></tr>' for r in A["rows"])
    return (f'<div class="panel" style="margin-bottom:18px"><h3 class="ct">Kiedy świat wyglądał podobnie do dzisiejszego?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">Opisujemy świat profilem z ostatnich 30 dni: ile wpisów w dzienniku wydarzeń dotyczyło wojen, rynków, banków, sankcji, wyborów, klęsk, pandemii, kryptowalut i innych tematów ({A["n_feat"]} cech). '
            f'Szukamy okresów z najbardziej podobnym profilem i patrzymy, co robił potem Bitcoin. To ciekawostka, a nie prognoza: to tylko kilka oddzielnych epizodów.</p>'
            f'<table class="tbl"><tr><th>Okres (koniec 30-dniowego okna)</th><th>Cena po 30 dniach</th><th>Cena po 90 dniach</th></tr>{rows}</table>'
            f'<p class="muted sm" style="margin:10px 0 0">W {A["up30"]*100:.0f}% tych okresów cena po 30 dniach była wyższa, typowa zmiana: {A["med30"]:+.0f}% po 30 dniach i {A["med90"]:+.0f}% po 90 dniach. '
            f'Bitcoin rósł w ok. 60% okien 30-dniowych w całej historii, więc samo to jeszcze nic nie dowodzi.</p></div>')


def scan_html(S):
    if S is None:
        return '<div class="card empty"><b>Brak danych ze świata</b></div>', ""
    s = S["summary"]
    nt = f"{s['n_tests']:,}".replace(",", " ")
    head = (f'<div class="card"><p style="margin:0 0 6px">Sprawdziliśmy <b>{nt} możliwych powiązań</b> między Bitcoinem a {s["n_series"]} wskaźnikami ze świata: '
            f'giełdami i walutami ponad 20 krajów, ropą, złotem i żywnością, zainteresowaniem polityką i kryzysami (liczba wejść na Wikipedię), pogodą w miejscach, gdzie kopie się Bitcoina, '
            f'plamami na Słońcu, trzęsieniami ziemi, fazami Księżyca i porą roku. Szukamy, czy to, co dzieje się dziś, zapowiada zmianę ceny Bitcoina za tydzień lub miesiąc.</p>'
            f'<p class="muted sm" style="margin:0 0 8px">Gdy sprawdza się tak dużo rzeczy, część wygląda na powiązaną tylko przez przypadek (jak trafienie szóstki w loterii przy tysiącach zakładów). '
            f'Dlatego stosujemy trzy sita: korektę na liczbę testów, sprawdzenie, czy zależność trzyma się w obu połowach historii, i test na najnowszych danych.</p>'
            f'<div class="funnel"><div><b>{s["raw_p05"]}</b><span>wyglądało na powiązane</span></div><div><b>~{s["expected_fp"]:.0f}</b><span>tyle wyszłoby samym przypadkiem</span></div>'
            f'<div><b>{s["n_fdr"]}</b><span>zostało po 1. sicie</span></div><div><b>{s["n_stable"]}</b><span>po 2. sicie</span></div>'
            f'<div><b>{s["n_disc"]}</b><span>po 3. sicie (prawdziwe odkrycia)</span></div></div></div>')

    def rows(df):
        out = ""
        for r in df.itertuples():
            out += (f'<tr><td>{html.escape(r.label)}</td><td>{r.h} dni</td><td>{r.lag} dni</td><td class="{"pos" if r.r >= 0 else "neg"}">{r.r:+.2f}</td>'
                    f'<td>{r.p*100:.1f}%</td><td class="{"pos" if r.r_test >= 0 else "neg"}">{r.r_test:+.2f}</td></tr>')
        return out
    th = ('<tr><th>Wskaźnik</th><th>Zmiana ceny za</th><th>Wskaźnik sprzed</th><th title="od −1 do +1">Siła powiązania</th>'
          '<th title="Szansa, że to przypadek, bez uwzględnienia liczby testów (po korekcie byłaby znacznie wyższa)">Szansa przypadku (bez korekty)</th><th title="Siła powiązania w najnowszych danych">W nowszych danych</th></tr>')
    if len(S["discoveries"]):
        d = f'<table class="tbl">{th}{rows(S["discoveries"])}</table>'
    else:
        d = ('<p class="muted" style="margin:0">Żadne powiązanie nie przeszło wszystkich trzech sit. To normalny wynik: przy tak szerokim przeszukiwaniu '
             'większość „odkryć” to przypadek. System szuka dalej przy każdej aktualizacji i da znać, jeśli coś się utrzyma.</p>')
    c = f'<table class="tbl">{th}{rows(S["candidates"])}</table>'
    co = ""
    for df_ in (S["comove_pos"], S["comove_neg"]):
        for r in df_.itertuples():
            w = min(abs(r.r), 0.6) / 0.6 * 50
            co += (f'<div class="sig sm-row"><div class="sig-l"><b>{html.escape(r.label)}</b></div><div class="dv"><i class="mid"></i>'
                   f'<span class="{"g" if r.r >= 0 else "r"}" style="left:{50 if r.r >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                   f'<div class="sig-v"><b>{r.r:+.2f}</b></div></div>')
    an = analogs_html(S.get("analogs"))
    return head, an + (f'<div class="two2"><div class="card"><h3 class="ct">Odkrycia, które przeszły wszystkie sita</h3>{d}'
                  f'<h3 class="ct" style="margin-top:18px">Najciekawsze podejrzenia (nie przeszły sit, ale obserwujemy)</h3>{c}'
                  f'<p class="muted sm" style="margin-top:8px">Siła powiązania to liczba od −1 do +1 (0 = brak związku). Te podejrzenia utrzymują się w czasie i działają też w nowszych danych, '
                  f'ale po uwzględnieniu liczby testów nie da się wykluczyć przypadku.</p></div>'
                  f'<div class="card"><h3 class="ct">Z czym Bitcoin porusza się razem w tym samym tygodniu</h3>'
                  f'<p class="muted sm" style="margin:0 0 8px">To opis, nie prognoza: plus = rosną razem, minus = w przeciwne strony.</p>{co}</div></div>')


def upcoming_html(E):
    rows = ""
    for u in E["upcoming"][:10]:
        h = u["hist"]
        hist = (f'zwykle po 3 dniach {h["mean_ret"]:+.1f}%, cena rosła w {h["pos"]*100:.0f}% przypadków (na podstawie {h["n"]} podobnych)' if h else "brak historii")
        rows += f'<tr><td>{u["date"]}</td><td>{html.escape(u["title"])}</td><td class="muted">{hist}</td></tr>'
    return f'<table class="tbl"><tr><th>Data</th><th>Wydarzenie</th><th>Jak zwykle reagował Bitcoin</th></tr>{rows}</table>'


def track_html(stats):
    if not stats:
        return ('<div class="card empty"><b>Dziennik prognoz dopiero się zapełnia</b>'
                '<p class="muted">Każdą prognozę zapisujemy, a gdy minie jej termin, porównujemy z prawdziwą ceną. '
                'Pierwsze oceny pojawią się po tygodniu od uruchomienia systemu.</p></div>')
    cov = stats["coverage80"] * 100
    return (f'<div class="tiles"><div class="card tile"><span class="muted sm">Sprawdzonych prognoz</span><b>{stats["n"]}</b></div>'
            f'<div class="card tile"><span class="muted sm">Cena mieściła się w zakresie</span><b>{cov:.0f}%</b><span class="muted sm">cel: 80%</span></div>'
            f'<div class="card tile"><span class="muted sm">Średnio pomyliliśmy się o</span><b>{stats["mape"]:.1f}%</b><span class="muted sm">środek prognozy vs prawdziwa cena</span></div>'
            + (f'<div class="card tile"><span class="muted sm">Trafny kierunek</span><b>{stats["direction"]*100:.0f}%</b><span class="muted sm">50% = tyle samo, co rzut monetą (liczone dla {stats["direction_n"]} prognoz ze wskazanym kierunkiem)</span></div></div>'
               if stats.get("direction") is not None else
               '<div class="card tile"><span class="muted sm">Trafny kierunek</span><b>nie dotyczy</b><span class="muted sm">prognozy były neutralne (środek równy cenie dziś), więc nie wskazywały kierunku</span></div></div>'))


def sources_html(status):
    ok = sum(1 for _, s, _ in status if s == "OK")
    pills = "".join(f'<div class="pill {"ok" if s == "OK" else "bad"}" title="{html.escape(i)}"><i></i>{html.escape(nm)}</div>'
                    for nm, s, i in status)
    return ok, len(status), pills


def corr_html(corr):
    names = {"ndx": "Nasdaq", "spx": "S&P 500", "gold": "Złoto", "dxy": "Dolar (DXY)"}
    rows = ""
    for k, v in corr.items():
        w = abs(v) * 50
        rows += (f'<div class="sig sm-row"><div class="sig-l"><b>{names[k]}</b></div><div class="dv"><i class="mid"></i>'
                 f'<span class="{"g" if v >= 0 else "r"}" style="left:{50 if v >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                 f'<div class="sig-v"><b>{v:+.2f}</b></div></div>')
    return rows


# ----------------------------------------------------------------------------------------------
# Główna funkcja
# ----------------------------------------------------------------------------------------------

def baro_svg(v):
    """Barometr: łuk 240 stopni od 'drogo' (lewo) do 'tanio' (prawo), igła z animacją."""
    cx = cy = 200
    r, a0, sweep = 150, 150, 240
    pt = lambda th, rad: (cx + rad * math.cos(math.radians(th)), cy + rad * math.sin(math.radians(th)))
    x0, y0 = pt(a0, r)
    x1, y1 = pt(a0 + sweep, r)
    segs = ""
    N = 72
    for i in range(N):
        t0, t1 = a0 + sweep * i / N, a0 + sweep * (i + 1) / N + 0.7
        xa, ya = pt(t0, r)
        xb, yb = pt(t1, r)
        k = (i + 0.5) / N
        segs += (f'<path d="M{xa:.1f} {ya:.1f}A{r} {r} 0 0 1 {xb:.1f} {yb:.1f}" style="stroke:color-mix(in srgb,var(--red) {100 - k * 100:.0f}%,var(--green) {k * 100:.0f}%)"/>')
    ticks = ""
    for vv in (-1, -0.5, 0, 0.5, 1):
        th = a0 + (vv + 1) / 2 * sweep
        xa, ya = pt(th, r + 18)
        xb, yb = pt(th, r + 28 if vv in (-1, 0, 1) else r + 24)
        ticks += f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="var(--line2)" stroke-width="2" stroke-linecap="round"/>'
    top = pt(270, r + 46)
    vt = f"{v:+.2f}".replace(".", ",").replace("-", "\u2212")
    return f"""<svg viewBox="0 0 400 360" role="img" aria-label="Ocena rynku {vt}">
<path d="M{x0:.1f} {y0:.1f}A{r} {r} 0 1 1 {x1:.1f} {y1:.1f}" fill="none" stroke="var(--track)" stroke-width="26" stroke-linecap="round"/>
<g fill="none" stroke-width="12" stroke-linecap="butt" opacity=".95">{segs}</g>
{ticks}
<text x="{pt(a0, r)[0] - 6:.0f}" y="{pt(a0, r)[1] + 46:.0f}" text-anchor="middle" style="fill:var(--red);font:600 12px var(--f-mono);letter-spacing:.14em">DROGO</text>
<text x="{pt(a0 + sweep, r)[0] + 6:.0f}" y="{pt(a0 + sweep, r)[1] + 46:.0f}" text-anchor="middle" style="fill:var(--green);font:600 12px var(--f-mono);letter-spacing:.14em">TANIO</text>
<text x="{top[0]:.0f}" y="{top[1] + 4:.0f}" text-anchor="middle" style="fill:var(--faint);font:600 10.5px var(--f-mono);letter-spacing:.14em">NORMALNIE</text>
<g class="needle" style="--rot:{v * 120:.1f}deg;filter:drop-shadow(0 0 8px color-mix(in srgb,var(--accent) 70%,transparent))"><path d="M195 200L205 200L200 {200 - (r - 30)}Z" fill="var(--accent)"/></g>
<circle cx="200" cy="200" r="12" fill="var(--fg)"/><circle cx="200" cy="200" r="4.5" fill="var(--bg)"/>
<text x="200" y="262" text-anchor="middle" style="fill:var(--fg);font:800 46px var(--f-display);letter-spacing:-.03em">{vt}</text>
</svg>"""


def isobars_svg():
    import random
    rnd = random.Random(11)
    paths = ""
    for k in range(11):
        base = 50 + k * 34
        ph = [rnd.uniform(0, 6.28) for _ in range(3)]
        pts = []
        for j in range(181):
            th = j / 180 * 2 * math.pi
            rr = base * (1 + 0.10 * math.sin(2 * th + ph[0]) + 0.06 * math.sin(3 * th + ph[1]) + 0.035 * math.sin(5 * th + ph[2]))
            pts.append((400 + rr * 1.25 * math.cos(th), 300 + rr * 0.85 * math.sin(th)))
        paths += '<path d="M' + "L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + 'Z"/>'
    return f'<svg class="iso" viewBox="0 0 800 600" aria-hidden="true">{paths}</svg>'


def forecast_tiles(res, ens):
    names = {7: "Za tydzień", 30: "Za miesiąc", 90: "Za 3 miesiące", 180: "Za pół roku"}
    p0 = res["price"]
    out = ""
    for f in res["forecasts"]:
        H = f["horizon"]
        lo, hi = math.log(f["p5"]), math.log(f["p95"])
        pos = lambda v: (math.log(v) - lo) / (hi - lo) * 100
        badge = '<span class="badge on">wsparte modelami</span>' if ens["horizons"][H]["alpha"] > 0 else '<span class="badge">typowe wahania</span>'
        d = f["target"]
        mon = ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"][int(d[5:7]) - 1]
        out += (f'<div class="ft"><span class="when">{names.get(H, f"Za {H} dni")} · {int(d[8:10])} {mon}</span>'
                f'<div class="rng">{f["p10"] / 1000:.0f} – {f["p90"] / 1000:.0f}<small> tys. USD</small></div>'
                f'<div class="cone"><div class="a" style="left:{pos(f["p10"]):.1f}%;width:{pos(f["p90"]) - pos(f["p10"]):.1f}%"></div>'
                f'<div class="m" style="left:{pos(f["p50"]):.1f}%"></div><div class="n" style="left:{pos(p0):.1f}%"></div></div>'
                f'<div class="foot"><span>{(f["p10"] / p0 - 1) * 100:+.0f}%</span><span>{(f["p90"] / p0 - 1) * 100:+.0f}%</span></div>'
                f'<div class="why">Szansa wzrostu ok. {f["prob_up"] * 100:.0f}%. {badge}</div></div>')
    return out


def verdict_html(res):
    fc = {f["horizon"]: f for f in res["forecasts"]}
    f = fc.get(30) or list(fc.values())[0]
    return f'{res["regime"]}. Za miesiąc cena najpewniej zmieści się między <b>{n(f["p10"] / 1000)}</b> a <b>{n(f["p90"] / 1000)} tys.</b> USD.'


def signals_html(res):
    fe = res["features"]
    by = {p["key"]: p for p in res["signals"]}
    out = ""
    for i, (title, sub, keys) in enumerate(GROUPS):
        rows, scs, ws = "", [], []
        for k in keys:
            p = by.get(k)
            if p is None or p["weight"] == 0:
                continue
            s = p["score"]
            if s is None:
                scale, val, txt = '<div class="scale na"></div>', "–", "brak danych"
            else:
                scs.append(s * p["weight"])
                ws.append(p["weight"])
                scale = f'<div class="scale"><i class="pin" style="left:{(s + 1) / 2 * 100:.0f}%"></i></div>'
                val = raw_value(k, fe)
                txt = "tanio" if s >= 0.35 else "drogo" if s <= -0.35 else "neutralnie"
            rows += (f'<div class="srow"><div class="s-name"><b>{LABELS[k]}</b><span>{SRC_NOTE[k]}</span></div>{scale}'
                     f'<div class="s-val"><b>{val}</b><span>{txt}</span></div></div>')
        avg = sum(scs) / sum(ws) if ws else None
        chip = "" if avg is None else ('<span class="gchip g">raczej tanio</span>' if avg >= 0.15 else '<span class="gchip r">raczej drogo</span>' if avg <= -0.15 else '<span class="gchip">neutralnie</span>')
        out += (f'<details class="grp"{" open" if i == 0 else ""}><summary><div class="gt"><h3>{title}</h3><p>{sub}</p></div>{chip}</summary>'
                f'<div class="rows">{rows}</div></details>')
    return out


def summary_bullets(res, ens, E):
    p0 = res["price"]
    none = all(h["alpha"] == 0 for h in ens["horizons"].values())
    if none:
        direction = ('<b>Czy cena pójdzie w górę, czy w dół? Tego nie umiemy wiarygodnie przewidzieć.</b> Sprawdziliśmy to na historii: '
                     'żadna z naszych metod nie była lepsza od prostego założenia „cena zostanie taka sama”. Szansa na wzrost to więc około 50 na 50.')
    else:
        act = ", ".join(str(H) for H, h in ens["horizons"].items() if h["alpha"] > 0)
        direction = (f'<b>Czy cena pójdzie w górę, czy w dół?</b> Na okresie {act} dni nasze metody okazały się lepsze od zgadywania i lekko przesuwają prognozę, '
                     'ale to wciąż bardzo niepewne. Na pozostałych okresach nie umiemy tego wiarygodnie przewidzieć.')
    nxt = next((u for u in E["upcoming"] if u["cat"] != "nfp"), None)
    items = [
        f'<li><div><b>Bitcoin kosztuje dziś {n(p0)} USD.</b> {res["regime"]}. <span class="muted">(Ocena rynku {res["composite"]:+.2f} w skali od −1 „drogo” do +1 „tanio”.)</span></div></li>',
        '<li><div><b>Gdzie może być cena?</b> Zakresy poniżej obejmują prawdziwą cenę w około 4 przypadkach na 5. Dla dowolnej liczby dni sprawdzisz to suwakiem w sekcji „Prognoza”.</div></li>',
        f'<li><div>{direction}</div></li>',
        '<li><div><b>Co potrafimy dobrze:</b> oceniać, jak mocno cena może się wahać. W testach na historii nasze zakresy trafiały w około 8 na 10 przypadków.</div></li>',
    ]
    if nxt:
        items.append(f'<li><div><b>Najbliższe ważne wydarzenie:</b> {html.escape(nxt["title"])} ({nxt["date"]}).</div></li>')
    return "".join(items)


def _now_str():
    """Godzina odświeżenia w czasie polskim (gdy brak bazy stref czasowych: UTC z oznaczeniem)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M (czas polski)")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _pl_decimals(page):
    """Zamienia kropkę dziesiętną na przecinek w tekście strony (poza znacznikami, stylami i skryptami)."""
    i = page.index("<script>")
    parts = re.split(r"(<style.*?</style>|<[^>]+>)", page[:i], flags=re.S)
    return "".join(p if p.startswith("<") else re.sub(r"(?<=\d)\.(?=\d)", ",", p) for p in parts) + page[i:]


def plan_data(df):
    """Miesięczne ceny BTC w złotych (kurs USD/PLN z Yahoo) do symulatora wpłat."""
    import world
    try:
        w = pd.read_csv(world.WORLD_CSV, index_col=0, parse_dates=True)["y_pln"]
    except Exception:
        return None
    pln = w.reindex(df.index.union(w.index)).sort_index().ffill(limit=7).reindex(df.index)
    px = (df["close"] * pln).dropna()
    m = px.resample("MS").first().dropna()
    if len(m) < 24:
        return None
    return {"m": [d.strftime("%Y-%m") for d in m.index], "p": [round(float(v), 2) for v in m.values]}


def write_dashboard(res, status, log, stats, ens, E, df, S=None, Rg=None):
    fr = res["frame"]
    reg_tiles, reg_verdict = regularity_html(Rg)
    scan_head, scan_body = scan_html(S)
    f = fr["features"]
    price = f["price"]
    ep = pd.Timestamp("1970-01-01")
    days = [(t - ep).days for t in price.index]
    comp = fr["composite"]
    ma200w = price.rolling(1400).mean()
    chg = lambda k: (price.iloc[-1] / price.iloc[-1 - k] - 1) * 100
    d1, d7, d30, d365 = chg(1), chg(7), chg(30), chg(365)
    ath = float(price.max())
    ath_date = price.idxmax()
    ok, tot, pills = sources_html(status)
    fe = res["features"]

    data = {
        "dates": days, "price": ser_list(price, 0), "real": ser_list(f["realprice"], 0), "ma200w": ser_list(ma200w, 0),
        "comp": ser_list(comp, 3), "halvings": [(pd.Timestamp(h) - ep).days for h in C.HALVINGS],
        "fc": [{"h": x["horizon"], **{k: x[k] for k in ("p5", "p10", "p25", "p50", "p75", "p90", "p95")}} for x in res["forecasts"]],
    }
    pdata = plan_data(df)
    if pdata:
        data["plan"] = pdata
    data["view"] = [x["view"]["p50"] for x in res["forecasts"]]
    data["up"] = [round(x["prob_up"], 4) for x in res["forecasts"]]
    evd = E["events"]
    sel = evd[((evd["kind"] == "curated") | ((evd["cat"] == "hack") & (evd["imp"] >= 2))) & (evd["date"] <= price.index[-1])]
    data["events"] = [{"d": (r.date - ep).days, "c": html.escape(_catlab(r.cat)), "t": html.escape(r.title), "r": (None if pd.isna(r.r3) else round(float(r.r3), 1))}
                      for r in sel.itertuples()]
    reg2_tiles, reg_rows = regions_html(ens, df, res)

    def delta(label, v):
        return f'<div class="dl"><span class="muted sm">{label}</span><b class="{"pos" if v >= 0 else "neg"}">{v:+.1f}%</b></div>'

    page = (TEMPLATE
            .replace("%%PRICE%%", n(res["price"]))
            .replace("%%ASOF%%", res["asof"])
            .replace("%%NOW%%", _now_str())
            .replace("%%NEVENTS%%", n(len(E["events"][E["events"]["date"] <= price.index[-1]]), 0))
            .replace("%%NSIG%%", str(sum(1 for p in res["signals"] if p["weight"] > 0)))
            .replace("%%DELTAS%%", delta("Dziś", d1) + delta("7 dni", d7) + delta("30 dni", d30) + delta("rok", d365))
            .replace("%%ATH%%", f'{(res["price"]/ath-1)*100:.0f}% od rekordu {n(ath)} USD ({ath_date:%Y-%m-%d})')
            .replace("%%GAUGE%%", baro_svg(res["composite"]))
            .replace("%%ISOBARS%%", isobars_svg())
            .replace("%%PRICERAW%%", f"{res['price']:.0f}")
            .replace("%%VERDICT%%", verdict_html(res))
            .replace("%%TILES%%", forecast_tiles(res, ens))
            .replace("%%SUMBULLETS%%", summary_bullets(res, ens, E))
            .replace("%%COMP%%", f'{res["composite"]:+.2f}')
            .replace("%%REGIME%%", res["regime"])
            .replace("%%REAL%%", n(fe.get("realprice")))
            .replace("%%MVRV%%", n(fe.get("mvrv"), 2))
            .replace("%%CYCLE%%", cycle_html(fe.get("halv_days")))
            .replace("%%FCARDS%%", forecast_cards(res))
            .replace("%%SIGNALS%%", signals_html(res))
            .replace("%%GLOSSARY%%", glossary_html())
            .replace("%%CONCLUSION%%", conclusion_text(ens))
            .replace("%%WHY%%", why_html(res, ens, E, df))
            .replace("%%MODELS%%", models_html(ens))
            .replace("%%MOVES%%", moves_html(E))
            .replace("%%STUDY%%", study_html(E))
            .replace("%%UPCOMING%%", upcoming_html(E))
            .replace("%%REGTILES%%", reg2_tiles)
            .replace("%%REGULARITY%%", reg_tiles)
            .replace("%%REGVERDICT%%", reg_verdict)
            .replace("%%SCANHEAD%%", scan_head)
            .replace("%%SCANBODY%%", scan_body)
            .replace("%%REGROWS%%", reg_rows)
            .replace("%%VOLW%%", f"średnia ruchów z ostatnich tygodni {ens['vol']['weights']['ewma']*100:.0f}%, model, w którym duże wahania \"zarażają\" kolejne dni {ens['vol']['weights']['garch']*100:.0f}%, model oparty na dziennej rozpiętości cen {ens['vol']['weights']['har']*100:.0f}%")
            .replace("%%TRACK%%", track_html(stats))
            .replace("%%CORR%%", corr_html(res["corr90"]))
            .replace("%%PILLS%%", pills)
            .replace("%%SRCOK%%", f"{ok}/{tot}")
            .replace("%%SRCCLS%%", "ok" if ok == tot else ("warn" if ok >= tot - 2 else "bad"))
            .replace("%%MIN_IC%%", str(C.MIN_IC))
            .replace("%%DATA%%", json.dumps(data, separators=(",", ":")))
            .replace("%%CHARTJS%%", CHARTJS))
    C.OUT_DIR.mkdir(exist_ok=True); C.DASHBOARD_HTML.parent.mkdir(exist_ok=True)
    C.DASHBOARD_HTML.write_text(_pl_decimals(page), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Szablon (CSS + HTML + JS)
# ----------------------------------------------------------------------------------------------
TEMPLATE = (C.ROOT / "template_head.html").read_text(encoding="utf-8") + (C.ROOT / "template_body.html").read_text(encoding="utf-8")
CHARTJS = (C.ROOT / "chart.js").read_text(encoding="utf-8")
