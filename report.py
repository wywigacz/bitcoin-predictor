"""Dashboard HTML: jeden samodzielny plik (CSS + JS + dane), działa offline."""
import html
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
    ("Czy Bitcoin jest tani czy drogi?", "Porównujemy dzisiejszą cenę z tym, ile średnio zapłacili posiadacze i jaka była cena w ostatnich latach. To najlepiej sprawdzające się wskaźniki.", ["mvrv", "ma200w", "mayer"]),
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


def signals_html(res):
    fe = res["features"]
    by = {p["key"]: p for p in res["signals"]}
    out = ""
    for title, sub, keys in GROUPS:
        rows = ""
        for k in keys:
            p = by.get(k)
            if p is None or p["weight"] == 0:
                continue
            s = p["score"]
            if s is None:
                bar = '<div class="dv"><span class="na">brak danych</span></div>'
                val = "–"
            else:
                w = abs(s) * 50
                bar = (f'<div class="dv"><i class="mid"></i><span class="{"g" if s >= 0 else "r"}" '
                       f'style="left:{50 if s >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>')
                val = raw_value(k, fe)
            verdict = "" if s is None else (" · tanio" if s >= 0.35 else " · drogo" if s <= -0.35 else " · neutralnie")
            rows += (f'<div class="sig"><div class="sig-l"><b>{LABELS[k]}</b><span class="muted sm">{SRC_NOTE[k]}</span></div>'
                     f'{bar}<div class="sig-v"><b>{val}</b><span class="muted sm">{"brak danych" if s is None else f"ocena {s:+.2f}"}{verdict.replace(" · ", " · ") if s is not None else ""}</span></div></div>')
        out += f'<section class="card grp"><h3>{title}</h3><p class="muted sm">{sub}</p>{rows}</section>'
    return out


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
                sk = f'<td class="{"pos" if r["skill"] > 0 else "neg"}">{r["skill"]*100:+.1f}%</td>'
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


def moves_html(E):
    rows = ""
    for m in E["moves"][:16]:
        ch = "".join(f'<span class="cause {x["kind"]}" title="{html.escape(x["title"])} ({x["date"]})">{html.escape(_catlab(x["cat"]))}</span>' for x in m["causes"]) \
            or '<span class="muted sm">nie znaleźliśmy pasującego wydarzenia</span>'
        rows += f'<tr><td>{m["date"]}</td><td class="{"pos" if m["ret"] >= 0 else "neg"}"><b>{m["ret"]:+.1f}%</b></td><td>{ch}</td></tr>'
    c = E["cover"]
    note = (f'Przeanalizowaliśmy {c["n_moves"]} największych skoków ceny w historii. Przy {c["explained_curated"]*100:.0f}% z nich znaleźliśmy obok ważne wydarzenie z naszej listy, '
            f'a przy {c["explained_any"]*100:.0f}% jakiekolwiek wydarzenie (także rutynowe). Ale nawet dla losowo wybranych dni jakieś wydarzenie znajduje się w {c["placebo_any"]*100:.0f}% przypadków. '
            f'Czyli samo „coś się wtedy działo” nie dowodzi, że to było przyczyną.')
    return (f'<table class="tbl"><tr><th>Data</th><th>Zmiana ceny</th><th>Co się działo w tym czasie (±1–2 dni)</th></tr>{rows}</table>'
            f'<p class="muted sm" style="margin-top:10px">{note}</p>')


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
        lines += (f'<li><b>Za {H} dni:</b> najlepsza metoda („{mname.lower()}”) była o {abs(b["skill"])*100:.1f}% {better} od zgadywania; '
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
    pos = ", ".join(LABELS[p["key"]].lower() for p in sg[:3] if p["score"] > 0) or "brak wyraźnych"
    neg = ", ".join(LABELS[p["key"]].lower() for p in sg[::-1][:3] if p["score"] < 0) or "brak wyraźnych"
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


def scan_html(S):
    if S is None:
        return '<div class="card empty"><b>Brak danych ze świata</b></div>', ""
    s = S["summary"]
    head = (f'<div class="card"><p style="margin:0 0 6px">Sprawdziliśmy <b>{s["n_tests"]:,} możliwych powiązań</b> między Bitcoinem a {s["n_series"]} wskaźnikami ze świata: '
            f'giełdami i walutami ponad 20 krajów, ropą, złotem i żywnością, zainteresowaniem polityką i kryzysami (liczba wejść na Wikipedię), pogodą w miejscach, gdzie kopie się Bitcoina, '
            f'plamami na Słońcu, trzęsieniami ziemi, fazami Księżyca i porą roku. Szukamy, czy to, co dzieje się dziś, zapowiada zmianę ceny Bitcoina za tydzień lub miesiąc.</p>'
            f'<p class="muted sm" style="margin:0 0 8px">Gdy sprawdza się tak dużo rzeczy, część wygląda na powiązaną tylko przez przypadek (jak trafienie szóstki w loterii przy tysiącach zakładów). '
            f'Dlatego stosujemy trzy sita: korektę na liczbę testów, sprawdzenie, czy zależność trzyma się w obu połowach historii, i test na najnowszych danych.</p>'
            f'<div class="funnel"><div><b>{s["raw_p05"]}</b><span>wyglądało na powiązane</span></div><div><b>~{s["expected_fp"]:.0f}</b><span>tyle wyszłoby samym przypadkiem</span></div>'
            f'<div><b>{s["n_fdr"]}</b><span>zostało po 1. sicie</span></div><div><b>{s["n_stable"]}</b><span>po 2. sicie</span></div>'
            f'<div><b>{s["n_disc"]}</b><span>po 3. sicie (prawdziwe odkrycia)</span></div></div></div>').replace(",", " ")

    def rows(df):
        out = ""
        for r in df.itertuples():
            out += (f'<tr><td>{html.escape(r.label)}</td><td>{r.h} dni</td><td>{r.lag} dni</td><td class="{"pos" if r.r >= 0 else "neg"}">{r.r:+.2f}</td>'
                    f'<td>{r.p*100:.1f}%</td><td class="{"pos" if r.r_test >= 0 else "neg"}">{r.r_test:+.2f}</td></tr>')
        return out
    th = ('<tr><th>Wskaźnik</th><th>Zmiana ceny za</th><th>Wskaźnik sprzed</th><th title="od −1 do +1">Siła powiązania</th>'
          '<th title="Szansa, że to przypadek (po uwzględnieniu liczby testów byłaby wyższa)">Szansa przypadku</th><th title="Siła powiązania w najnowszych danych">W nowszych danych</th></tr>')
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
    return head, (f'<div class="two2"><div class="card"><h3 class="ct">Odkrycia, które przeszły wszystkie sita</h3>{d}'
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
            f'<div class="card tile"><span class="muted sm">Trafny kierunek</span><b>{stats["direction"]*100:.0f}%</b><span class="muted sm">50% = tyle samo, co rzut monetą</span></div></div>')


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
    evd = E["events"]
    sel = evd[((evd["kind"] == "curated") | ((evd["cat"] == "hack") & (evd["imp"] >= 2))) & (evd["date"] <= price.index[-1])]
    data["events"] = [{"d": (r.date - ep).days, "c": _catlab(r.cat), "t": r.title, "r": (None if pd.isna(r.r3) else round(float(r.r3), 1))}
                      for r in sel.itertuples()]
    reg2_tiles, reg_rows = regions_html(ens, df, res)

    def delta(label, v):
        return f'<div class="dl"><span class="muted sm">{label}</span><b class="{"pos" if v >= 0 else "neg"}">{v:+.1f}%</b></div>'

    page = (TEMPLATE
            .replace("%%PRICE%%", n(res["price"]))
            .replace("%%ASOF%%", res["asof"])
            .replace("%%NOW%%", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("%%DELTAS%%", delta("24 h", d1) + delta("7 dni", d7) + delta("30 dni", d30) + delta("rok", d365))
            .replace("%%ATH%%", f'{(res["price"]/ath-1)*100:.0f}% od rekordu {n(ath)} USD ({ath_date:%Y-%m-%d})')
            .replace("%%GAUGE%%", gauge_svg(res["composite"]))
            .replace("%%COMP%%", f'{res["composite"]:+.2f}')
            .replace("%%REGIME%%", res["regime"])
            .replace("%%REAL%%", n(fe.get("realprice")))
            .replace("%%MVRV%%", n(fe.get("mvrv"), 2))
            .replace("%%CYCLE%%", cycle_html(fe.get("halv_days")))
            .replace("%%FCARDS%%", forecast_cards(res))
            .replace("%%SIGNALS%%", signals_html(res))
            .replace("%%SUMMARY%%", summary_html(res, ens, E))
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
            .replace("%%VOLW%%", ", ".join(f"{k} {v*100:.0f}%" for k, v in ens["vol"]["weights"].items()))
            .replace("%%TRACK%%", track_html(stats))
            .replace("%%CORR%%", corr_html(res["corr90"]))
            .replace("%%PILLS%%", pills)
            .replace("%%SRCOK%%", f"{ok}/{tot}")
            .replace("%%SRCCLS%%", "ok" if ok == tot else ("warn" if ok >= tot - 2 else "bad"))
            .replace("%%MIN_IC%%", str(C.MIN_IC))
            .replace("%%DATA%%", json.dumps(data, separators=(",", ":"))))
    C.OUT_DIR.mkdir(exist_ok=True); C.DASHBOARD_HTML.parent.mkdir(exist_ok=True)
    C.DASHBOARD_HTML.write_text(page, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Szablon (CSS + HTML + JS)
# ----------------------------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Predictor</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='16' fill='%23f7931a'/%3E%3Cpath d='M12 8h6a3.5 3.5 0 0 1 1 6.7A3.7 3.7 0 0 1 18.5 24H12zM15 11v3.5h2.7a1.7 1.7 0 0 0 0-3.5zm0 6v4h3.3a2 2 0 0 0 0-4z' fill='%23fff'/%3E%3C/svg%3E">
<style>
:root{--bg:#070a10;--bg2:#0d121b;--card:#111826;--card2:#161f30;--line:rgba(255,255,255,.07);--line2:rgba(255,255,255,.12);
--fg:#e9edf5;--muted:#8e98ad;--track:#1d2638;--accent:#f7931a;--green:#34d399;--red:#fb7185;--amber:#fbbf24;--blue:#60a5fa;--violet:#a78bfa;
--glow:radial-gradient(900px 420px at 15% -10%,rgba(247,147,26,.13),transparent 60%),radial-gradient(800px 380px at 95% 0%,rgba(96,165,250,.10),transparent 55%);
--shadow:0 1px 0 rgba(255,255,255,.04) inset,0 18px 40px -22px rgba(0,0,0,.7)}
@media (prefers-color-scheme:light){:root{--bg:#f4f6fa;--bg2:#fff;--card:#fff;--card2:#f6f8fc;--line:rgba(15,23,42,.08);--line2:rgba(15,23,42,.14);
--fg:#0f172a;--muted:#64708a;--track:#e6ebf4;--accent:#e8820c;--green:#059669;--red:#e11d48;--amber:#d97706;--blue:#2563eb;--violet:#7c3aed;
--glow:radial-gradient(900px 420px at 15% -10%,rgba(247,147,26,.16),transparent 60%),radial-gradient(800px 380px at 95% 0%,rgba(37,99,235,.10),transparent 55%);
--shadow:0 1px 2px rgba(15,23,42,.05),0 14px 30px -20px rgba(15,23,42,.25)}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);background-image:var(--glow);background-repeat:no-repeat;color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1,h2,h3{margin:0;font-weight:650;letter-spacing:-.01em}h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:600;margin:44px 0 14px;display:flex;align-items:center;gap:10px}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
.muted{color:var(--muted)}.sm{font-size:12.5px}.pos{color:var(--green)}.neg{color:var(--red)}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:18px;padding:18px 20px;box-shadow:var(--shadow)}
/* header */
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
.brand{display:flex;align-items:center;gap:12px}.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#ffb347,var(--accent));display:grid;place-items:center;box-shadow:0 8px 24px -8px var(--accent)}
.logo svg{width:22px;height:22px}.brand h1{font-size:19px}.brand span{display:block;font-size:12.5px;color:var(--muted);margin-top:-2px}
.status{display:flex;gap:8px;align-items:center;font-size:12.5px;color:var(--muted)}
.chip{display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);font-size:12.5px}
.chip i{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 3px color-mix(in srgb,var(--green) 25%,transparent)}
.chip.warn i{background:var(--amber);box-shadow:0 0 0 3px color-mix(in srgb,var(--amber) 25%,transparent)}.chip.bad i{background:var(--red)}
/* hero */
.hero{display:grid;grid-template-columns:1.25fr 1fr 1.1fr;gap:16px}
.price-card .lbl{font-size:12.5px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.big{font-size:clamp(38px,6vw,58px);font-weight:700;letter-spacing:-.035em;line-height:1.05;margin:6px 0 4px}
.big small{font-size:.38em;color:var(--muted);font-weight:500;margin-left:6px;letter-spacing:0}
.deltas{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}
.dl b{display:block;font-size:16px}
.gauge-card{text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge{width:100%;max-width:290px}.gt{stroke:var(--line2);stroke-width:1.5}.gl{fill:var(--muted);font-size:10.5px}
.needle{transform-origin:150px 150px;transform:rotate(var(--rot));animation:sweep 1.4s cubic-bezier(.2,.9,.2,1) both}
@keyframes sweep{from{transform:rotate(-90deg)}to{transform:rotate(var(--rot))}}
.gv{font-size:30px;font-weight:700;margin-top:-6px;letter-spacing:-.02em}.gr{font-size:13px;color:var(--muted);max-width:260px;margin-top:2px}
.kv{display:grid;gap:14px}.kv .row{display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding-bottom:12px;border-bottom:1px solid var(--line)}
.kv .row:last-child{border:0;padding:0}.kv b{font-size:21px;letter-spacing:-.01em}
/* cycle */
.cycle{margin-top:16px}.zones{position:relative;display:flex;gap:3px;height:34px}
.zone{background:color-mix(in srgb,var(--c) 22%,var(--track));border-radius:6px;display:grid;place-items:center;font-size:11px;color:var(--fg);overflow:hidden;white-space:nowrap;border-top:2px solid var(--c)}
.zone span{opacity:.85;padding:0 4px}
.marker{position:absolute;top:-9px;bottom:-9px;width:0}.marker i{position:absolute;left:-1.5px;top:0;bottom:0;width:3px;border-radius:2px;background:var(--fg);box-shadow:0 0 0 3px color-mix(in srgb,var(--fg) 20%,transparent)}
.cyc-foot{margin-top:12px;font-size:13px;color:var(--muted)}.cyc-foot b{color:var(--fg)}
/* chart */
.chart-card{padding:18px 14px 12px}.chart-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;padding:0 8px 6px}
.seg{display:inline-flex;background:var(--bg2);border:1px solid var(--line);border-radius:12px;padding:3px}
.seg button{all:unset;cursor:pointer;padding:6px 14px;border-radius:9px;font-size:12.5px;color:var(--muted);transition:.15s}
.seg button:hover{color:var(--fg)}.seg button.on{background:var(--card2);color:var(--fg);box-shadow:0 0 0 1px var(--line2)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--muted)}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.chart{position:relative;width:100%;touch-action:pan-y}.chart svg{display:block;width:100%;height:auto;overflow:visible}
.cx{position:absolute;top:0;bottom:0;width:1px;background:var(--line2);pointer-events:none;display:none}
.tip{position:absolute;pointer-events:none;display:none;z-index:5;min-width:170px;background:color-mix(in srgb,var(--bg2) 94%,transparent);backdrop-filter:blur(8px);border:1px solid var(--line2);border-radius:12px;padding:10px 12px;font-size:12.5px;box-shadow:0 16px 36px -14px rgba(0,0,0,.6)}
.tip b{display:block;font-size:13px;margin-bottom:4px}.tip div{display:flex;justify-content:space-between;gap:16px;color:var(--muted)}.tip div span:last-child{color:var(--fg)}
.ax{fill:var(--muted);font-size:11px}.gridl{stroke:var(--line)}.hv{stroke:var(--line2);stroke-dasharray:3 4}.hvl{fill:var(--muted);font-size:10.5px}
/* forecasts */
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.fc header,.q header{display:flex;justify-content:space-between;align-items:center}.fc h3,.q h3{font-size:16px}
.badge{font-size:11px;padding:3px 9px;border-radius:999px;background:var(--track);color:var(--muted);white-space:nowrap}.badge.on{background:color-mix(in srgb,var(--green) 18%,transparent);color:var(--green)}
.fc-med{font-size:28px;font-weight:700;letter-spacing:-.025em;margin-top:10px;line-height:1.1}.fc-med small{font-size:.45em;color:var(--muted);font-weight:500}
.fc-chg{font-size:13.5px;font-weight:600;margin-bottom:16px}
.range{position:relative;height:10px;background:var(--track);border-radius:6px;margin:8px 0 6px}
.range .r90{position:absolute;top:0;bottom:0;background:color-mix(in srgb,var(--blue) 38%,transparent);border-radius:6px}
.range .r50{position:absolute;top:0;bottom:0;background:var(--blue);border-radius:6px}
.range .tick{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1.5px;background:var(--accent);border-radius:2px}
.range .now{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:var(--fg);border:2px solid var(--bg2)}
.range-l{display:flex;justify-content:space-between;font-size:12px}.range-l span:not(.muted){font-weight:600}
.pup{display:flex;align-items:center;gap:12px;margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}
.ring{--p:50;width:54px;height:54px;border-radius:50%;background:conic-gradient(var(--green) calc(var(--p)*1%),var(--track) 0);display:grid;place-items:center;position:relative;flex:none}
.ring::before{content:"";position:absolute;inset:6px;border-radius:50%;background:var(--card)}.ring b{position:relative;font-size:13.5px}
.note{margin-top:14px;font-size:13px;color:var(--muted);max-width:80ch}.legend2{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:var(--muted);align-items:center}
.legend2 i{display:inline-block;vertical-align:middle;margin-right:6px}
/* signals */
.sgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.grp h3{font-size:16px}.grp>p{margin:2px 0 8px}
.sig{display:grid;grid-template-columns:1.15fr 1.2fr .95fr;gap:14px;align-items:center;padding:11px 0;border-top:1px solid var(--line)}
.sig-l b{display:block;font-weight:600;font-size:14px}.sig-v{text-align:right}.sig-v b{display:block;font-size:14px}
.dv{position:relative;height:8px;background:var(--track);border-radius:5px}.dv .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--line2)}
.dv span{position:absolute;top:0;bottom:0;border-radius:5px}.dv span.g{background:linear-gradient(90deg,color-mix(in srgb,var(--green) 55%,transparent),var(--green))}.dv span.r{background:linear-gradient(270deg,color-mix(in srgb,var(--red) 55%,transparent),var(--red))}
.dv .na{position:static;font-size:12px;color:var(--muted);display:block;text-align:center;margin-top:-4px}.dv:has(.na){background:transparent}
/* quality */
.q dl{margin:12px 0 0;display:grid;gap:8px}.q dl div{display:flex;justify-content:space-between;gap:10px;font-size:13px}.q dt{color:var(--muted)}.q dd{margin:0;font-weight:650}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.tile{display:flex;flex-direction:column;gap:2px}.tile b{font-size:30px;letter-spacing:-.02em}
.empty b{font-size:16px}.empty p{margin:6px 0 0}
.two{display:grid;grid-template-columns:1.6fr 1fr;gap:14px;margin-top:14px}
.pills{display:flex;flex-wrap:wrap;gap:8px}.pill{display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);font-size:12.5px}
.pill i{width:7px;height:7px;border-radius:50%;background:var(--green)}.pill.bad i{background:var(--red)}.pill.bad{border-color:color-mix(in srgb,var(--red) 40%,transparent)}
.disc{margin-top:40px;padding:16px 20px;border-radius:16px;border:1px solid color-mix(in srgb,var(--amber) 35%,transparent);background:color-mix(in srgb,var(--amber) 8%,transparent);font-size:13px;color:var(--muted)}
.disc b{color:var(--fg)}footer{margin-top:26px;text-align:center;font-size:12px;color:var(--muted)}
.tbl{border-collapse:collapse;width:100%;font-size:13.5px}.tbl th{color:var(--muted);font-weight:600;text-align:left;font-size:12px;padding:6px 8px;border-bottom:1px solid var(--line)}
.tbl td{padding:8px;border-bottom:1px solid var(--line);vertical-align:middle}.tbl tr:last-child td{border-bottom:0}.tbl .dv{min-width:70px}
.tbl tr.sigrow td:first-child,.tbl tr.sigrow td:last-child{font-weight:700}.tbl tr.ens td{background:color-mix(in srgb,var(--accent) 8%,transparent);font-weight:600}
.two2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.ct{font-size:15px;margin-bottom:10px}.two2>*,.grid2>*,.grid4>*,.tiles3>*{min-width:0}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.q header{flex-wrap:wrap;gap:8px}.q .badge{white-space:normal}
.card:has(.tbl){overflow-x:auto}
.tiles3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.tiles3 .tile b{font-size:26px}
.cause{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:999px;margin:2px 4px 2px 0;background:var(--track);color:var(--fg);cursor:default}
.cause.curated{background:color-mix(in srgb,var(--accent) 22%,var(--track))}.cause.calendar{opacity:.8}.cause.auto{background:color-mix(in srgb,var(--blue) 18%,var(--track))}
.view{margin-top:12px;padding:10px 12px;border-radius:12px;background:var(--bg2);border:1px dashed var(--line2);display:flex;flex-direction:column;gap:2px}.view b{font-size:17px}
.why ul{margin:0 0 10px;padding-left:18px;font-size:13.5px}.why li{margin:5px 0}.why p{margin:6px 0;font-size:13.5px}.why .tbl{margin-top:6px}
.pbar{position:relative;height:6px;background:linear-gradient(90deg,var(--track),color-mix(in srgb,var(--blue) 40%,var(--track)));border-radius:4px;margin:8px 0 2px}
.pbar i{position:absolute;top:-4px;width:3px;height:14px;margin-left:-1.5px;border-radius:2px;background:var(--accent)}
.funnel{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:10px}.funnel div{background:var(--bg2);border:1px solid var(--line);border-radius:12px;padding:10px 12px;display:flex;flex-direction:column}
.funnel b{font-size:24px;letter-spacing:-.02em}.funnel span{font-size:12px;color:var(--muted)}
.summary h3{font-size:19px;margin-bottom:8px}.summary ul{margin:0;padding-left:20px;font-size:15.5px}.summary li{margin:9px 0}.summary ul.inner{margin:4px 0 0;font-size:15px}
.gloss dt{font-weight:650;margin-top:12px}.gloss dd{margin:2px 0 0;color:var(--muted);font-size:14px}
.evt{display:inline-flex;align-items:center;gap:6px;cursor:pointer}.evt input{accent-color:var(--accent)}
.reveal{opacity:0;transform:translateY(10px);animation:rise .7s cubic-bezier(.2,.8,.2,1) forwards}
@keyframes rise{to{opacity:1;transform:none}}
@media (max-width:980px){.funnel{grid-template-columns:repeat(2,1fr)}.two2,.tiles3,.grid2{grid-template-columns:1fr}.hero{grid-template-columns:1fr 1fr}.price-card{grid-column:1/-1}.grid4{grid-template-columns:repeat(2,1fr)}.sgrid,.two{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
@media (max-width:620px){.hero,.grid4{grid-template-columns:1fr}.deltas{grid-template-columns:repeat(2,1fr)}.sig{grid-template-columns:1fr 1fr;gap:8px 14px}.sig .dv{grid-column:1/-1;order:3}.wrap{padding:18px 14px 40px}}
@media (prefers-reduced-motion:reduce){.reveal,.needle{animation:none;opacity:1;transform:none}.needle{transform:rotate(var(--rot))}}
</style></head><body><div class="wrap">

<div class="top reveal">
 <div class="brand"><div class="logo"><svg viewBox="0 0 24 24"><path d="M8 4h6.5a3.6 3.6 0 0 1 1.2 7 3.8 3.8 0 0 1-1.5 7.3H8zM11 7v3.6h3a1.8 1.8 0 0 0 0-3.6zm0 6.4V17h3.6a1.8 1.8 0 0 0 0-3.6z" fill="#fff"/></svg></div>
 <div><h1>BTC Predictor</h1><span>Ile może kosztować Bitcoin za tydzień, miesiąc i pół roku, i dlaczego tak sądzimy</span></div></div>
 <div class="status"><span>dane z %%ASOF%% · wygenerowano %%NOW%%</span><span class="chip %%SRCCLS%%"><i></i>źródła %%SRCOK%%</span></div>
</div>

<section class="hero">
 <div class="card price-card reveal" style="animation-delay:.05s"><div class="lbl">Bitcoin · USD</div>
  <div class="big">%%PRICE%%<small>USD</small></div><div class="muted sm">%%ATH%%</div>
  <div class="deltas">%%DELTAS%%</div></div>
 <div class="card gauge-card reveal" style="animation-delay:.12s"><div class="lbl muted sm" style="letter-spacing:.08em;text-transform:uppercase">Ocena rynku: tani czy drogi?</div>
  %%GAUGE%%<div class="gv">%%COMP%%</div><div class="gr">%%REGIME%%</div></div>
 <div class="card kv reveal" style="animation-delay:.19s">
  <div class="row"><span class="muted" title="Średnia cena, po jakiej ostatnio przesunięto wszystkie monety">Średnia cena zakupu monet</span><b>%%REAL%% <small class="muted">USD</small></b></div>
  <div class="row"><span class="muted" title="Cena dziś podzielona przez średnią cenę zakupu. Powyżej 1 posiadacze są średnio na plusie">Zysk posiadaczy (MVRV)</span><b>%%MVRV%%</b></div>
  <div style="padding-top:2px"><span class="muted sm">Gdzie jesteśmy w cyklu 4-letnim (od halvingu do halvingu)</span>%%CYCLE%%</div></div>
</section>

%%SUMMARY%%

<h2>Cena i możliwe scenariusze</h2>
<section class="card chart-card reveal" style="animation-delay:.1s">
 <div class="chart-head"><div class="legend"><span><i style="background:var(--fg)"></i>Cena</span><span><i style="background:var(--accent)"></i>Środek prognozy</span>
  <span><i style="background:var(--amber)"></i>Średnia cena zakupu monet</span><span><i style="background:var(--violet)"></i>Średnia z 4 lat</span>
  <label class="evt"><input type="checkbox" id="evtoggle" checked> wydarzenia</label></div>
  <div class="seg" id="rng"><button data-m="3">3M</button><button data-m="12" class="on">1R</button><button data-m="36">3L</button><button data-m="0">Wszystko</button></div></div>
 <div class="chart" id="chart"><div class="cx" id="cx"></div><div class="tip" id="tip"></div></div>
 <div class="legend2"><span><i style="width:22px;height:8px;border-radius:4px;background:color-mix(in srgb,var(--blue) 35%,transparent)"></i>bardzo prawdopodobny zakres (9 na 10)</span>
  <span><i style="width:22px;height:8px;border-radius:4px;background:color-mix(in srgb,var(--blue) 65%,transparent)"></i>najbardziej prawdopodobny zakres (1 na 2)</span><span>Pionowe linie: halvingi. Kropki: ważne wydarzenia (zielone: cena potem rosła, czerwone: spadała), najedź kursorem</span></div>
</section>

<h2>Gdzie może być cena</h2>
<section class="grid4">%%FCARDS%%</section>
<p class="note">Niebieski pasek to typowy zakres: cena powinna się w nim zmieścić w około 4 przypadkach na 5. Pomarańczowa kreska to środek prognozy, biała kropka to dzisiejsza cena. Zakres jest szeroki, bo Bitcoin mocno się waha, i to wcale nie jest błąd prognozy.</p>

<h2>Dlaczego taka prognoza? Wyjaśnienie krok po kroku</h2>
<section class="grid2">%%WHY%%</section>

<h2>Z czego składa się ocena rynku</h2>
<section class="sgrid">%%SIGNALS%%</section>
<p class="note">Każdy wskaźnik dostaje ocenę od −1 (historycznie drogo) do +1 (historycznie tanio). Ocena rynku to średnia z nich, a te, które lepiej sprawdzały się w przeszłości, liczą się bardziej (najbardziej: zysk posiadaczy i średnia z 4 lat).</p>

<h2>Ocena rynku w czasie (zielone: tanio, czerwone: drogo)</h2>
<section class="card chart-card reveal"><div class="chart" id="chart2"><div class="tip" id="tip2"></div></div></section>

<h2>Mapa wydarzeń</h2>
<p class="note" style="margin-top:0">Zebraliśmy ponad 800 wydarzeń: ważne decyzje i wiadomości (np. zakazy w Chinach, upadek giełdy FTX, zatwierdzenie funduszy ETF), terminy (posiedzenia amerykańskiego banku centralnego, wybory), włamania do giełd oraz nagłe załamania na giełdach i rynkach świata wykrywane automatycznie. Każde jest naniesione na oś czasu ceny (kropki na wykresie wyżej), a my sprawdzamy, jak zwykle reagował Bitcoin.</p>
<section class="two2"><div class="card"><h3 class="ct">Największe skoki ceny i co się wtedy działo</h3>%%MOVES%%</div>
<div class="card"><h3 class="ct">Jak zwykle reaguje Bitcoin na różne rodzaje wydarzeń</h3>%%STUDY%%</div></section>
<h2>Nadchodzące wydarzenia</h2>
<section class="card">%%UPCOMING%%</section>

<h2>Nastroje na świecie i ruchy dużych graczy</h2>
<section class="tiles3">%%REGTILES%%</section>
<section class="card" style="margin-top:14px"><h3 class="ct">Zainteresowanie Bitcoinem w różnych krajach (ostatnie 30 dni, w porównaniu z zeszłym rokiem)</h3><p class="muted sm" style="margin:2px 0 8px">Liczymy, ile osób czyta hasło „Bitcoin” na Wikipedii w danym języku. Gdy zainteresowanie rośnie, zwykle rośnie też cena (i odwrotnie).</p>%%REGROWS%%</section>

<h2>Czy rynek ma jakieś wzorce, czy to czysty przypadek?</h2>
<p class="note" style="margin-top:0">Sprawdzamy kilkoma sposobami, czy ruchy ceny Bitcoina mają jakąś powtarzalną strukturę, czy przypominają rzut monetą. Liczymy to z ostatniego roku. Pasek pod liczbą pokazuje, jak dzisiejsza wartość wypada na tle całej historii.</p>
<section class="tiles3">%%REGULARITY%%</section>
<p class="note"><b>Wniosek:</b> %%REGVERDICT%%</p>

<h2>Szukanie powiązań na całym świecie</h2>
%%SCANHEAD%%
%%SCANBODY%%

<h2>Które metody prognozowania naprawdę działają?</h2>
<p class="note" style="margin-top:0">Wypróbowaliśmy kilka różnych metod prognozowania. Każdą sprawdziliśmy na historii tak, jakby używano jej na bieżąco, i porównaliśmy z najprostszym założeniem: „cena zostanie taka sama”. Metoda trafia do prognozy tylko wtedy, gdy wyraźnie je pokona. Zmienność ceny liczymy z trzech uzupełniających się sposobów (%%VOLW%%).</p>
<section class="grid2">%%MODELS%%</section>
<p class="note">%%CONCLUSION%%</p>

<h2>Czy nasze dotychczasowe prognozy się sprawdziły?</h2>
%%TRACK%%

<div class="two"><div><h2 style="margin-top:30px">Z czym Bitcoin ostatnio się porusza (90 dni)</h2><div class="card">%%CORR%%</div></div>
<div><h2 style="margin-top:30px">Źródła danych</h2><div class="pills">%%PILLS%%</div></div></div>

<h2>Słowniczek: trudne słowa po ludzku</h2><section class="card"><dl class="gloss">%%GLOSSARY%%</dl></section>
<div class="disc"><b>To nie jest porada inwestycyjna.</b> Nikt nie potrafi wiarygodnie przewidzieć ceny Bitcoina. Pokazujemy możliwe scenariusze i uczciwie mówimy, na ile są wiarygodne. Historia obejmuje tylko cztery pełne cykle, a rynek się zmienia, więc nie opieraj na tym ważnych decyzji finansowych.</div>
<footer>BTC Predictor · dane: Bitstamp, CoinMetrics, Blockchain.com, Yahoo Finance, DefiLlama, OKX, CoinGecko, alternative.me</footer>
</div>

<script>
window.__BTC__=%%DATA%%;
(function(){
const D=window.__BTC__, N=D.dates.length, LAST=D.dates[N-1], MAXH=Math.max(...D.fc.map(f=>f.h));
const css=getComputedStyle(document.documentElement), C=n=>css.getPropertyValue(n).trim();
const fmtD=d=>new Date(d*864e5).toISOString().slice(0,10);
const fmtP=v=>v==null?'–':Math.round(v).toLocaleString('pl-PL').replace(/\s/g,' ');
const el=document.getElementById('chart'), cx=document.getElementById('cx'), tip=document.getElementById('tip');
let months=12,showEv=true;
const evBox=document.getElementById('evtoggle');if(evBox)evBox.onchange=()=>{showEv=evBox.checked;draw();};
const keys=['p5','p10','p25','p50','p75','p90','p95'];
const fcPts=k=>[[LAST,D.price[N-1]]].concat(D.fc.map(f=>[LAST+f.h,f[k]]));
function interp(k,x){const p=fcPts(k);for(let i=1;i<p.length;i++){if(x<=p[i][0]){const t=(x-p[i-1][0])/(p[i][0]-p[i-1][0]);return Math.exp(Math.log(p[i-1][1])+t*(Math.log(p[i][1])-Math.log(p[i-1][1])));}}return p[p.length-1][1];}
function niceLog(lo,hi){const out=[];const a=Math.floor(Math.log10(lo)),b=Math.ceil(Math.log10(hi));
 for(let e=a;e<=b;e++)for(const m of [1,2,5]){const v=m*Math.pow(10,e);if(v>=lo&&v<=hi)out.push(v);}return out;}
let geo={};
function draw(){
 const W=el.clientWidth||900,H=Math.round(Math.max(300,Math.min(460,W*.42))),pl=58,pr=16,pt=12,pb=26;
 const x0=months?LAST-months*30.44:D.dates[0], x1=LAST+MAXH+Math.max(6,months?months*1.2:60);
 let i0=0;while(i0<N&&D.dates[i0]<x0)i0++;
 let lo=Infinity,hi=-Infinity;
 const upd=v=>{if(v!=null&&v>0){if(v<lo)lo=v;if(v>hi)hi=v;}};
 for(let i=i0;i<N;i++){upd(D.price[i]);upd(D.real[i]);upd(D.ma200w[i]);}
 D.fc.forEach(f=>{upd(f.p5);upd(f.p95);});
 lo*=.93;hi*=1.07;const L0=Math.log(lo),L1=Math.log(hi);
 const X=x=>pl+(x-x0)/(x1-x0)*(W-pl-pr),Y=v=>pt+(1-(Math.log(v)-L0)/(L1-L0))*(H-pt-pb);
 geo={x0,x1,pl,pr,W,H,X,Y,i0};
 let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><defs>
 <linearGradient id="af" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${C('--accent')}" stop-opacity=".30"/><stop offset="1" stop-color="${C('--accent')}" stop-opacity="0"/></linearGradient>
 <linearGradient id="bf" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C('--blue')}" stop-opacity=".10"/><stop offset="1" stop-color="${C('--blue')}" stop-opacity=".34"/></linearGradient>
 <linearGradient id="bf2" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C('--blue')}" stop-opacity=".22"/><stop offset="1" stop-color="${C('--blue')}" stop-opacity=".58"/></linearGradient>
 <clipPath id="cp"><rect x="${pl}" y="${pt}" width="${W-pl-pr}" height="${H-pt-pb}"/></clipPath></defs>`;
 niceLog(lo,hi).forEach(v=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${Y(v)}" y2="${Y(v)}" class="gridl"/><text x="${pl-8}" y="${Y(v)+4}" text-anchor="end" class="ax">${v>=1000?(v/1000)+'k':v}</text>`;});
 const span=(x1-x0)/30.44, step=span>60?24:span>30?12:span>14?6:span>6?3:1;
 const d0=new Date(x0*864e5);let mo=new Date(Date.UTC(d0.getUTCFullYear(),d0.getUTCMonth()+1,1));
 while(mo.getTime()/864e5<x1){const m=mo.getUTCMonth();const x=mo.getTime()/864e5;
  if((mo.getUTCFullYear()*12+m)%step===0){s+=`<text x="${X(x)}" y="${H-7}" text-anchor="middle" class="ax">${step>=12?mo.getUTCFullYear():mo.toISOString().slice(0,7)}</text>`;}
  mo=new Date(Date.UTC(mo.getUTCFullYear(),m+1,1));}
 s+='<g clip-path="url(#cp)">';
 D.halvings.forEach(h=>{if(h>=x0&&h<=x1)s+=`<line x1="${X(h)}" x2="${X(h)}" y1="${pt}" y2="${H-pb}" class="hv"/><text x="${X(h)+5}" y="${pt+11}" class="hvl">halving</text>`;});
 const band=(a,b,fill)=>{const A=fcPts(a),B=fcPts(b).slice().reverse();s+=`<polygon points="${A.concat(B).map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')}" fill="${fill}"/>`;};
 band('p5','p95','url(#bf)');band('p25','p75','url(#bf2)');
 const line=(arr,col,w,dash)=>{let d='',pen=false;for(let i=i0;i<N;i++){const v=arr[i];if(v==null){pen=false;continue;}d+=(pen?'L':'M')+X(D.dates[i]).toFixed(1)+' '+Y(v).toFixed(1);pen=true;}
  s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" ${dash?`stroke-dasharray="${dash}"`:''} stroke-linejoin="round"/>`;};
 line(D.ma200w,C('--violet'),1.4,'5 4');line(D.real,C('--amber'),1.4,'5 4');
 let ad='M'+X(D.dates[i0]).toFixed(1)+' '+(H-pb);for(let i=i0;i<N;i++)ad+='L'+X(D.dates[i]).toFixed(1)+' '+Y(D.price[i]).toFixed(1);ad+='L'+X(LAST).toFixed(1)+' '+(H-pb)+'Z';
 s+=`<path d="${ad}" fill="url(#af)"/>`;
 line(D.price,C('--fg'),1.7);
 geo.ev=[];
 if(showEv)D.events.forEach(e=>{if(e.d<x0||e.d>LAST)return;const p=D.price[e.d-D.dates[0]];if(p==null)return;
  const col=e.r==null?C('--muted'):(e.r>=0?C('--green'):C('--red'));geo.ev.push({x:X(e.d),y:Y(p),e});
  s+=`<circle cx="${X(e.d)}" cy="${Y(p)}" r="4" fill="${col}" stroke="${C('--bg2')}" stroke-width="1.5"/>`;});
 s+=`<path d="${fcPts('p50').map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('')}" fill="none" stroke="${C('--accent')}" stroke-width="2.2" stroke-dasharray="6 4"/>`;
 s+=`<line x1="${X(LAST)}" x2="${X(LAST)}" y1="${pt}" y2="${H-pb}" class="hv"/>`;
 D.fc.forEach(f=>{s+=`<circle cx="${X(LAST+f.h)}" cy="${Y(f.p50)}" r="4" fill="${C('--accent')}" stroke="${C('--bg2')}" stroke-width="2"/>`;});
 s+=`<circle cx="${X(LAST)}" cy="${Y(D.price[N-1])}" r="5" fill="${C('--fg')}" stroke="${C('--bg2')}" stroke-width="2"/></g></svg>`;
 el.querySelector('svg')?.remove();el.insertAdjacentHTML('afterbegin',s);
}
function hover(e){
 const r=el.getBoundingClientRect(),px=e.clientX-r.left;if(px<geo.pl||px>geo.W-geo.pr){off();return;}
 const x=geo.x0+(px-geo.pl)/(geo.W-geo.pl-geo.pr)*(geo.x1-geo.x0);let html;
 const near=(geo.ev||[]).reduce((b,o)=>Math.abs(o.x-px)<9&&(!b||Math.abs(o.x-px)<Math.abs(b.x-px))?o:b,null);
 if(near){const e=near.e;html=`<b>${fmtD(e.d)}</b><div style="display:block;color:var(--fg);margin-bottom:4px">${e.t}</div><div><span>Kategoria</span><span>${e.c}</span></div>`+(e.r==null?'':`<div><span>BTC po 3 dniach</span><span style="color:${e.r>=0?C('--green'):C('--red')}">${e.r>0?'+':''}${e.r}%</span></div>`);}
 else if(x<=LAST+.5){let lo=0,hi=N-1;while(lo<hi){const m=(lo+hi)>>1;if(D.dates[m]<x)lo=m+1;else hi=m;}
  const i=Math.max(0,lo-(Math.abs(D.dates[lo-1]-x)<Math.abs(D.dates[lo]-x)?1:0));
  html=`<b>${fmtD(D.dates[i])}</b><div><span>Cena</span><span>${fmtP(D.price[i])} USD</span></div>`+(D.real[i]!=null?`<div><span>Średnia cena zakupu</span><span>${fmtP(D.real[i])}</span></div>`:'')+(D.ma200w[i]!=null?`<div><span>Średnia z 4 lat</span><span>${fmtP(D.ma200w[i])}</span></div>`:'');
 }else{const xs=Math.min(x,LAST+MAXH);html=`<b>${fmtD(Math.round(x))} <span style="color:var(--accent)">prognoza</span></b>`+
  [['P90','p90'],['Mediana','p50'],['P10','p10']].map(([a,k])=>`<div><span>${a}</span><span>${fmtP(interp(k,xs))} USD</span></div>`).join('');}
 cx.style.display='block';cx.style.left=px+'px';tip.innerHTML=html;tip.style.display='block';
 const tw=tip.offsetWidth;tip.style.left=Math.min(Math.max(px+14,0),r.width-tw)+'px';tip.style.top='12px';}
function off(){cx.style.display='none';tip.style.display='none';}
el.addEventListener('pointermove',hover);el.addEventListener('pointerleave',off);
document.querySelectorAll('#rng button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#rng button').forEach(x=>x.classList.remove('on'));b.classList.add('on');months=+b.dataset.m;draw();});

/* wykres wskaźnika złożonego */
const el2=document.getElementById('chart2'),tip2=document.getElementById('tip2');let g2={};
function draw2(){
 const W=el2.clientWidth||900,H=Math.round(Math.max(190,Math.min(260,W*.24))),pl=40,pr=16,pt=10,pb=24;
 let i0=0;while(i0<N&&D.comp[i0]==null)i0++;const x0=D.dates[i0],x1=LAST;
 const X=x=>pl+(x-x0)/(x1-x0)*(W-pl-pr),Y=v=>pt+(1-(v+1)/2)*(H-pt-pb);g2={x0,x1,pl,pr,W,H};
 let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`;
 [-1,-.5,0,.5,1].forEach(v=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${Y(v)}" y2="${Y(v)}" class="gridl" ${v===0?`style="stroke:var(--line2)"`:''}/><text x="${pl-8}" y="${Y(v)+4}" text-anchor="end" class="ax">${v>0?'+':''}${v}</text>`;});
 for(let y=new Date(x0*864e5).getUTCFullYear()+1;y<=new Date(x1*864e5).getUTCFullYear();y++){const d=Date.UTC(y,0,1)/864e5;if(y%2===0)s+=`<text x="${X(d)}" y="${H-7}" text-anchor="middle" class="ax">${y}</text>`;}
 D.halvings.forEach(h=>{if(h>=x0)s+=`<line x1="${X(h)}" x2="${X(h)}" y1="${pt}" y2="${H-pb}" class="hv"/><text x="${X(h)+5}" y="${pt+11}" class="hvl">halving</text>`;});
 const area=(f,col)=>{let d='M'+X(D.dates[i0])+' '+Y(0),last=null;for(let i=i0;i<N;i++){const v=D.comp[i];if(v==null)continue;d+='L'+X(D.dates[i]).toFixed(1)+' '+Y(f(v)).toFixed(1);last=i;}
  d+='L'+X(D.dates[last]).toFixed(1)+' '+Y(0)+'Z';s+=`<path d="${d}" fill="${col}" fill-opacity=".34"/>`;};
 area(v=>Math.max(v,0),C('--green'));area(v=>Math.min(v,0),C('--red'));
 let d='',pen=false;for(let i=i0;i<N;i++){const v=D.comp[i];if(v==null){pen=false;continue;}d+=(pen?'L':'M')+X(D.dates[i]).toFixed(1)+' '+Y(v).toFixed(1);pen=true;}
 s+=`<path d="${d}" fill="none" stroke="${C('--fg')}" stroke-width="1.3"/></svg>`;
 el2.querySelector('svg')?.remove();el2.insertAdjacentHTML('afterbegin',s);}
el2.addEventListener('pointermove',e=>{const r=el2.getBoundingClientRect(),px=e.clientX-r.left;if(px<g2.pl||px>g2.W-g2.pr){tip2.style.display='none';return;}
 const x=g2.x0+(px-g2.pl)/(g2.W-g2.pl-g2.pr)*(g2.x1-g2.x0);let i=Math.round(N-1-(LAST-x));i=Math.min(N-1,Math.max(0,i));const v=D.comp[i];
 if(v==null){tip2.style.display='none';return;}tip2.innerHTML=`<b>${fmtD(D.dates[i])}</b><div><span>Wskaźnik</span><span>${v>0?'+':''}${v.toFixed(2)}</span></div><div><span>Cena</span><span>${fmtP(D.price[i])} USD</span></div>`;
 tip2.style.display='block';tip2.style.left=Math.min(Math.max(px+14,0),r.width-tip2.offsetWidth)+'px';tip2.style.top='8px';});
el2.addEventListener('pointerleave',()=>tip2.style.display='none');
function all(){draw();draw2();}
all();let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(all,120);});
})();
</script></body></html>
"""
