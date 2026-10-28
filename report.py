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
    ("Wycena on-chain", "Czy przeciętny posiadacz jest na zysku? Najlepsze wskaźniki dna.", ["mvrv", "ma200w", "mayer"]),
    ("Górnicy", "Stres i kapitulacja górników bywały sygnałem dna.", ["puell", "hashribbon"]),
    ("Rynek i dźwignia", "Nastroje, lewar na kontraktach i „suchy proch”.", ["fng", "funding", "stable"]),
    ("Makro i płynność", "Od 2020 BTC jest aktywem ryzykownym, ale reżim się zmienia.", ["m2", "ndx", "dxy", "vix", "rates"]),
    ("Cykl", "Halvingi i ostrzeżenie o szczycie.", ["halving", "pi"]),
]
LABELS = {
    "mvrv": "MVRV", "ma200w": "200-tygodniowa średnia", "mayer": "Mayer Multiple", "puell": "Puell Multiple",
    "hashribbon": "Hash Ribbons", "fng": "Fear & Greed", "funding": "Funding rate", "stable": "Stablecoiny 30d",
    "m2": "Płynność M2", "dxy": "Indeks dolara", "ndx": "Nasdaq 90d", "vix": "VIX", "rates": "Krótkie stopy",
    "pi": "Pi Cycle Top", "halving": "Faza halvingu",
}
SRC_NOTE = {
    "mvrv": "cena / Realized Price", "ma200w": "cena względem średniej z 4 lat", "mayer": "cena / średnia 200d",
    "puell": "przychody górników vs rok", "hashribbon": "30d vs 60d hash rate", "fng": "kontrariańsko",
    "funding": "średnia 7d na 8 h", "stable": "zmiana podaży", "m2": "opóźnienie ~70 dni", "dxy": "zmiana 30d",
    "ndx": "momentum", "vix": "poziom", "rates": "zmiana 180d, p.p.", "pi": "111d / (2×350d)", "halving": "dni od halvingu",
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
            "puell": n(g("puell"), 2), "hashribbon": "kapitulacja" if g("hashcap") == 1 else ("brak" if g("hashcap") == 0 else "–"),
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
<text x="{cx-r}" y="{cy+22}" text-anchor="middle" class="gl">przegrzanie</text>
<text x="{cx+r}" y="{cy+22}" text-anchor="middle" class="gl">okazja</text></svg>"""


def cycle_html(days):
    total = 1460
    zones = [(0, 365, "Ekspansja", "var(--blue)"), (365, 560, "Szczyt", "var(--amber)"),
             (560, 770, "Bessa", "var(--red)"), (770, 1000, "Dno", "var(--green)"), (1000, 1460, "Odbudowa", "var(--violet)")]
    seg = "".join(f'<div class="zone" style="flex:{b-a};--c:{c}" title="{nm}: dni {a}–{b} po halvingu"><span>{nm}</span></div>' for a, b, nm, c in zones)
    d = days if days is not None and not math.isnan(days) else 0
    cur = next((nm for a, b, nm, _ in zones if a <= d < b), "Odbudowa")
    pos = min(max(d / total, 0), 1) * 100
    return (f'<div class="cycle"><div class="zones">{seg}<div class="marker" style="left:{pos:.2f}%"><i></i></div></div>'
            f'<div class="cyc-foot"><b>{n(d)} dni</b> po halvingu z {C.HALVINGS[-1]} · faza: <b>{cur}</b> · '
            f'następny halving ok. {C.NEXT_HALVING}</div></div>')


def forecast_cards(res):
    cards = ""
    p0 = res["price"]
    for f in res["forecasts"]:
        q = res["quality"][f["horizon"]]
        lo, hi = math.log(f["p5"]), math.log(f["p95"])
        pos = lambda v: (math.log(v) - lo) / (hi - lo) * 100
        chg = (f["p50"] / p0 - 1) * 100
        up = f["prob_up"] * 100
        badge = ('<span class="badge on">przechył sygnałów</span>' if q["active"]
                 else '<span class="badge">bez przechyłu</span>')
        cards += f"""<article class="card fc">
<header><h3>{f['horizon']} dni</h3>{badge}</header>
<div class="muted sm">do {f['target']}</div>
<div class="fc-med">{n(f['p50'])}<small> USD</small></div>
<div class="fc-chg {'pos' if chg >= 0 else 'neg'}">{chg:+.1f}% vs dziś</div>
<div class="range"><div class="r90" style="left:{pos(f['p10']):.1f}%;width:{pos(f['p90'])-pos(f['p10']):.1f}%"></div>
<div class="r50" style="left:{pos(f['p25']):.1f}%;width:{pos(f['p75'])-pos(f['p25']):.1f}%"></div>
<div class="tick" style="left:{pos(f['p50']):.1f}%"></div><div class="now" style="left:{pos(p0):.1f}%" title="cena dziś"></div></div>
<div class="range-l"><span>{n(f['p10'])}</span><span class="muted">80% pasmo</span><span>{n(f['p90'])}</span></div>
<div class="pup"><div class="ring" style="--p:{up:.0f}"><b>{up:.0f}%</b></div><span class="muted sm">szansa, że cena<br>będzie wyższa niż dziś</span></div>
</article>"""
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
            rows += (f'<div class="sig"><div class="sig-l"><b>{LABELS[k]}</b><span class="muted sm">{SRC_NOTE[k]}</span></div>'
                     f'{bar}<div class="sig-v"><b>{val}</b><span class="muted sm">waga {p["weight"]:g}'
                     f'{"" if s is None else f" · {s:+.2f}"}</span></div></div>')
        out += f'<section class="card grp"><h3>{title}</h3><p class="muted sm">{sub}</p>{rows}</section>'
    return out


def quality_html(res):
    out = ""
    for h, q in res["quality"].items():
        on = q["active"]
        ic = n(q["ic"], 3) if q["ic"] == q["ic"] else "–"
        sk = n(q["skill"] * 100, 1, "%") if q["skill"] == q["skill"] else "–"
        hit = n(q["hit"] * 100, 0, "%") if q["hit"] == q["hit"] else "–"
        out += (f'<div class="card q"><header><h3>{h} dni</h3><span class="badge {"on" if on else ""}">'
                f'{"przechył włączony" if on else "tylko zmienność"}</span></header>'
                f'<dl><div><dt>Korelacja z wynikiem (IC)</dt><dd>{ic}</dd></div>'
                f'<div><dt>Poprawa błędu vs „bez zmian”</dt><dd>{sk}</dd></div>'
                f'<div><dt>Trafność kierunku</dt><dd>{hit}</dd></div>'
                f'<div><dt>Punktów testu</dt><dd>{q["n"]}</dd></div></dl></div>')
    return out


def track_html(stats):
    if not stats:
        return ('<div class="card empty"><b>Dziennik prognoz się zapełnia</b>'
                '<p class="muted">Każda prognoza jest zapisywana i porównywana z rzeczywistą ceną po upływie horyzontu. '
                'Pierwsze oceny pojawią się po 7 dniach od uruchomienia.</p></div>')
    cov = stats["coverage80"] * 100
    return (f'<div class="tiles"><div class="card tile"><span class="muted sm">Oceniono prognoz</span><b>{stats["n"]}</b></div>'
            f'<div class="card tile"><span class="muted sm">Pokrycie pasma 80%</span><b>{cov:.0f}%</b><span class="muted sm">cel: 80%</span></div>'
            f'<div class="card tile"><span class="muted sm">Średni błąd mediany</span><b>{stats["mape"]:.1f}%</b></div>'
            f'<div class="card tile"><span class="muted sm">Trafność kierunku</span><b>{stats["direction"]*100:.0f}%</b></div></div>')


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
def write_dashboard(res, status, log, stats):
    fr = res["frame"]
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
            .replace("%%QUALITY%%", quality_html(res))
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
.reveal{opacity:0;transform:translateY(10px);animation:rise .7s cubic-bezier(.2,.8,.2,1) forwards}
@keyframes rise{to{opacity:1;transform:none}}
@media (max-width:980px){.hero{grid-template-columns:1fr 1fr}.price-card{grid-column:1/-1}.grid4{grid-template-columns:repeat(2,1fr)}.sgrid,.two{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
@media (max-width:620px){.hero,.grid4{grid-template-columns:1fr}.deltas{grid-template-columns:repeat(2,1fr)}.sig{grid-template-columns:1fr 1fr;gap:8px 14px}.sig .dv{grid-column:1/-1;order:3}.wrap{padding:18px 14px 40px}}
@media (prefers-reduced-motion:reduce){.reveal,.needle{animation:none;opacity:1;transform:none}.needle{transform:rotate(var(--rot))}}
</style></head><body><div class="wrap">

<div class="top reveal">
 <div class="brand"><div class="logo"><svg viewBox="0 0 24 24"><path d="M8 4h6.5a3.6 3.6 0 0 1 1.2 7 3.8 3.8 0 0 1-1.5 7.3H8zM11 7v3.6h3a1.8 1.8 0 0 0 0-3.6zm0 6.4V17h3.6a1.8 1.8 0 0 0 0-3.6z" fill="#fff"/></svg></div>
 <div><h1>BTC Predictor</h1><span>Prognozy rozkładu ceny na bazie wskaźników on-chain, makro i cykli</span></div></div>
 <div class="status"><span>dane z %%ASOF%% · wygenerowano %%NOW%%</span><span class="chip %%SRCCLS%%"><i></i>źródła %%SRCOK%%</span></div>
</div>

<section class="hero">
 <div class="card price-card reveal" style="animation-delay:.05s"><div class="lbl">Bitcoin · USD</div>
  <div class="big">%%PRICE%%<small>USD</small></div><div class="muted sm">%%ATH%%</div>
  <div class="deltas">%%DELTAS%%</div></div>
 <div class="card gauge-card reveal" style="animation-delay:.12s"><div class="lbl muted sm" style="letter-spacing:.08em;text-transform:uppercase">Wskaźnik złożony</div>
  %%GAUGE%%<div class="gv">%%COMP%%</div><div class="gr">%%REGIME%%</div></div>
 <div class="card kv reveal" style="animation-delay:.19s">
  <div class="row"><span class="muted">Realized Price</span><b>%%REAL%% <small class="muted">USD</small></b></div>
  <div class="row"><span class="muted">MVRV</span><b>%%MVRV%%</b></div>
  <div style="padding-top:2px"><span class="muted sm">Faza cyklu halvingowego</span>%%CYCLE%%</div></div>
</section>

<h2>Cena i prognoza</h2>
<section class="card chart-card reveal" style="animation-delay:.1s">
 <div class="chart-head"><div class="legend"><span><i style="background:var(--fg)"></i>Cena</span><span><i style="background:var(--accent)"></i>Mediana prognozy</span>
  <span><i style="background:var(--amber)"></i>Realized Price</span><span><i style="background:var(--violet)"></i>200-tyg. średnia</span></div>
  <div class="seg" id="rng"><button data-m="3">3M</button><button data-m="12" class="on">1R</button><button data-m="36">3L</button><button data-m="0">Wszystko</button></div></div>
 <div class="chart" id="chart"><div class="cx" id="cx"></div><div class="tip" id="tip"></div></div>
 <div class="legend2"><span><i style="width:22px;height:8px;border-radius:4px;background:color-mix(in srgb,var(--blue) 35%,transparent)"></i>pasmo 90% (P5–P95)</span>
  <span><i style="width:22px;height:8px;border-radius:4px;background:color-mix(in srgb,var(--blue) 65%,transparent)"></i>pasmo 50% (P25–P75)</span><span>Pionowe linie: halvingi</span></div>
</section>

<h2>Prognozy na horyzonty</h2>
<section class="grid4">%%FCARDS%%</section>
<p class="note">Pasmo 80% (P10–P90) powinno obejmować cenę w ok. 4 przypadkach na 5. Jego szerokość wynika z bieżącej zmienności i empirycznego rozkładu błędów z całej historii BTC. Pomarańczowy znacznik to mediana, biała kropka to dzisiejsza cena.</p>

<h2>Sygnały składowe</h2>
<section class="sgrid">%%SIGNALS%%</section>
<p class="note">Każdy sygnał ma ocenę od −1 (historycznie przegrzanie) do +1 (historycznie okazja). Wskaźnik złożony to średnia ważona dostępnych sygnałów; wagi odzwierciedlają to, jak sprawdzały się wskaźniki w opracowaniach (najwyżej: MVRV i 200-tygodniowa średnia). Wagi zmienisz w <code>config.py</code>.</p>

<h2>Wskaźnik złożony w czasie</h2>
<section class="card chart-card reveal"><div class="chart" id="chart2"><div class="tip" id="tip2"></div></div></section>

<h2>Czy model ma przewagę? Test na historii</h2>
<section class="grid4">%%QUALITY%%</section>
<p class="note">Test walk-forward: w każdym punkcie model uczy się wyłącznie na danych znanych wtedy, po czym prognozuje przyszłość. „Przechył sygnałów” (przesunięcie mediany według wskaźnika złożonego) włącza się tylko, gdy korelacja IC ≥ %%MIN_IC%% i błąd jest mniejszy niż przy prognozie „bez zmian”. W przeciwnym razie mediana to cena bieżąca, a pasmo wynika wyłącznie ze zmienności. Obserwacje na dłuższych horyzontach nakładają się, więc efektywna liczba niezależnych punktów jest znacznie mniejsza niż „punktów testu”. Traktuj wynik ostrożnie.</p>

<h2>Skuteczność zapisanych prognoz</h2>
%%TRACK%%

<div class="two"><div><h2 style="margin-top:30px">Korelacja z BTC · 90 dni</h2><div class="card">%%CORR%%</div></div>
<div><h2 style="margin-top:30px">Źródła danych</h2><div class="pills">%%PILLS%%</div></div></div>

<div class="disc"><b>To nie jest rekomendacja inwestycyjna.</b> Cykle halvingowe opierają się na czterech obserwacjach, progi wskaźników dobrano na historii, a struktura rynku (ETF-y, spółki skarbcowe) się zmienia. Traktuj wynik jako rozkład scenariuszy, nie przepowiednię.</div>
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
let months=12;
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
 s+=`<path d="${fcPts('p50').map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('')}" fill="none" stroke="${C('--accent')}" stroke-width="2.2" stroke-dasharray="6 4"/>`;
 s+=`<line x1="${X(LAST)}" x2="${X(LAST)}" y1="${pt}" y2="${H-pb}" class="hv"/>`;
 D.fc.forEach(f=>{s+=`<circle cx="${X(LAST+f.h)}" cy="${Y(f.p50)}" r="4" fill="${C('--accent')}" stroke="${C('--bg2')}" stroke-width="2"/>`;});
 s+=`<circle cx="${X(LAST)}" cy="${Y(D.price[N-1])}" r="5" fill="${C('--fg')}" stroke="${C('--bg2')}" stroke-width="2"/></g></svg>`;
 el.querySelector('svg')?.remove();el.insertAdjacentHTML('afterbegin',s);
}
function hover(e){
 const r=el.getBoundingClientRect(),px=e.clientX-r.left;if(px<geo.pl||px>geo.W-geo.pr){off();return;}
 const x=geo.x0+(px-geo.pl)/(geo.W-geo.pl-geo.pr)*(geo.x1-geo.x0);let html;
 if(x<=LAST+.5){let lo=0,hi=N-1;while(lo<hi){const m=(lo+hi)>>1;if(D.dates[m]<x)lo=m+1;else hi=m;}
  const i=Math.max(0,lo-(Math.abs(D.dates[lo-1]-x)<Math.abs(D.dates[lo]-x)?1:0));
  html=`<b>${fmtD(D.dates[i])}</b><div><span>Cena</span><span>${fmtP(D.price[i])} USD</span></div>`+(D.real[i]!=null?`<div><span>Realized Price</span><span>${fmtP(D.real[i])}</span></div>`:'')+(D.ma200w[i]!=null?`<div><span>200-tyg. średnia</span><span>${fmtP(D.ma200w[i])}</span></div>`:'');
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
