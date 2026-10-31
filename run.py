"""BTC predictor: aktualizacja danych -> analiza -> prognozy -> dashboard.

  python run.py                    aktualizacja przyrostowa (domyślnie)
  python run.py --full             pełne pobranie historii od zera
  python run.py --install-schedule zadanie Harmonogramu Windows (co 6 godzin)
  python run.py --remove-schedule  usuń zadanie
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import config as C
import confirm
import dirtrack
import ensemble
import events
import extra as xt
import judge
import lev
import opthist
import model
import markets
import news
import regularity
import report
import rv
import scan
import sim
import sources
import store
import touch
import trader
import volev
import whales
import world
import worldlog

TASK = "BTC-Predictor"


def update(full=False):
    old = store.load_daily()
    since = None if (full or old.empty) else old.index.max() - pd.Timedelta(days=30)
    print("Pobieranie danych" + (" (pełna historia)" if since is None else f" od {since:%Y-%m-%d}") + " ...")
    fetched, status = sources.fetch_all(since)
    for name, st, info in status:
        print(f"  [{st:5}] {name}: {info}")
    if "close" not in fetched:
        sys.exit("Brak ceny BTC - przerywam (dane nie zostały zmienione).")
    raw = store.merge_save(old, fetched)
    df = model.build_frame(raw)

    res = model.analyze(df)                      # sygnały, wskaźnik złożony, reżim
    fr = res["frame"]
    print("Zdarzenia i atrybucja ruchów ...")
    E = events.analyze_events(df)
    print("Dziennik świata (Wikipedia) i nagłówki o Bitcoinie (GDELT) ...")
    try:
        last = df.index[-1] - pd.Timedelta(days=1)
        worldlog.update(last, budget_s=3600 if full else 300)
        worldlog.save_daily(pd.date_range(worldlog.START, last))
    except Exception as e:
        print("  [uwaga] dziennik świata:", str(e)[:120])
    try:
        if not os.environ.get("BTC_SKIP_NEWS"):                 # pozwala pominąć, gdy nagłówki pobiera inny proces
            news.update_news(df, budget_s=1800 if full else 120)
    except Exception as e:
        print("  [uwaga] nagłówki GDELT:", str(e)[:120])
    print("Akcje i obligacje (S&P 500 + fundusze) ...")
    MKT = None
    try:
        px, sp = markets.update()
        mf = markets.features(px, sp)
        mf.to_csv(markets.FEAT, float_format="%.6g")
        MKT = {"comove": markets.comove(df, px, sp), "analogs": markets.corr_analogs(df, px), "feat": mf}
    except Exception as e:
        print("  [uwaga] rynki:", str(e)[:150])
    print("Płynność dolara, pozycje CFTC, DeFi, Lightning, Hacker News, pogoda kosmiczna ...")
    try:
        xt.update()
    except Exception as e:
        print("  [uwaga] dodatkowe sygnały:", str(e)[:120])
    print("Zmienność z danych 5-minutowych (Binance) ...")
    try:
        rv.update()
    except Exception as e:
        print("  [uwaga] dane 5-minutowe:", str(e)[:120])
    print("Historia opcji Deribit i dźwignia Binance ...")
    for nm, fn in (("historia opcji", lambda: opthist.update(max_days=30)), ("dźwignia", lev.update)):
        try:
            fn()
        except Exception as e:
            print(f"  [uwaga] {nm}:", str(e)[:120])
    print("Duzi gracze: fundusze ETF, Strategy, traderzy Hyperliquid ...")
    try:
        whales.update()
    except Exception as e:
        print("  [uwaga] duzi gracze:", str(e)[:120])
    print("Dane ze świata (rynki, uwaga, pogoda, Słońce) ...")
    try:
        raw_w, fails = world.update_world(full=full or not world.WORLD_CSV.exists())
        if fails:
            print("  [uwaga] brak szeregów:", ", ".join(fails))
    except Exception as e:
        print("  [uwaga] świat:", str(e)[:120])
        raw_w = pd.read_csv(world.WORLD_CSV, index_col=0, parse_dates=True) if world.WORLD_CSV.exists() else None
    if raw_w is not None and len(raw_w):
        extra = pd.DataFrame(index=df.index)
        for k in ("ntx", "fees", "txvol", "uaddr", "difficulty", "hash", "minerrev", "adr", "volume", "mvrv", "fng"):
            if k in df:
                extra[f"d_{k}"] = df[k]
        if "cbclose" in df:
            extra["d_cbprem"] = (df["cbclose"] / df["close"] - 1) * 100
        if "bfund" in df:
            extra["d_bfund"] = df["bfund"].rolling(7, min_periods=3).mean() * 100
        cols = raw_w.columns
        if "y_btcf" in cols:
            extra["d_basis"] = (raw_w["y_btcf"].reindex(df.index).ffill(limit=4) / df["close"] - 1) * 100
        if "y_c_eth" in cols:
            extra["d_ethbtc"] = raw_w["y_c_eth"].reindex(df.index) / df["close"]
            alts = [c for c in ("y_c_xrp", "y_c_ltc", "y_c_bch", "y_c_xmr", "y_c_ada", "y_c_trx", "y_c_link", "y_c_doge") if c in cols]
            if alts:                                     # stały koszyk notowany od 2017, liczony tylko gdy są wszystkie (bez skoków przy debiutach)
                A = raw_w[alts].reindex(df.index).dropna()
                extra["d_altbtc"] = np.exp(np.log(A).mean(axis=1)) / df["close"].reindex(A.index)
        raw_x = raw_w.loc["2011-08-01":].join(extra, how="outer")
        T = world.transforms(raw_x, df.index)
    else:
        T = None
    print("Regularność układu i skaner zależności ...")
    Rg = regularity.run(df)
    S = scan.run_scan(df, T) if T is not None else None
    print("Test potwierdzający (duzi gracze, hipotezy spisane z góry) ...")
    CF = None
    try:
        CF = confirm.run(df)
        print("  " + ", ".join(f'{r["id"]}: {r["status"]}' for r in CF["rows"]))
    except Exception as e:
        print("  [uwaga] test potwierdzający:", str(e)[:120])
    DT = None
    try:                                                         # osobny tor kierunku (szansa wzrostu w 7 dni, wynik Briera)
        DT = dirtrack.run(df, CF)
        if DT:
            print(f'  tor kierunku: {"zdał" if DT["passed"] else "nie zdał"}, dziś {DT["p_now"] * 100:.0f}% szans na wzrost w 7 dni')
    except Exception as e:
        print("  [uwaga] tor kierunku:", str(e)[:120])
    conf = None
    if CF and CF["confirmed"]:                                   # tylko potwierdzone sygnały trafiają do osobnej metody w zespole
        conf = pd.concat([confirm._z(CF["features"][c]) for c in CF["confirmed"]], axis=1).mean(axis=1)
    print("Zespół modeli (walk-forward) ...")
    SHORT = [1, 3]                                               # krótkie horyzonty: tylko test, czy da się pokonać rzut monetą
    ens = ensemble.run(df, E["events"], fr["features"], fr["scores"], fr["composite"], Rg, T, conf, SHORT + C.HORIZONS)
    ens["short"] = {h: ens["horizons"].pop(h) for h in SHORT}
    ens["forecasts"] = [f for f in ens["forecasts"] if f["horizon"] not in SHORT]
    print("Szanse dotknięcia poziomów ceny ...")
    TCH, SIM = None, None
    sig_now = ensemble.vol_models(df)[0]
    try:
        TCH = touch.run(df, sig_now)
    except Exception as e:
        print("  [uwaga] szanse dotknięcia:", str(e)[:120])
    print("Sędzia: rynki zakładów (Polymarket, Kalshi) ...")
    JD = None
    try:
        JD = judge.run(df, sig_now, (TCH or {}).get("best", "drift"), df.index[-1])
    except Exception as e:
        print("  [uwaga] sędzia:", str(e)[:120])
    print("Spokojnie czy burzliwie (zdarzenia zmienności) ...")
    VE = None
    try:
        VE = volev.run(df, sig_now)
        print("  " + ", ".join(f'{k}: {v["p_now"] * 100:.0f}%{" (potwierdzone)" if v["confirmed"] else ""}' for k, v in VE.items() if v.get("p_now") is not None))
    except Exception as e:
        print("  [uwaga] zdarzenia zmienności:", str(e)[:120])
    print("Wirtualny portfel (decyzja dnia i test na historii) ...")
    try:
        SIM = sim.run(df, sig_now)
        if SIM.get("live"):
            print(f"  portfel: {SIM['live']['value']:,.0f} zł przy wpłatach {SIM['live']['deposited']:,.0f} zł, w BTC {SIM['live']['w'] * 100:.0f}%")
    except Exception as e:
        print("  [uwaga] wirtualny portfel:", str(e)[:120])
    TR = None
    try:                                                         # wirtualny trader: decyzja przy każdej aktualizacji w chmurze
        TR = trader.run(df, sig_now, VE)
        if TR.get("log"):
            r0 = TR["log"][0]
            print(f'  trader: {r0["local"]} {r0["action"]} {r0["amount_pln"]:,.0f} zł, portfel {TR["value"]:,.0f} zł' + ("" if TR["wrote"] else " (bez nowej decyzji)"))
    except Exception as e:
        print("  [uwaga] trader:", str(e)[:120])
    res["forecasts"] = ens["forecasts"]
    for f in res["forecasts"]:
        f["pred_z"] = 0.0
        f["p5"], f["p95"] = f.get("p5"), f.get("p95")
    res["quality"] = {h: {"active": v["alpha"] > 0, "n": v["n"]} for h, v in ens["horizons"].items()}
    log = store.update_log(res, df)
    stats = store.log_stats(log)
    try:                                                         # dziennik na przyszłość: każda metoda, tylko dopisywany
        ens["ledger"] = store.ledger_stats(store.update_ledger(res["asof"], res["price"], ens, df["close"]))
    except Exception as e:
        print("  [uwaga] dziennik metod:", str(e)[:120])

    C.OUT_DIR.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "frame"}
    slim["status"] = status
    slim["log_stats"] = stats
    slim["ensemble"] = {k: v for k, v in ens.items() if k not in ("Xraw", "hmm")}
    slim["ensemble"]["hmm"] = {k: v for k, v in ens["hmm"].items() if k != "p_hi_series"}
    slim["events"] = {"table": E["table"].round(4).to_dict("records"), "moves": E["moves"], "cover": E["cover"], "upcoming": E["upcoming"]}
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["scan"] = ({"summary": S["summary"], "discoveries": S["discoveries"].round(4).to_dict("records"),
                     "candidates": S["candidates"].round(4).to_dict("records")} if S else None)
    slim["regularity"] = {k: v for k, v in Rg.items() if k not in ("rolling", "kalman")}
    slim["regularity"]["kalman"] = {k: v for k, v in Rg["kalman"].items() if k != "series"}
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["confirm"] = {k: v for k, v in CF.items() if k != "features"} if CF else None
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["touch"] = {k: v for k, v in TCH.items()} if TCH else None
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["volev"] = VE
    slim["judge"] = JD
    slim["dirtrack"] = DT
    slim["trader"] = {k: v for k, v in (TR or {}).items() if k != "log"} or None
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    report.write_dashboard(res, status, log, stats, ens, E, df, S, Rg, MKT, CF, TCH, SIM, VE, JD, DT, TR)
    report.write_journal(TR)
    print_summary(res)
    return res


def print_summary(res):
    print(f"\nBTC {res['price']:,.0f} USD  ({res['asof']})  zmienność dzienna {res['daily_vol']*100:.2f}%")
    print(f"Wskaźnik złożony: {res['composite']:+.2f}  -> {res['regime']}")
    print(f"{'Horyzont':>9} {'Data':>11} {'P10':>9} {'Mediana':>9} {'P90':>9} {'P(wzrost)':>10}  model")
    for f in res["forecasts"]:
        q = res["quality"][f["horizon"]]
        tag = "zespół modeli" if q["active"] else "tylko zmienność"
        print(f"{f['horizon']:>7} d {f['target']:>11} {f['p10']:>9,.0f} {f['p50']:>9,.0f} {f['p90']:>9,.0f} {f['prob_up']*100:>9.0f}%  {tag}")
    print(f"\nDashboard: {C.DASHBOARD_HTML}")


def schedule(install=True):
    if install:
        cmd = ["schtasks", "/Create", "/F", "/TN", TASK, "/SC", "HOURLY", "/MO", "6",
               "/TR", f'"{sys.executable}" "{C.ROOT / "run.py"}"']
    else:
        cmd = ["schtasks", "/Delete", "/F", "/TN", TASK]
    print(subprocess.run(cmd, capture_output=True, text=True).stdout or "gotowe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--install-schedule", action="store_true")
    ap.add_argument("--remove-schedule", action="store_true")
    a = ap.parse_args()
    if a.install_schedule:
        schedule(True)
    elif a.remove_schedule:
        schedule(False)
    else:
        update(a.full)
