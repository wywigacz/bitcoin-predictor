"""BTC predictor: aktualizacja danych -> analiza -> prognozy -> dashboard.

  python run.py                    aktualizacja przyrostowa (domyślnie)
  python run.py --full             pełne pobranie historii od zera
  python run.py --install-schedule zadanie Harmonogramu Windows (co 6 godzin)
  python run.py --remove-schedule  usuń zadanie
"""
import argparse
import json
import subprocess
import sys

import pandas as pd

import config as C
import model
import report
import sources
import store

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

    res = model.analyze(df)
    log = store.update_log(res, df)
    stats = store.log_stats(log)

    C.OUT_DIR.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "frame"}
    slim["status"] = status
    slim["log_stats"] = stats
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    report.write_dashboard(res, status, log, stats)
    print_summary(res)
    return res


def print_summary(res):
    print(f"\nBTC {res['price']:,.0f} USD  ({res['asof']})  zmienność dzienna {res['daily_vol']*100:.2f}%")
    print(f"Wskaźnik złożony: {res['composite']:+.2f}  -> {res['regime']}")
    print(f"{'Horyzont':>9} {'Data':>11} {'P10':>9} {'Mediana':>9} {'P90':>9} {'P(wzrost)':>10}  model")
    for f in res["forecasts"]:
        q = res["quality"][f["horizon"]]
        tag = "przechył ON" if q["active"] else "bez przechyłu"
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
