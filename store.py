"""Zapis danych dziennych i dziennik prognoz (z samooceną)."""
import numpy as np
import pandas as pd

import config as C

LOG_COLS = ["made_on", "horizon", "target_date", "price0", "p10", "p50", "p90", "pred_z",
            "composite", "actual", "in_band", "err_pct"]


def load_daily():
    if not C.DAILY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.DAILY_CSV, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def merge_save(old, fetched):
    new = pd.DataFrame(fetched)
    new.index.name = "date"
    merged = new.combine_first(old) if len(old) else new     # nowe wartości mają pierwszeństwo
    merged = merged.sort_index()
    C.DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(C.DAILY_CSV, float_format="%.10g")
    return merged


def load_log():
    if not C.LOG_CSV.exists():
        return pd.DataFrame(columns=LOG_COLS)
    return pd.read_csv(C.LOG_CSV, parse_dates=["made_on", "target_date"])


def update_log(res, df):
    """Dopisz dzisiejsze prognozy (1 wiersz na horyzont/dzień) i uzupełnij zrealizowane wyniki."""
    log = load_log()
    made = pd.Timestamp(res["asof"])
    rows = [{"made_on": made, "horizon": f["horizon"], "target_date": pd.Timestamp(f["target"]),
             "price0": res["price"], "p10": f["p10"], "p50": f["p50"], "p90": f["p90"],
             "pred_z": f["pred_z"], "composite": res["composite"],
             "actual": np.nan, "in_band": np.nan, "err_pct": np.nan} for f in res["forecasts"]]
    new = pd.DataFrame(rows, columns=LOG_COLS)
    if len(log):
        key = log["made_on"].astype(str) + "|" + log["horizon"].astype(str)
        log = log[~key.isin(new["made_on"].astype(str) + "|" + new["horizon"].astype(str))]
        log = pd.concat([log, new], ignore_index=True)
    else:
        log = new
    close = df["close"]
    due = log["actual"].isna() & (log["target_date"] <= close.index.max())
    for i in log.index[due]:
        a = float(close.loc[log.at[i, "target_date"]])
        log.at[i, "actual"] = a
        log.at[i, "in_band"] = float(log.at[i, "p10"] <= a <= log.at[i, "p90"])
        log.at[i, "err_pct"] = (a / log.at[i, "p50"] - 1) * 100
    log = log.sort_values(["made_on", "horizon"])
    log.to_csv(C.LOG_CSV, index=False, float_format="%.6g")
    return log


def log_stats(log):
    d = log.dropna(subset=["actual"])
    if d.empty:
        return None
    # kierunek ma sens tylko dla prognoz, które go zakładały (środek różni się od ceny o > 0,5%)
    dirs = d[(d["p50"] / d["price0"] - 1).abs() > 0.005]
    sign_ok = np.sign(dirs["p50"] - dirs["price0"]) == np.sign(dirs["actual"] - dirs["price0"])
    out = {"n": int(len(d)), "coverage80": float(d["in_band"].mean()),
           "mape": float(d["err_pct"].abs().mean()),
           "direction": float(sign_ok.mean()) if len(dirs) >= 5 else None, "direction_n": int(len(dirs)), "by_h": {}}
    for h, g in d.groupby("horizon"):
        out["by_h"][int(h)] = {"n": int(len(g)), "coverage80": float(g["in_band"].mean()),
                               "mape": float(g["err_pct"].abs().mean())}
    return out
