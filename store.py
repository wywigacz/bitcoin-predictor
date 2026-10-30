"""Zapis danych dziennych i dziennik prognoz (z samooceną)."""
import numpy as np
import pandas as pd

import config as C

LOG_COLS = ["made_on", "horizon", "target_date", "price0", "p10", "p50", "p90", "pred_z",
            "composite", "actual", "in_band", "err_pct", "p5", "p95", "prob_up"]


def load_daily():
    if not C.DAILY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.DAILY_CSV, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


# Strażnik danych: w październiku 2026 kilka kolumn wymieszało się między sobą (np. cena w wonach w kolumnie DVOL).
# Wartości niemożliwe fizycznie usuwamy, zanim trafią do modeli.
_in = lambda s, lo, hi: s.isna() | s.between(lo, hi)           # brak danych do porównania nie jest błędem
CHECKS = {
    "cbclose": lambda d: _in(d["cbclose"] / d["close"], 0.8, 1.25),
    "dvol": lambda d: _in(d["dvol"], 10, 400),
    "btckrw": lambda d: _in(d["btckrw"] / d["usdkrw"].ffill(limit=5) / d["close"], 0.6, 2.0) if "usdkrw" in d else d["btckrw"] > 0,
    "pcr": lambda d: _in(d["pcr"], 0.05, 20),
    "cot_lev": lambda d: _in(d["cot_lev"], -1, 1),           # pozycja netto jako ułamek otwartych pozycji
    "fng": lambda d: _in(d["fng"], 0, 100),
}


def sanity(df):
    for col, ok in CHECKS.items():
        if col in df:
            m = ok(df) | df[col].isna()
            m = m.fillna(True)                       # brak danych do porównania: zostawiamy
            if (~m).any():
                print(f"  [strażnik] {col}: usunięto {int((~m).sum())} nieprawdopodobnych wartości")
                df.loc[~m, col] = np.nan
    return df


def merge_save(old, fetched):
    new = pd.DataFrame(fetched)
    new.index.name = "date"
    merged = new.combine_first(old) if len(old) else new     # nowe wartości mają pierwszeństwo
    merged = merged.sort_index()
    merged = sanity(merged)
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
             "actual": np.nan, "in_band": np.nan, "err_pct": np.nan,
             "p5": f.get("p5"), "p95": f.get("p95"), "prob_up": f.get("prob_up")} for f in res["forecasts"]]
    new = pd.DataFrame(rows, columns=LOG_COLS)
    if len(log):
        # tylko dopisywanie (od 2026-10-30): pierwszy zapis z danego dnia wygrywa, wcześniejszych prognoz nigdy nie poprawiamy
        key = log["made_on"].astype(str) + "|" + log["horizon"].astype(str)
        new = new[~(new["made_on"].astype(str) + "|" + new["horizon"].astype(str)).isin(key)]
        log = pd.concat([log, new], ignore_index=True) if len(new) else log
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


LEDGER_CSV = C.DATA_DIR / "ledger.csv"
QCOLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]


def _code_version():
    import os
    import subprocess
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.ROOT, timeout=10).stdout.strip()
        except Exception:
            sha = ""
    return sha[:10]


def update_ledger(asof, price, ens, close):
    """Dziennik na przyszłość (od 2026-10-30): codziennie kwantyle KAŻDEJ metody na każdy horyzont, w cenach, z numerem wersji kodu.
    Tylko dopisywany: wiersz (dzień, horyzont, metoda) zapisany raz nigdy się nie zmienia; po terminie dopisujemy tylko wynik.
    To jedyny w pełni czysty test: żadna z tych prognoz nie mogła widzieć swojej przyszłości."""
    made = pd.Timestamp(asof)
    sha = _code_version()
    rows = []
    blocks = list((ens.get("horizons") or {}).items()) + list((ens.get("short") or {}).items())
    for H, v in blocks:
        H = int(H)
        scale = float(v.get("sigma_H") or np.nan)
        cand = dict(v.get("live_z") or {})
        if v.get("ens_z") is not None:
            cand["published"] = v["ens_z"]
        for m, z in cand.items():
            z = np.asarray(z, float)
            if len(z) != 7 or not np.isfinite(z).all() or not np.isfinite(scale):
                continue
            rows.append({"made_on": made, "horizon": H, "model": m, "target_date": made + pd.Timedelta(days=H), "price0": price,
                         **{c: float(price * np.exp(scale * q)) for c, q in zip(QCOLS, np.sort(z))}, "scale": scale, "code": sha,
                         "actual": np.nan, "in80": np.nan, "pinball": np.nan})
    new = pd.DataFrame(rows)
    old = pd.read_csv(LEDGER_CSV, parse_dates=["made_on", "target_date"]) if LEDGER_CSV.exists() else pd.DataFrame()
    if len(old) and len(new):
        k_old = old["made_on"].astype(str) + "|" + old["horizon"].astype(str) + "|" + old["model"]
        k_new = new["made_on"].astype(str) + "|" + new["horizon"].astype(str) + "|" + new["model"]
        new = new[~k_new.isin(k_old)]
    L = pd.concat([old, new], ignore_index=True) if len(old) else new
    if L.empty:
        return L
    due = L["actual"].isna() & (L["target_date"] <= close.index.max())
    qs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    for i in L.index[due]:
        a = float(close.asof(L.at[i, "target_date"]))
        L.at[i, "actual"] = a
        L.at[i, "in80"] = float(L.at[i, "q10"] <= a <= L.at[i, "q90"])
        y = np.log(a / L.at[i, "price0"]) / L.at[i, "scale"]
        qz = np.log(L.loc[i, QCOLS].to_numpy(float) / L.at[i, "price0"]) / L.at[i, "scale"]
        d = y - qz
        L.at[i, "pinball"] = float(np.maximum(qs * d, (qs - 1) * d).mean())
    L = L.sort_values(["made_on", "horizon", "model"])
    L.to_csv(LEDGER_CSV, index=False, float_format="%.6g")
    return L


def ledger_stats(L):
    """Rozliczenie dziennika: dla każdej metody i horyzontu pokrycie pasma 80% i przewaga nad „bez zmian” (tylko rozstrzygnięte)."""
    if L is None or L.empty:
        return None
    d = L.dropna(subset=["actual"])
    out = {"rows": int(len(L)), "matured": int(len(d)), "first": str(L["made_on"].min())[:10], "by": []}
    for (H, m), g in d.groupby(["horizon", "model"]):
        nv = d[(d["horizon"] == H) & (d["model"] == "naive")].set_index("made_on")["pinball"]
        gg = g.set_index("made_on")["pinball"]
        j = gg.index.intersection(nv.index)
        sk = float(1 - gg[j].mean() / nv[j].mean()) if len(j) and m != "naive" and nv[j].mean() > 0 else np.nan
        out["by"].append({"horizon": int(H), "model": m, "n": int(len(g)), "cov80": float(g["in80"].mean()), "skill": sk})
    return out


def log_stats(log):
    d = log.dropna(subset=["actual"])
    if d.empty:
        return None
    # kierunek ma sens tylko dla prognoz, które go zakładały (środek różni się od ceny o > 0,5%)
    dirs = d[(d["p50"] / d["price0"] - 1).abs() > 0.005]
    sign_ok = np.sign(dirs["p50"] - dirs["price0"]) == np.sign(dirs["actual"] - dirs["price0"])
    out = {"n": int(len(d)), "coverage80": float(d["in_band"].mean()),
           "mape": float(d["err_pct"].abs().mean()),
           "direction": float(sign_ok.mean()) if len(dirs) >= 5 else None, "direction_n": int(len(dirs)), "by_h": {},
           "first": str(log["made_on"].min())[:10]}
    # wynik Briera dla „szansy wzrostu” (0 = idealnie, 0,25 = rzut monetą); liczony tylko dla prognoz, które ją zapisały
    if "prob_up" in d and d["prob_up"].notna().sum() >= 5:
        b = d.dropna(subset=["prob_up"])
        up = (b["actual"] > b["price0"]).astype(float)
        out["brier"] = float(((b["prob_up"] - up) ** 2).mean())
        out["brier_n"] = int(len(b))
    if "p5" in d and d["p5"].notna().sum() >= 5:
        b = d.dropna(subset=["p5", "p95"])
        out["coverage90"] = float(((b["actual"] >= b["p5"]) & (b["actual"] <= b["p95"])).mean())
    for h, g in d.groupby("horizon"):
        out["by_h"][int(h)] = {"n": int(len(g)), "coverage80": float(g["in_band"].mean()),
                               "mape": float(g["err_pct"].abs().mean())}
    return out
