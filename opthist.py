"""Historia rynku opcji Deribit odtworzona z publicznego archiwum transakcji (history.deribit.com, bez klucza, od 2018).

Na każdy dzień (UTC) bierzemy do 3 próbek po 1000 transakcji opcjami BTC (od 00:00, 08:00 i 16:00) i z transakcji z terminem
14–60 dni liczymy (delta z modelu Blacka dla podanej przy transakcji zmienności implikowanej):
  rr25   - asymetria 25-delta: mediana zmienności opcji kupna (delta 0,15–0,35) minus opcji sprzedaży (delta −0,35…−0,15), w pkt,
  atm    - zmienność „przy cenie” (opcje kupna z deltą 0,4–0,6),
  pcr    - stosunek wolumenu opcji sprzedaży do kupna (wszystkie terminy w próbce),
  ntrades - ile transakcji weszło do policzenia.
Liczymy tak samo dla przeszłości i co dzień na bieżąco, więc szereg jest spójny w czasie (H13 używa tej wersji).
Wynik: data/opt_hist.csv (dopisywany dzień po dniu)."""
import math
import sys
import time

import numpy as np
import pandas as pd
import requests

import config as C

OUT = C.DATA_DIR / "opt_hist.csv"
URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
UA = {"User-Agent": "Mozilla/5.0 (btc-predictor research)"}
START = pd.Timestamp("2018-01-01")
SLICES = (0, 8, 16)


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _fetch(s, e):
    for i in range(4):
        try:
            r = requests.get(URL, params={"currency": "BTC", "kind": "option", "start_timestamp": s, "end_timestamp": e, "count": 1000,
                                          "include_old": "true", "sorting": "asc"}, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()["result"]["trades"]
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return []


def day_stats(day):
    d0 = int(pd.Timestamp(day).timestamp() * 1000)
    trades = []
    for h in SLICES:
        s = d0 + h * 3600_000
        trades += _fetch(s, min(s + 8 * 3600_000, d0 + 86400_000) - 1)
        time.sleep(0.15)
    rows = []
    for t in trades:
        try:
            _, exp, k, cp = t["instrument_name"].split("-")
            T = (pd.Timestamp(pd.to_datetime(exp, format="%d%b%y")) + pd.Timedelta(hours=8) - pd.Timestamp(t["timestamp"], unit="ms")).total_seconds() / (365 * 86400)
            iv, F, K = float(t["iv"]) / 100, float(t["index_price"]), float(k)
            if T <= 0 or iv <= 0 or F <= 0:
                continue
            d1 = (math.log(F / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
            delta = _ncdf(d1) if cp == "C" else _ncdf(d1) - 1
            rows.append((cp, T * 365, delta, iv * 100, float(t.get("amount") or 0)))
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            continue
    if not rows:
        return None
    D = pd.DataFrame(rows, columns=["cp", "days", "delta", "iv", "amt"])
    M = D[(D["days"] >= 14) & (D["days"] <= 60)]
    c25 = M[(M["cp"] == "C") & M["delta"].between(0.15, 0.35)]["iv"]
    p25 = M[(M["cp"] == "P") & M["delta"].between(-0.35, -0.15)]["iv"]
    atm = M[(M["cp"] == "C") & M["delta"].between(0.4, 0.6)]["iv"]
    vol_p, vol_c = D.loc[D["cp"] == "P", "amt"].sum(), D.loc[D["cp"] == "C", "amt"].sum()
    return {"date": pd.Timestamp(day).strftime("%Y-%m-%d"),
            "rr25": float(c25.median() - p25.median()) if len(c25) >= 3 and len(p25) >= 3 else np.nan,
            "atm": float(atm.median()) if len(atm) >= 3 else np.nan,
            "pcr": float(vol_p / vol_c) if vol_c > 0 else np.nan, "ntrades": int(len(D))}


def update(max_days=None, verbose=True):
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["date"])
    have = set(old["date"].astype(str))
    last = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1))
    days = [d for d in pd.date_range(START, last) if d.strftime("%Y-%m-%d") not in have]
    if max_days:
        days = days[:max_days]
    t0, done = time.time(), 0
    for d in days:
        s = day_stats(d)
        if s is None:
            s = {"date": d.strftime("%Y-%m-%d"), "rr25": np.nan, "atm": np.nan, "pcr": np.nan, "ntrades": 0}
        pd.DataFrame([s]).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False, float_format="%.4g")
        done += 1
        if verbose and done % 100 == 0:
            print(f"  opcje: {done}/{len(days)} dni, {time.time() - t0:.0f}s", flush=True)
    if verbose:
        print(f"  [OK   ] Deribit (historia opcji): {done} nowych dni, {time.time() - t0:.0f}s", flush=True)
    return load()


def load():
    if not OUT.exists():
        return None
    d = pd.read_csv(OUT, parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    d["rr25"] = d["rr25"].where(d["rr25"].abs() <= 60)           # pojedyncze błędne wyceny (np. −420 pkt) usuwamy
    d["atm"] = d["atm"].where(d["atm"].between(5, 400))
    return d


if __name__ == "__main__":
    update(int(sys.argv[1]) if len(sys.argv) > 1 else None)
