"""Historia dźwigni i pozycjonowania z publicznego archiwum Binance (data.binance.vision, kontrakty USDT-M BTCUSDT, bez klucza).

  metrics (pliki dzienne od 2020-09-01, co 5 minut): otwarte pozycje (BTC), stosunek długich do krótkich u największych traderów
     (pozycje) i u ogółu kont, stosunek wolumenu kupujących do sprzedających (taker),
  premiumIndexKlines 1d (od 2020-01): premia kontraktu wieczystego nad ceną indeksu (baza),
  fundingRate (od 2020-01): stawka za lewar co 8 godzin.
Zapisujemy tylko dzienne agregaty: data/lev.csv. Dzień D jest publikowany po jego końcu, więc w testach używamy go od D+1."""
import io
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import config as C

OUT = C.DATA_DIR / "lev.csv"
B = "https://data.binance.vision/data/futures/um"
UA = {"User-Agent": "Mozilla/5.0 (btc-predictor research)"}
M_START = pd.Timestamp("2020-09-01")
P_START = pd.Timestamp("2020-01-01")


def _zip_csv(url):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                raw = z.read(z.namelist()[0]).decode(errors="ignore")
                first = raw.split("\n", 1)[0]
                header = 0 if any(ch.isalpha() for ch in first) else None
                return pd.read_csv(io.StringIO(raw), header=header)
        except (requests.RequestException, zipfile.BadZipFile):
            pass
        time.sleep(2 * (i + 1))
    return None


def _metrics_day(d):
    df = _zip_csv(f"{B}/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d:%Y-%m-%d}.zip")
    if df is None or df.empty:
        return None
    g = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(dtype=float)
    return {"date": d.strftime("%Y-%m-%d"), "oi_btc": float(g("sum_open_interest").iloc[-1]),
            "oi_usd": float(g("sum_open_interest_value").iloc[-1]),
            "top_ls_pos": float(g("sum_toptrader_long_short_ratio").mean()), "top_ls_acc": float(g("count_toptrader_long_short_ratio").mean()),
            "all_ls": float(g("count_long_short_ratio").mean()), "taker_ratio": float(g("sum_taker_long_short_vol_ratio").mean())}


def _monthly(kind, m):
    if kind == "premium":
        df = _zip_csv(f"{B}/monthly/premiumIndexKlines/BTCUSDT/1d/BTCUSDT-1d-{m:%Y-%m}.zip")
        if df is None:
            return {}
        ts = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        ts = np.where(ts > 1e14, ts // 1000, ts)
        return {pd.Timestamp(int(t), unit="ms").strftime("%Y-%m-%d"): float(c) for t, c in zip(ts, pd.to_numeric(df.iloc[:, 4], errors="coerce"))}
    df = _zip_csv(f"{B}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{m:%Y-%m}.zip")
    if df is None:
        return {}
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    t = np.where(t > 1e14, t // 1000, t)
    s = pd.Series(pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(), index=pd.to_datetime(t, unit="ms").normalize())
    return {k.strftime("%Y-%m-%d"): float(v) for k, v in s.groupby(level=0).mean().items()}


def update(verbose=True):
    old = pd.read_csv(OUT, index_col=0) if OUT.exists() else pd.DataFrame()
    t0 = time.time()
    yday = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    have = set(old.index[old["oi_btc"].notna()]) if "oi_btc" in old else set()
    rows = []
    for d in pd.date_range(M_START, yday):
        if d.strftime("%Y-%m-%d") in have:
            continue
        s = _metrics_day(d)
        if s:
            rows.append(s)
            if len(rows) % 100 == 0:                             # zapis co 100 dni: przerwanie nie gubi pobranego
                part = pd.DataFrame(rows).set_index("date")
                (part.combine_first(old) if len(old) else part).sort_index().rename_axis("date").to_csv(OUT, float_format="%.8g")
        time.sleep(0.05)
    new = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    # baza i stawka za lewar: pliki miesięczne (bieżący miesiąc jeszcze nie istnieje, więc dociągamy od 2 mies. wstecz)
    have_p = set(old.index[old["premium"].notna()]) if "premium" in old else set()
    prem, fund = {}, {}
    for m in pd.date_range(P_START, yday, freq="MS"):
        days = pd.date_range(m, m + pd.offsets.MonthEnd(0))
        if sum(d.strftime("%Y-%m-%d") in have_p for d in days) >= len(days) - 2 and m < yday - pd.DateOffset(months=2):
            continue
        prem.update(_monthly("premium", m))
        fund.update(_monthly("funding", m))
    pf = pd.DataFrame({"premium": pd.Series(prem), "funding": pd.Series(fund)})
    merged = new.combine_first(old) if len(old) else new
    merged = pf.combine_first(merged) if len(merged) else pf
    merged = merged.sort_index()
    merged.index.name = "date"
    if len(merged):
        merged.to_csv(OUT, float_format="%.8g")
    if verbose:
        print(f"  [OK   ] Binance (dźwignia, baza, stawki): {len(new)} nowych dni pozycji, {time.time() - t0:.0f}s", flush=True)
    return merged


def load():
    return pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else None


if __name__ == "__main__":
    update()
