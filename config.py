"""Ustawienia systemu prognozowania BTC. Wagi sygnałów można edytować tutaj."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
DAILY_CSV = DATA_DIR / "daily.csv"
LOG_CSV = DATA_DIR / "forecast_log.csv"
LATEST_JSON = OUT_DIR / "latest.json"
DASHBOARD_HTML = ROOT / "docs" / "index.html"

HORIZONS = [7, 30, 90, 180]          # dni
HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
NEXT_HALVING = "2028-04-17"          # szacunek

# Model
EWMA_ALPHA = 0.03                    # zmienność: EWMA kwadratów zwrotów (półokres ~23 dni)
SLOPE_RIDGE = 0.15                   # skurczenie nachylenia regresji
DRIFT_SHRINK = 0.0                   # część historycznego dryfu BTC (0 = nie ekstrapolujemy dawnych zysków)
MIN_TRAIN = 300                      # min. próbek do dopasowania
WF_STEP = 7                          # krok walk-forward (dni)
WF_START = 1500                      # indeks startu walk-forward (~4 lata historii)
MIN_IC = 0.05                        # próg korelacji OOS, by włączyć "przechył" sygnałów
MIN_SKILL = 0.0                      # próg poprawy MSE vs naiwna prognoza
MIN_WEIGHT = 2.5                     # minimalna suma wag dostępnych sygnałów

# Wagi według "poziomów wiarygodności" z opracowania (0 wyłącza sygnał)
WEIGHTS = {
    "mvrv": 1.0,        # MVRV = cena / Realized Price: najlepszy wskaźnik dna
    "ma200w": 1.0,      # cena vs 200-tygodniowa średnia
    "mayer": 0.4,       # cena vs 200-dniowa średnia
    "puell": 0.5,       # przychody górników vs średnia roczna
    "hashribbon": 0.4,  # kapitulacja górników (30d vs 60d hash rate)
    "fng": 0.3,         # Fear & Greed (kontrariańsko)
    "funding": 0.4,     # funding rate perpów (dźwignia)
    "stable": 0.3,      # wzrost podaży stablecoinów (30d)
    "m2": 0.5,          # wzrost M2 z opóźnieniem ~70 dni (jeśli FRED dostępny)
    "dxy": 0.2,         # zmiana indeksu dolara (30d)
    "ndx": 0.4,         # momentum Nasdaq (90d), BTC jako aktywo ryzykowne
    "vix": 0.3,         # VIX
    "rates": 0.3,       # zmiana krótkich stóp (13-tyg. bony, 180d)
    "pi": 0.2,          # Pi Cycle Top (tylko ostrzeżenie o szczycie)
    "halving": 0.6,     # faza cyklu halvingowego
}
