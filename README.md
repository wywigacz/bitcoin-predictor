# BTC Predictor (lokalnie)

System prognozowania rozkładu ceny Bitcoina oparty na opracowaniach z folderu nadrzędnego
(wskaźniki on-chain, makro, cykle halvingowe). Działa lokalnie w Pythonie, bez kluczy API.

## Uruchomienie

```
python run.py            # aktualizacja przyrostowa + analiza + dashboard
python run.py --full     # pełne pobranie historii od zera
```

Wynik: `docs/index.html` (otwórz w przeglądarce), `output/latest.json` (dane maszynowe).
Dwuklik na `start.bat` robi to samo i otwiera panel.

## Samodzielna aktualizacja

```
python run.py --install-schedule   # zadanie Harmonogramu Windows, co 6 godzin
python run.py --remove-schedule
```

## Jak to działa

1. **Dane** (`sources.py`): cena (Bitstamp), MVRV i adresy (CoinMetrics), hash rate i przychody górników
   (Blockchain.com), Fear & Greed, Nasdaq/S&P/DXY/VIX/stopy/złoto (Yahoo), stablecoiny (DefiLlama),
   funding i open interest (OKX), dominacja BTC (CoinGecko), M2 (FRED, opcjonalnie).
   Tabela dzienna: `data/daily.csv`. Awaria jednego źródła nie zatrzymuje reszty.
2. **Sygnały** (`model.py`): 15 wskaźników -> ocena -1..+1, ważone wg wiarygodności z opracowań
   (wagi w `config.py`). Wynik: wskaźnik złożony (faza rynku).
3. **Prognoza**: rozkład ceny na 7/30/90/180 dni. Szerokość pasma = zmienność (EWMA) i empiryczne
   reszty z całej historii. Środek przesuwa regresja wskaźnika złożonego, ale **tylko** gdy test
   walk-forward (bez podglądania przyszłości) wykaże przewagę nad „bez zmian”.
4. **Samoocena** (`store.py`): każda prognoza trafia do `data/forecast_log.csv` i po upływie horyzontu
   jest porównywana z realną ceną (pokrycie pasma 80%, błąd, kierunek).

## Ograniczenia

- Brak darmowych źródeł: przepływy do ETF, Google Trends, globalne M2 (używane jest M2 USA z FRED,
  gdy FRED odpowiada). Sygnały bez danych są pomijane, a wagi przeliczane.
- Progi sygnałów pochodzą z opracowań i historii (4 cykle), więc test walk-forward ocenia głównie
  tę część, która uczy się z danych, a nie same progi.
- To nie jest rekomendacja inwestycyjna.
