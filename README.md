# BTC Predictor

System prognozowania **rozkładu** ceny Bitcoina (nie jednej liczby) z uczciwą weryfikacją na historii.
Panel na żywo: https://wywigacz.github.io/bitcoin-predictor/ (aktualizuje się sam co 6 godzin przez GitHub Actions).

## Uruchomienie lokalne

```
python run.py            # aktualizacja przyrostowa + analiza + panel (docs/index.html)
python run.py --full     # pełne pobranie historii od zera
```

Wymaga: pandas, numpy, scipy, scikit-learn, requests (`pip install -r requirements.txt`). Bez kluczy API.

## Co robi

| Moduł | Zadanie |
|---|---|
| `sources.py` | cena, on-chain (MVRV), górnicy, nastroje, makro, stablecoiny, funding, Deribit (DVOL, put/call), CFTC, Upbit (kimchi), Wikipedia w 11 językach |
| `world.py` | ~120 szeregów ze świata: giełdy i waluty 20+ krajów, towary, uwaga na Wikipedii (polityka, kryzysy), pogoda w regionach kopania, plamy słoneczne, trzęsienia ziemi, księżyc |
| `events.py`, `events_data.py` | baza wydarzeń: ręczne, kalendarz (FOMC, opcje), włamania (DefiLlama), automatycznie wykrywane szoki rynkowe; atrybucja ruchów i event study |
| `worldlog.py` | dziennik świata: co działo się każdego dnia od 2011 (Wikipedia Current events), podział na kategorie i ~20 grup słów kluczowych (wojna, banki, sankcje, wybory, klęski, krypto...) -> szeregi dla skanera |
| `news.py` | nagłówki prasowe o Bitcoinie z dni dużych ruchów (GDELT, z pamięcią podręczną i cierpliwym tempem zapytań) |
| `model.py` | 15 sygnałów (MVRV, 200W, Puell, makro...) -> wskaźnik złożony |
| `regularity.py` | Hurst (DFA), entropia permutacyjna, iloraz wariancji, grupowanie zmienności, indeks Hilla, EVT (GPD), łańcuch Markowa, filtr Kalmana, periodogram |
| `scan.py` | skaner setek hipotez z korektą na wielokrotne testy (FDR) + stabilność + test poza próbą |
| `ensemble.py` | zmienność (EWMA + GJR-GARCH + HAR), HMM, power law; modele: naive, wskaźnik złożony, ridge, analogi kNN, boosting kwantylowy, skaner świata; zespół z bramką istotności |
| `report.py`, `template_*.html`, `chart.js` | panel HTML (jeden plik), w tym interaktywny symulator regularnych wpłat (historia + scenariusze; edukacyjny, bez porad i bez transakcji) |

## Zasady rzetelności

- **Walk-forward**: każdy model uczy się wyłącznie na danych znanych w danym dniu; selekcja cech skanera odbywa się wewnątrz walk-forward.
- **Benchmark „bez zmian”** jest kalibrowany tak samo jak modele; porównanie w pinball loss, test Diebolda-Mariano.
- **Bramka istotności**: model wchodzi do prognozy dopiero przy p ≤ 0,10 (pełna waga p ≤ 0,03). Dziś żaden jej nie przechodzi, więc środek prognozy = cena bieżąca, a pasmo wynika ze zmienności.
- **Kalibracja pasma**: ~80% wyników ma mieścić się w P10-P90 (kontrolowane w teście).
- Audyt: losowe cechy i potasowany cel nie dają przewagi, a celowy wyciek przyszłości daje +34% (potok wykrywa prawdziwy sygnał).
- Dziennik prognoz (`data/forecast_log.csv`) porównuje każdą prognozę z realną ceną po upływie horyzontu.

## Ograniczenia

- Znane słabości: parametry GARCH/HAR i wagi mieszanki zmienności dobrane na całej próbie (łagodny wyciek, wpływ na skalę, nie kierunek); progi sygnałów pochodzą z opracowań.
- Brak darmowych źródeł: przepływy do ETF, Google Trends, globalne M2 (FRED bywa niedostępny, sygnał jest wtedy pomijany).
- Cykle halvingowe to 4 obserwacje. To nie jest rekomendacja inwestycyjna.

## Harmonogram lokalny (opcjonalnie)

```
python run.py --install-schedule   # zadanie Windows co 6 godzin
python run.py --remove-schedule
```
