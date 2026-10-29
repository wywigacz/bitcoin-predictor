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
| `world.py` | ~250 szeregów ze świata: giełdy i waluty 20+ krajów, towary i energia (gaz TTF, węgiel, benzyna), spółki wydobywcze BTC, uwaga na Wikipedii (~130 haseł: ludzie, instytucje, bańki, klimat, technologie), pogoda w regionach kopania, plamy słoneczne, trzęsienia ziemi, księżyc, kalendarz i „dzikie karty” (retrogradacja Merkurego liczona z orbit Keplera, piątek 13., mundial/igrzyska/Super Bowl, ramadan, Złoty Tydzień, Święto Dziękczynienia, zmiana czasu) |
| `extra.py` | płynność dolara (NY Fed: reverse repo, SOFR, EFFR, SOMA; Skarb USA: TGA; płynność netto), pozycje funduszy w CFTC na S&P, Nasdaq, 10-latkach, jenie, euro, VIX, złocie i ropie, DefiLlama (TVL, DEX, USDT, USDC), Lightning (mempool.space), CoinMetrics (BTC na giełdach, napływy), pogoda kosmiczna GFZ (Kp, Ap, F10.7), Hacker News (miesięczne wzmianki, przesunięte o miesiąc, by nie podglądać przyszłości) |
| `whales.py` | duzi gracze: dzienne przepływy do ETF na bitcoina (TFTC, CC BY 4.0), transakcje Strategy (Bitbo, zmiana liczona z różnic stanu), stawka finansowania Hyperliquid oraz zdjęcia pozycji 150 najlepszych i 150 najgorszych traderów Hyperliquid co 6 godzin (`data/hl_snapshots.csv`, nagrywane od 2026-10-29); salda portfeli państw z mempool.space: USA (adres Silk Road), Niemcy (7 adresów BKA, suma 49 859 BTC zgodna z raportami), Salwador (oficjalna lista adresów z bitcoin.gob.sv); historia transakcji w `data/entities_cache.json` |
| `touch.py` | szansa dotknięcia poziomów ceny w 7/30/90 dni: symulacja 20 000 ścieżek z historycznych standaryzowanych ruchów i zmienności wracającej do mediany; test wstecz od 2016 vs historyczna częstość liczona tylko z przeszłości (Brier) |
| `confirm.py` | osobny test potwierdzający hipotez spisanych z góry 2026-10-29: partia 1 (H1–H6, duzi gracze, Bonferroni 0,05/6) i partia 2 (H7–H10, trend 1–4 tygodni, uwaga, połączenie sygnałów, 0,05/4); Spearman, n/horyzont, jednostronnie, zgodny znak w obu połowach; potwierdzone sygnały tworzą metodę „conf” w zespole (ta sama bramka istotności co reszta) |
| `store.py` | strażnik danych: usuwa wartości niemożliwe (np. cena Coinbase poza 0,8–1,25 ceny Bitstamp, DVOL poza 10–400), po naprawie pomieszanych kolumn w październiku 2026 |
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
