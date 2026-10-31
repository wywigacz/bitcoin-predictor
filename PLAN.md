# Plan: co jeszcze można zrobić, żeby Barometr lepiej przewidywał

Spisany 30 października 2026. Powstał tak: osiem niezależnych „perspektyw” (dźwignia, opcje, dane z łańcucha, uwaga i nastroje, rynki zakładów, metody kierunku, metody ryzyka, portfel oraz audyt naszego kodu) zgłosiło 52 pomysły; każdą grupę ocenił osobny sceptyk z dostępem do badań (dowody, szansa przejścia naszego protokołu, największe ryzyko); liczby, które o czymś przesądzały, policzyłem sam od nowa na naszych danych, a źródła danych sprawdziłem, naprawdę je pobierając. Pomysły, których nie dało się potwierdzić, są na końcu, z powodami.

## Odpowiedź wprost

**Kierunku ceny nie poprawimy większą ilością danych.** To nie jest wniosek z jednego testu, tylko z trzech niezależnych rund (skaner 8,5 tys. testów, 13 hipotez spisanych z góry, teraz 52 pomysły przepuszczone przez sceptyków). Jedyne, co ma dowody w badaniach, to słaby trend (nasza H7), a jego siła maleje z roku na rok. Doszła do tego ważna rzecz o nas samych: nasz próg dopuszczenia metody do prognozy (2% poprawy straty pinball) matematycznie wymaga przewidywalnego przesunięcia o 0,2 odchylenia, czyli R² około 4%. Sygnał wielkości H7 (0,13 odchylenia) daje tylko 0,84%. Czyli „szansa wzrostu 50%” jest po części cechą naszej miary, nie tylko rynku. Osobny tor dla kierunku (punkt 11) mógłby pokazać trafność 52–54%, ale udowodnienie tego poza próbą zajmie lata (moc testu na odciętych 2 latach: około 15%). Realistycznie: 50% zostanie na panelu jeszcze długo i to jest uczciwe.

**Zakres, ryzyko i prawdopodobieństwa da się poprawić, i to tanio.** Zmienność jest jedyną rzeczą, którą ten system naprawdę przewiduje (korelacja 0,50 między naszą prognozą a tym, co się stało tydzień później). Trzy rzeczy potwierdzone dziś trzykrotnie (agent, sceptyk, ja): pasma na 90–180 dni są za wąskie w spokojnych okresach, „dotknięcie poziomu” liczymy po zamknięciach dnia, choć panel obiecuje „choć raz po drodze”, i brakuje produktu „tydzień spokojny czy burzliwy”, który ma mierzalną skuteczność.

**Portfel wymaga uczciwszego opisu, zanim cokolwiek do niego dodamy.** Aktywny portfel bije regularne kupowanie tylko w 48% dwuletnich okien od 2014 roku (mediana 0,994). Płytszy dołek w 97% okien (średnio o 7,9 pkt proc.) wynika głównie z tego, że trzyma mniej bitcoina: stały udział 65% daje podobne obsunięcie. Przewaga „+47% nad DCA od 2018” zależy od wyboru okna trendu (28 dni: 1,47; 112 dni: 0,64). Strategia „ryzyko” sama przegrywa z DCA (0,78).

**Dwie nowe, wieloletnie historie danych, których nie mieliśmy, są dostępne za darmo:** pozycjonowanie na Binance od września 2020 (co 5 minut) i transakcje opcjami na Deribit od 2018 roku. Warto je dodać, ale jako sygnały ryzyka krachu i do kalibracji pasm, nie jako przepowiednie kierunku; dla kierunku spodziewamy się odrzucenia.

## Co naprawiłem już dziś (błędy znalezione przez audyt)

1. **Kolumna pozycji funduszy na CME (`cot_lev`)** zawierała w 88% wierszy liczby odsłon włoskiej Wikipedii (pozostałość pomieszania kolumn z 29 października). Odbudowana z nowego pobrania z CFTC, datowana na dzień publikacji (piątek, 3 dni po stanie z wtorku), ze strażnikiem zakresu.
2. **Niedokończona świeca dnia.** Bitstamp zwracał też bieżący, niezamknięty dzień, więc prognozy i dziennik na żywo liczyły się od ceny z środka dnia (horyzont „7 dni” był naprawdę około 7,6 dnia). Teraz pobieramy tylko zamknięte dni UTC, a panel podpisuje cenę datą zamknięcia.

## Stan wdrożenia

- 30.10.2026: punkty 1–3 wdrożone (pasma zależne od stanu zmienności na 7/30/90 dni, „Spokojnie czy burzliwie?”, dotknięcia po maksimach i minimach dnia).
- 30.10.2026: punkty 7–9 wdrożone (dźwignia Binance i hipotezy H15–H17, sędzia z Polymarket i Kalshi, historia opcji Deribit od 2018 dla H13). Wynik sędziego: rynek zakładów zgaduje dotknięcia lepiej niż nasz model (Brier 0,131 vs 0,160 w ostatnich 2 latach).
- 30.10.2026: punkty 4–6 wdrożone (krok dzienny dla 1/3/7 dni, poprawka HLN, „nierozstrzygnięte”, najmniejszy wykrywalny efekt, monitor i reguła emerytury H7/H10, H14; blok odporności portfela; dziennik wszystkich metod).

## Plan w kolejności (zysk × szansa, że przejdzie protokół) / nakład

Nakład: 1 = godziny, 3 = dzień–dwa, 5 = tydzień. „Szansa” to ocena sceptyka, że pomysł przejdzie nasz protokół (hipoteza z góry, test poza próbą, egzamin na 2 latach).

| # | Co | Zysk | Nakład | Szansa |
|---|---|---|---|---|
| 1 | Pasma zależne od stanu zmienności (wariancja H-dniowa wracająca do normy) + audyt kalibracji po stanach | zakres | 2 | 0,45–0,6 |
| 2 | „Tydzień spokojny czy burzliwy”, szansa dnia z ruchem ≥5% / ≥10%, szansa obsunięcia | nowy, sprawdzalny produkt | 2 | 0,7 |
| 3 | Dotknięcie liczone po maksimach i minimach dnia, z poprawką na próbkowanie | uczciwość + ryzyko | 2 | 0,55 |
| 4 | Statystyka we właściwym rozmiarze: dzienne kroki dla 1–7 dni, poprawka na małe próby, werdykt „nierozstrzygnięte”, minimalny wykrywalny efekt przy każdej bramce | wiarygodność | 2 | 0,8 |
| 5 | Uczciwość portfela: wszystkie daty startu, przedział ufności, „ślepy” portfel o tej samej ekspozycji, sigma bez zaglądania w przyszłość, tabela kruchości, 1/N obok Hedge | wiarygodność | 2–3 | 0,5–0,75 |
| 6 | Dziennik na przyszłość: kwantyle każdej metody co dzień, tylko dopisywany, z numerem wersji kodu | wiarygodność | 1 | 0,6 |
| 7 | Historia dźwigni z Binance (od 2020-09) jako sygnał ryzyka ogona; archiwum stawek za lewar zamiast zablokowanego Bybit | ryzyko | 3 | 0,2–0,3 |
| 8 | Zewnętrzny sędzia dla dotknięć: rozstrzygnięte rynki Polymarket (od 2024-02) i log Kalshi | wiarygodność | 2–3 | 0,4 |
| 9 | Historia opcji Deribit (od 2018): asymetria 25-delta, struktura terminowa; H13 rozstrzygalna teraz, nie w 2027 | ryzyko (opcje) | 4 | 0,15 (H13), 0,3 (rozkład z opcji) |
| 10 | Uczciwe powtórzenie wyścigu zmienności z danych 5-minutowych (HARQ, właściwy sędzia) | zakres (1–7 dni) | 3 | 0,4 |
| 11 | Wspólna seria testów „regresorów wariancji”: uwaga, odpływy z giełd, ton newsów (GDELT), udział opłat | zakres | 2 | 0,3 |
| 12 | Osobny tor dla kierunku: prawdopodobieństwo wzrostu liczone wprost i rozliczane wynikiem Briera | kierunek | 2–3 | 0,15 |
| 13 | Podatek (19%, rozliczenie roczne) i odsetki od gotówki w portfelu, wersja minimalna | wiarygodność portfela | 2 | 0,45 |

### 1. Pasma zależne od stanu zmienności

Dziś pasmo na H dni to zmienność dzienna × √H. Gdy jest spokojnie, zmienność zwykle wraca do normy, więc pasmo na 90–180 dni jest za wąskie; gdy jest burzliwie, za szerokie. Moje liczby (od 2015, pokrycie pasmem 80% w tercylu spokojnym / średnim / burzliwym): 30 dni: 67% / 77% / 86%; 90 dni: 52% / 74% / 85%; 180 dni: 48% / 73% / 82%. Wariancja H-dniowa wracająca do normy (parametr powrotu z naszego GJR-GARCH, okres połowiczny 23 dni) wyrównuje rozrzut do 1,21 / 1,25 / 1,26 na 180 dni.

Zastrzeżenia sceptyka, które uwzględnimy: poziom „normy” liczyć z okna kroczącego 2–3 lat (zmienność bitcoina maleje od dekady; w okresie egzaminu 2024–26 efekt jest już mały), horyzonty główne to 7 i 30 dni, 90 i 180 tylko opisowo; wdrożyć albo tę metodę, albo mnożniki pasma zależne od stanu (nie obie naraz, bo się dublują). Najpierw sam audyt: tabela „gdzie pasma zawodzą” (po stanie zmienności, stanie HMM, roku), bo to diagnostyka bez ryzyka.

Test z góry: „standaryzacja wariancją wracającą do normy obniża stratę pinball rozkładu bazowego na 7 i 30 dni (DM, HAC) i sprowadza pokrycie 80% do ±5 pkt proc. w każdym tercylu, w teście „jak na żywo” i w egzaminie”.

### 2. Tydzień spokojny czy burzliwy

Publikujemy obok uczciwego 50/50: (a) szansę, że zmienność w następnym tygodniu będzie powyżej mediany z roku; (b) szansę co najmniej jednego dnia z ruchem ≥5% i ≥10% w 7 i 30 dni (z już liczonych ścieżek); (c) szansę obsunięcia o 10/20% w 30 dni. Replikacje: korelacja 0,50; wynik Briera lepszy o 6,5% od klimatologii i o 3,6–5,1% od „będzie jak w zeszłym tygodniu”, także w egzaminie. Punktem odniesienia na panelu będzie ta trudniejsza wersja (persistencja).

### 3. Dotknięcie po maksimach i minimach

Panel pyta „czy choć raz dotknie poziomu po drodze”, a liczymy po zamknięciach dnia. Różnica: +10% w 30 dni zdarza się w 57,9% okien po maksimach dnia, a w 50,7% po zamknięciach; −10% w 7 dni: 18,6% wobec 11,7%. Poprawka: losować wspólnie trójki (zamknięcie, maksimum, minimum) lub użyć mostu Browna (Broadie–Glasserman–Kou 1997), z zabezpieczeniem przed pojedynczymi „knotami” jednej giełdy (sprawdzanie z 5-minutowymi danymi Binance). Jeśli nowy model nie utrzyma skuteczności w egzaminie, zostajemy przy zamknięciach i przeredagowujemy panel.

### 4. Statystyka we właściwym rozmiarze

Testy na 1 i 3 dni używają 579 z około 4050 dostępnych dni (krok tygodniowy dla wszystkich horyzontów). Egzamin na 90 i 180 dni opiera się na 91/79 tygodniowych punktach, czyli około 7 i 3 niezależnych obserwacjach, a wyświetla „nie zdany” jak pewnik. Zmiany: krok dzienny dla 1/3/7 dni (dopasowanie co tydzień, ocena co dzień, błędy HAC z opóźnieniem H+1), poprawka Harveya–Leybourne’a–Newbolda dla małych prób, werdykt „nierozstrzygnięte” gdy efektywna liczba obserwacji < 20, i przy każdej bramce zdanie „najmniejszy efekt, jaki ten test mógłby wykryć”, z jasną zasadą: test za słaby, by wykryć, nadal znaczy „nie używamy”. Do tego datowana reguła emerytury dla H7/H10 (zanik efektu w dwóch kolejnych rocznych sprawdzeniach).

### 5. Uczciwość portfela

Blok odporności w `sim.py` i na panelu: wyniki dla każdej daty startu od 2014 (okna 2 i 4 lata), przedział ufności z bootstrapu blokowego, sigma liczona tylko z przeszłości (dziś parametry zmienności są dopasowane na całej historii, także na okresie egzaminu), tabela kruchości okna trendu, 1/N obok Hedge (są praktycznie równe: 1,15 vs 1,14), monitor zaniku H7 i prerejestracja H14 = kontynuacja H7 na danych po 29.10.2026. Najważniejsze: porównanie ze „ślepym” portfelem o tej samej średniej ekspozycji w BTC, bo to on jest uczciwym punktem odniesienia dla obsunięć. Panel będzie mówił wprost: aktywny portfel to mniejsza ekspozycja w złych okresach, nie wyższy majątek.

### 6. Dziennik na przyszłość

Co dzień dopisujemy kwantyle każdej metody (nie tylko opublikowane), z numerem commitu z GitHub Actions; nic nie nadpisujemy. Każdy dzień bez tego to bezpowrotnie stracony dowód. Regułę wejścia metody do prognozy ustalić z poprawnym progiem (sceptyk pokazał, że przedział 6–14% na ogony przy 52 wierszach odrzucałby poprawny model w 63% przypadków).

### 7. Historia dźwigni z Binance

Archiwum `data.binance.vision` (sprawdzone): otwarte pozycje, stosunek długich do krótkich u największych traderów i u ogółu, przewaga kupujących nad sprzedającymi, co 5 minut od 1.09.2020 (około 2200 małych plików; trzymamy tylko dzienne agregaty); premia kontraktów wieczystych od 1.01.2020; stawki za lewar od 1.01.2020 (zastąpią zablokowany w chmurze Bybit i 36-dniowy OKX). Historii likwidacji nie ma.

Hipoteza z góry (ryzyko, nie kierunek): „otwarte pozycje / podaż BTC w górnym decylu (znane od D+2) → częstsze dotknięcie −10% w 30 dni i więcej wypadnięć poniżej P10 pasma” (jednostronnie, bootstrap blokowy, nie test dwumianowy, bo epizody są skupione). Ryzyko: udział Binance w rynku maleje, a od 2024 część wysokich pozycji to zabezpieczony arbitraż ETF, więc „dużo pozycji” znaczy dziś co innego niż w 2021. Kaskady likwidacji z 5-minutowych spadków pozycji tylko jako etykieta zdarzeń (dane praktyków pokazują raczej kontynuację spadku niż odbicie). Carry/baza (CME z Yahoo + premia Binance) później, też jako test ryzyka ogona; Deribit sprzed 2021 to tylko kwartalne, cienkie kontrakty.

### 8. Zewnętrzny sędzia dla dotknięć

Polymarket ma od lutego 2024 rozstrzygnięte miesięczne rynki „czy BTC dotknie X” z dzienną historią cen (rozliczane po 1-minutowych świecach Coinbase). To jedyny natychmiastowy, niezależny sprawdzian jedynego modelu, który ma skuteczność. Z tego komputera adres nie rozwiązuje się w DNS (filtr lokalny), z serwera GitHuba prawdopodobnie zadziała; trzeba to sprawdzić. Przy około 33 miesiącach to raport uczciwości („rynek mówi 19%, my 23%”), nie test zaliczony/niezaliczony. Kalshi (31 rozstrzygniętych rynków miesięcznych od sierpnia 2026, bez klucza) logować co 6 godzin i pokazywać obok własnej liczby, z zastrzeżeniem o spreadzie.

### 9. Historia opcji Deribit

Archiwum transakcji opcjami (sprawdzone: od stycznia 2018 ze zmiennością implikowaną przy każdej transakcji, gęste od 2019/2020; 550 wygasłych kontraktów terminowych od 2016). Jednorazowe pobranie to tysiące zapytań, kilka godzin lokalnie. Z tego: dzienna asymetria 25-delta, struktura terminowa, popyt netto. Dzięki temu H13 (już zarejestrowana) rozstrzyga się na około 2400 dniach zamiast czekać do 2027. Uczciwe oczekiwanie: odrzucenie (sceptyk sprawdził, że cytowana praca o bitcoinie bada wpływ popytu na ceny opcji, nie prognozę kursu; doktorat SMU 2026 widzi efekt, który słabnie od 2024). Dopiero potem rozkład z opcji (Breeden–Litzenberger) jako kandydat do pasm i dotknięć na 30 dni, oceniany względem obecnej mieszanki z DVOL, nie względem „bez zmian”.

### 10. Wyścig zmienności z danych 5-minutowych, powtórzony uczciwie

Nasz wcześniejszy test oceniał modele na szumnym celu (dzienny r²). Gdy sędzią jest prawdziwa zmienność z 5 minut, HAR-RV wygrywa o 8–9% (QLIKE 0,273 vs 0,296), a mieszanka 30/70 o 9%. Ale pokrycie pasm jednodniowych prawie się nie zmienia (0,840 vs 0,832). Dlatego kryterium przyjęcia to strata pinball gotowych pasm na 1/3/7 dni, a nie QLIKE. Wariant HARQ (z kwartycznością), dopasowania rozszerzane co rok, rok 2017 pominięty.

### 11. Regresory wariancji, jedna wspólna seria

Kandydaci (wszystkie bez klucza): skok uwagi na Wikipedii (mamy), odpływy z giełd i obrót względem rezerw (Coin Metrics `FlowOutExNtv`, nieużywane), ton newsów o bitcoinie z GDELT (sprawdzone: dzienna historia od 2017; endpoint liczby artykułów dziś nie odpowiedział), udział opłat w przychodach górników (mamy kolumny). Każdy jako hipoteza o wariancji, przyjmowany tylko, gdy poprawia QLIKE na odłożonych danych. Oczekiwany zysk mały: uwaga podąża za ruchem ceny, który model zmienności już widzi.

### 12. Osobny tor dla kierunku

Prawdopodobieństwo wzrostu liczone wprost (probit z sygnałów, które przeszły test, z kalibracją izotoniczną), oceniane wynikiem Briera względem 0,5 i testem znaku z błędami HAC, codziennie, nie co tydzień. Dopiero po punkcie 4. Uczciwe oczekiwanie: 52–54% trafień, moc testu na odciętych 2 latach około 15%, więc wynik „nierozstrzygnięte” przez lata; na panelu zostaje 50%, ale z dopiskiem, ile wynosi trafność na żywo. Rodzina trendu (stan względem średniej 200-dniowej) jako jedna hipoteza główna, bo to ten sam czynnik co H7.

### 13. Podatek i odsetki w portfelu

Wersja minimalna, dopiero po punkcie 5: 19% od rocznego dochodu z walut wirtualnych (koszty rozliczane kasowo, nadwyżka przechodzi na następny rok, zgodnie z art. 30b ust. 1a ustawy o PIT; nie FIFO), odsetki od gotówki po stopie NBP (BIS publikuje ją bez klucza) pomniejszone o podatek Belki. Pokazywać obok wyniku przed podatkiem. Wstępny rachunek agenta: przewaga 1,14 nad DCA spada do około 1,10–1,115.

## Nagrywać od dziś (historii nie ma, więc test będzie możliwy za rok lub dwa)

- Surowe wiersze pełnego „uśmiechu” opcji z Deribit przy każdej aktualizacji (już pobieramy podsumowanie, trzeba tylko zapisywać): ocena za lata.
- Kalshi: miesięczne rynki „dotknie / nie dotknie” (punkt 8).
- Podaż krótko- i długoterminowych posiadaczy (BGeometrics, bez klucza, ale tylko 4 lata wstecz, okno się przesuwa i historia przepada; licencja tylko do użytku własnego, więc bez publikowania na panelu bez wyjaśnienia).
- Premia Strategy do wartości posiadanych bitcoinów (mNAV) z danych, które mamy; noga GBTC nie żyje od stycznia 2024.
- Daty wejścia w życie i terminy konsultacji przepisów o krypto z Federal Register (bez klucza): nie do prognozy, ale do listy nadchodzących wydarzeń na osi czasu.

## Nie warto (sprawdzone lub ocenione, z powodem)

- Pozycje na kredyt z Bitfinex od 2016: brak dowodów, giełda straciła znaczenie po 2019, więc reguła „ten sam znak w obu połowach historii” jest prawie nie do spełnienia.
- Rankingi aplikacji (Coinbase w App Store), StockTwits, głosy CoinGecko, gwiazdki GitHub: ta sama rodzina „uwaga i nastrój”, która u nas przegrała; zdarzenie szczytowe raz na cykl, dowód za dekadę.
- Google Trends sklejane z 90-dniowych okien od 2011: sklejanie mnoży błąd próbkowania; badania mówią, że uwaga podąża za ceną.
- Przepływy górników, dni-monety zniszczone, fale HODL: lore z wykresów, brak testów poza próbą, 4-letnie okno danych daje 1–3 niezależne obserwacje.
- Dni publikacji CPI/NFP jako mnożnik zmienności: efekt dla bitcoina mały i zależny od epoki (Fed NY 2023: bitcoin nie reaguje systematycznie na dane makro), wymaga klucza do FRED.
- Mempool i zatory jako regresor zmienności: od 2023 opłaty napędzają ordinals i runes niezależnie od ceny.
- Czwarta strategia portfela na implikowanej zmienności (DVOL): sigma już zawiera 30% DVOL; w kryptowalutach efekt dźwigni jest odwrócony (Baur i Dimpfl 2018); 3,6 roku danych nie odróżni jej od obecnej.
- Momentum zależne od reżimu + meta-labeling: potrzebna różnica korelacji większa niż cały efekt momentum.
- Przepływ zleceń i likwidacje jako sygnał kierunku: działają w sekundach i minutach; a odbicie po kaskadzie likwidacji dane praktyków wręcz przeczą.
- Teksty FOMC, zgłoszenia do SEC jako sygnał: brak mocy prognostycznej; Federal Register tylko jako kalendarz.
- Dziennik rewizji danych on-chain: dziś żaden sygnał na żywo nie zależy od danych z łańcucha; wrócić, gdy jakiś wejdzie (i wtedy logować skrót całej kolumny, bo Coin Metrics przepisuje całą historię po znalezieniu nowych adresów giełd).

## Największy zakład

Punkty 1–3 razem: pasma, które wiedzą, w jakim stanie jest rynek, dotknięcia liczone tak, jak obiecuje panel, i „tydzień spokojny czy burzliwy”. To jedyne obszary, w których dziś trzy niezależne rachunki pokazały ten sam, mierzalny zysk, i wszystkie trzy są tanie.

## Kluczowe źródła sprawdzone dziś

- Diebold, Hickman, Inoue, Schuermann (1998), Converting 1-day volatility to h-day volatility: scaling by √h is worse than you think, Risk 11.
- Broadie, Glasserman, Kou (1997), A continuity correction for discrete barrier options, Mathematical Finance 7(4).
- Christoffersen (1998), Evaluating interval forecasts, International Economic Review 39(4).
- Harvey, Leybourne, Newbold (1997), Testing the equality of prediction mean squared errors, International Journal of Forecasting 13.
- Patton (2011), Volatility forecast comparison using imperfect volatility proxies, Journal of Econometrics 160; Bollerslev, Patton, Quaedvlieg (2016), Exploiting the errors, Journal of Econometrics 192.
- Schmeling, Schrimpf, Todorov (2023), Crypto carry, BIS Working Paper 1087: wysoka premia kontraktów zapowiada krachy, nie niższą średnią stopę.
- Cederburg, O’Doherty, Wang, Yan (2020), On the performance of volatility-managed portfolios, JFE 138: w czasie rzeczywistym zarządzanie zmiennością przegrywa w 72 ze 103 strategii.
- Baur, Dimpfl (2018), Asymmetric volatility in cryptocurrencies, Economics Letters 173.
- Benigno, Rosa (2023), The Bitcoin-Macro Disconnect, NY Fed Staff Report 1052.
- Page, Clemen (2013), Do prediction markets produce well-calibrated probability forecasts?, Economic Journal 123.
- Hoang, Baur (2020), Forecasting bitcoin volatility: evidence from the options market, Journal of Futures Markets.
- Źródła danych sprawdzone pobraniem: data.binance.vision (metrics od 2020-09-01, premiumIndexKlines i fundingRate od 2020-01), history.deribit.com (transakcje opcjami od 2018-01, 550 wygasłych kontraktów), api.elections.kalshi.com (KXBTCMAXMON/MINMON), api.gdeltproject.org (timelinetone od 2017), api.bitcoin-data.com (okno 4 lat), api.blockchain.info (mempool od 2016-06), publicreporting.cftc.gov.
