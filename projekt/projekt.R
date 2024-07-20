# Prognozowanie i symulacje, dr hab. Elżbieta Kubińska, WSB-NLU, 2024
# Autor: Andrzej Kocielski, 24-06-2024

# Przeprowadzenie analizy szeregów czasowych i prognozowania przyszłych wartości na podstawie modelowania ARIMA

# Zbierz dane: 
#   - Wejdź na stronę stooq.pl i Bank danych lokalnych.
#   - Pobierz szereg czasowy z co najmniej 150-200 pomiarami.
# Przedstaw graficznie dane:
#   - Wyznacz ACF i PACF:
#   - Zastosuj auto.arima():
#   - Użyj funkcji auto.arima() z pakietu forecast, aby dopasować model ARIMA do danych.
# Zrób prognozę na 10 lat do przodu:
#   - Zastosuj funkcję simulate() z pakietu forecast do prognozowania.
#   - Użyj funkcji pakietu fanplot, aby wykonać prognozę na 10 lat do przodu, wraz z przedziałami ufności opartego na metodzie Monte Carlo.
# Model z różnicowaniem logarytmów:
#   - Dokonaj różnicowania logarytmów danych, aby usunąć trend i stabilizować wariację.
#   - Dopasuj model za pomocą auto.arima() na zróżnicowanych logarytmach danych.
# Model z wyodrębnionym trendem deterministycznym:
#   - Wyodrębnij trend deterministyczny, rozważając funkcje liniową, kwadratową i wykładniczą.
#   - Dopasuj model ARIMA do reszt za pomocą auto.arima().
# Ocena modelu:
#   - Porównaj modele za pomocą kryteriów informacyjnych (AIC, BIC) oraz reszt.


#rm(list=ls()) 
library(tseries)
library(forecast)
library(ggplot2)
library(fanplot)


##### 1. Wczytanie danych i przygotowanie środowiska

# Tygodniowe notowania indeksu WIG od 30-05-2004 do 02-06-2024
# Źródło: https://stooq.pl/q/d/?s=wig&c=0&d1=20040528&d2=20240528&o=1111111&i=w

#setwd("/home/ak/Desktop/_moje/_coding/WSB/05-prognozowanie-symulacja/projekt/")
dane <- read.csv("wig_w.csv")

# Przekształcenie kolumny dat na format daty
dane$Data <- as.Date(dane$Data, format="%Y-%m-%d")
# Konwersja typu wartoście indeksu na 'numeric'
dane$Zamkniecie <- as.numeric(dane$Zamkniecie)

# Podstawowe parametry zbioru danych
summary(dane)
sd(dane$Zamkniecie)


##### 2. Sprawdzenie czy dane nadają się do modelu ARIMA

# graficzna prezentacja danych - wykres zmian indeksu
ggplot(dane, aes(x=Data, y=Zamkniecie)) +
  geom_line() +
  labs(title="Wartości zamknięcia indeksu WIG", x="Data", y="Zamknięcie")
# Widzimy dużą zmienność względem średniej jak i wariancji (przypomina błądzenie losowe), typowe dla notowań giełdowych


# test Augmented Dickey-Fuller (ADF), sprawdzający stacjonarność danych (założenie ARIMA); H0: szereg niestacjonarny jeżeli p-value > 0.05
adf.test(dane$Zamkniecie) 
# Wniosek: p-value > 0,05, czyli nie ma podstaw do odrzucenia H0, tj. szereg niestacjonarny -> wymaga dopasowania


# Autokorelacja (ACF) - jak są ze sobą skorelowane wartości opóźnione (przeszłe): 
acf(dane$Zamkniecie, main="ACF Ceny Zamknięcia") 

# Widzimy silne dodatnie korelacje przy małych opóźnieniach - wartości szeregu czasowego 
#  mają tendencję do bycia podobnymi do wartości poprzednich.
# Wartości ACF opadają stopniowo - wskazuje na obecność trendu w danych. 
# Z wykresu odczytuje obecność trendu lub sezonowości -> wymaga dopasowania

# Autokorelacja częściowa (PACF)
pacf(dane$Zamkniecie, main="PACF Ceny Zamknięcia")
# Interpretacja PACF - przy eliminacji wpływu obserwacji pośrednich:
# Widzimy słabe naprzemiennie zmieniające się (dodatnie-ujemne) korelacje dla rosnących lagów.
# Może to o świadczyć o stacjonarności szeregu po transformacji, np. po różnicowaniu. 
# Nieznacznie większe wartości dla lag 2, 3, 15 oraz 23 sugeruje, że model AR być odpowiedni dla tych opóźnień.

# Sprawdzenie parametrów ARIMA dla naszych danych
summary(model_surowy <- arima(dane$Zamkniecie))
# AIC = 22690.16; RMSE = 12527.51

# Diagnostyka reszt (czy spełnia założania iid)
tsdisplay(residuals(model_surowy), main="Reszty Modelu ARIMA")
checkresiduals(model_surowy)
# wykres rozkładu reszt w umiarkowanym tylko stopniu przypomina normalny, tj. nie spełnia iid


##### Dekompozycja modelu

szereg <- dane$Zamkniecie
szereg_ts <- ts(szereg, start = c(2004, 6), frequency = 52)
dekompozycja <- decompose(szereg_ts)
plot(dekompozycja)

# Wyświetlenie komponentów
par(mfrow = c(3, 1))

trend <- dekompozycja$trend
plot(trend, main = "Trend")

sezonowosc <- dekompozycja$seasonal
plot(sezonowosc, main = "Sezonowość")

reszty <- dekompozycja$random
plot(reszty, main = "Reszty")
par(mfrow=c(1,1))

# Wniosek: szereg należy poddać wybiciu trendu i usunięciu autokorelacji


############## Ręczne dostosowanie szeregu 

##### 3. Stabilizacja (wybicie trendu) po przez różnicowanie

# analiza różnic między kolejnymi obserwacjami w szeregu czasowym
y1 <- diff(dane$Zamkniecie, lag=1, differences=1) #diff(x, lag = 1, differences = 1) 
cat("średnia modelu y1:", mean(y1), ", odchylenie standardowe:", sd(y1))
plot(y1, type="l", main="różnicowanie lag=1")
adf.test(y1) # p-value < 0.05 -> sugeruje na szereg niestacjonarny, czyli trend został usunięty

y2 <- diff(dane$Zamkniecie, lag=1, differences=4) # wielokrotne różnicowanie
plot(y2, type="l", main="różnicowanie wielokrotne")
cat("średnia modelu y2:", mean(y2), ", odchylenie standardowe:", sd(y2))
adf.test(y2) # p-value < 0.05, niewielka poprawa względem diff=1

y3 <- diff(log(dane$Zamkniecie), lag=1, differences=1) # logarytmowanie
plot(y3, type="l", main="różnicowanie log(dane)")
cat("średnia modelu y3:", mean(y3), ", odchylenie standardowe:", sd(y3))
adf.test(y3) # p-value < 0.05, ale model y2 wydaje się lepszy

# Diagnostyka reszt dla wybranego modelu po dostosowaniu szeregu
tsdisplay(residuals(arima(y2)), main="Reszty modelu y2 (diff=3, lag=1)")

# Wniosek: nawet jednokrotne różnicowanie wyraźnie stabilizuje szereg,
# Na podstawie ACF i PACF, wydaje się, iż odpowiednim będzie AR(2) oraz MA(2)


##### 4. Analiza odwrotności różnicowania (a więc do postaci szeregu pierwotnego - integrowanie)
z1 <- diffinv(dane$Zamkniecie, lag=1, differences=1) # diff(log(dane)) pokazuje zmianę procentową
adf.test(z1)
cat("średnia modelu z1:", mean(z1), ", odchylenie standardowe:", sd(z1))
tsdisplay(residuals(arima(z1)), main="Integrowanie Modelu z1 (diffinv lag 1)")

z2 <- diffinv(log(dane$Zamkniecie), lag=1, differences=1) # logartmowane wartości
tsdisplay(residuals(arima(z1)), main="Integrowanie Modelu z2 (diffinv(log)) lag 1)")
adf.test(z2)

# Wniosek zastosowanie odwrotności różnicowania wyodrębnia trend - taki model nie nadaje się do ARIMA


##### 5. Analiza z wyodrębnionym trendem deterministycznym 

# Trend liniowy
dane$Trend_lin <- 1:nrow(dane)
x1 <- lm(Zamkniecie ~ dane$Trend_lin, data=dane)
summary(x1)
x1_residuals <- residuals(x1)
checkresiduals(x1)
cat("średnia modelu x1_residuals:", mean(x1_residuals), ", odchylenie standardowe:", sd(x1_residuals))
adf.test(x1_residuals) # p-value > 0.05 -> sugeruje na szereg niestacjonarny, czyli bez poprawy :/
Box.test(x1_residuals, type="Ljung-Box") # p-value < 0.05, czyli odrzucamy H0 o brak autokorelacji reszt

# Trend kwadratowy 
dane$Trend_kw <- 1:nrow(dane)
dane$Trend_kw <- (dane$Trend_lin)^2
print(dane$Trend_kw)
x2 <- lm(Zamkniecie~dane$Trend_kw, data=dane)
summary(x2)
checkresiduals(x2)
x2_residuals <- residuals(x2)
adf.test(x2_residuals) # p-value > 0.05 -> jest autokrelacja
Box.test(x2_residuals, type="Ljung-Box") # p-value < 0.05, czyli jest autokorelacja

# trend logarytmiczny 
dane$Trend_log <- 1:nrow(dane)
x3 <- lm(Zamkniecie ~ log(dane$Trend_log), data=dane)
summary(x3)
checkresiduals(x3)
x3_residuals <- residuals(x3)
adf.test(x3_residuals) # p-value > 0.05 -> jest autokrelacja
Box.test(x3_residuals, type="Ljung-Box") # p-value < 0.05, czyli jest autokorelacja

# Wniosek: istniejący w szergu trend nie jest podobny do trendu liniowego, kwadratowego ani logarytmicznego


##### 6. Symulacja - ręczny dobór parametrów do modelu ARIMA

# Stabilizacja wariancji - transformacja Boxa-Coxa
BoxCox(dane$Zamkniecie, lambda="auto")
par(mfrow=c(2,1))
plot(dane$Zamkniecie, main = "Orginalne dane", 
     xlab="Nr. obserwacji", ylab="WIG")
grid()
plot(BoxCox(dane$Zamkniecie, lambda="auto"), main="Po transformacji Box-Cox", 
     xlab="Nr. obserwacji", ylab="WIG")
grid()


##### 7. Porównanie kilku modeli ARIMA oraz ocena dopasowania (AIC) 

# Wartości parametrów dobrane metodą prób i błędów

# Symulacja dla AR(1), I(1) MA(2); 
model112 <- arima.sim(n = 200, model = list(order = c(1, 1, 2), # środkowa liczba dotyczy rzędu integrowania (I)
                                    ar = c(0.5), # jeden parametr phi, bo zadaliśmy AR(1)
                                    ma = c(0.4, -0.5))) # dwa parametry theta, bo zadaliśmy MA(2)
par(mfrow=c(1,1))
autoplot(print(model112), main="Symulowane dane ARIMA(1, 1, 2)")
tsdisplay(residuals(arima(model112)), main="Symulowane dane ARIMA(1, 1, 2)")
model112 <- arima(dane$Zamkniecie, order = c(1, 1, 2)) 
summary(model112) # AIC = 17950.17

model201 <- arima.sim(n=200, model=list(order = c(2, 0, 1), ar=c(.8,-.4), ma=.2))
autoplot(print(model201), main="Symulowane dane ARIMA(2, 0, 1)")
tsdisplay(residuals(arima(model201)), main="Symulowane dane ARIMA(2, 0, 1)")
model201 <- arima(dane$Zamkniecie, order = c(2, 0, 1)) 
summary(model201) # AIC = 17978.33

model011 <- arima.sim(n=200, model=list(order = c(0, 1, 1), ma=.2))
autoplot(print(model011), main="Symulowane dane ARIMA(0, 1, 1)")
tsdisplay(residuals(arima(model011)), main="Symulowane dane ARIMA(0, 1, 1)")
model011 <- arima(dane$Zamkniecie, order = c(0, 1, 1)) 
summary(model011) # AIC = 17952.93


model112_2 <- arima(diff(dane$Zamkniecie), order = c(1, 1, 2)) # różnicowany
summary(model112_2) # AIC = 17943.17

model112_3 <- arima(log(dane$Zamkniecie), order = c(1, 1, 2)) # logarytmowany
summary(model112_3) # AIC = -4545.95

# sprawdzenie reszt dla ostatniego modelu
model112_3_reszty <- residuals(model112_3)
Box.test(model112_3_reszty, type="Ljung-Box") # p-value > 0.05 (brak korelacji reszt)
hist(model112_3_reszty, main="histogram")
qqnorm(model112_3_reszty, main="wykres kwantylowy")
qqline(model112_3_reszty)
shapiro.test(model112_3_reszty)

# Wniosek: model112_3 ma najniższą wartość AIB, ergo jest najbardziej dopasowany


##### 8. Model ARIMA za pomocą auto.arima()

# Automatyczne dopasowanie modelu ARIMA
model_auto <- auto.arima(dane$Zamkniecie)
summary(model_auto) # ARIMA(0,1,2); AIC = 17949.23

model_auto_reszty <- residuals(model_auto)
Box.test(model_auto_reszty, type="Ljung-Box") # p-value > 0.05 (brak korelacji reszt)
hist(model_auto_reszty, main="histogram")
qqnorm(model_auto_reszty, main="wykres kwantylowy")
qqline(model_auto_reszty)
shapiro.test(model_auto_reszty)


##### 9. Porównanie modeli za pomocą AIC i BIC
# model112_3 - najlepszy ręcznie dopasowany
# model_auto - automatycznie dopasowany

# analiza korelacji
tsdisplay(residuals(model112_3), main="Reszty modelu dopasowanego ręcznie")
tsdisplay(residuals(model_auto), main="Reszty modelu dopasowanego automatycznie")

# AIC oraz BIC
model112_3_aic <- AIC(model112_3)
model112_3_bic <- BIC(model112_3)
cat("Model dopasowany ręcznie AIC:", model112_3_aic, "BIC:", model112_3_bic, "\n")

model_auto_aic <- AIC(model_auto)
model_auto_bic <- BIC(model_auto)
cat("Model dopasowany automatycznie AIC:", model_auto_aic, "BIC:", model_auto_bic, "\n")

# analiza reszt
checkresiduals(model112_3)
checkresiduals(model_auto)

# Wniosek, wartość AIB dla modelu ręcznie jest dużo niższa - lepsze dopasowanie, 
# ale wykresy reszt są względnie podobne


##### 10. Prognoza przyszłych obserwacji

# Usuwamy kolumny
dane <- subset(dane, select = -c(Data, Wolumen))

x_lat <- 1
forecast_horizon <- x_lat * 52 # wyrażony w tygodniach, tj. liczba przyszłych obserwacjach

# Prognoza i wykres
forecast_data <- forecast(model_auto, h=forecast_horizon)
autoplot(forecast_data)

# Wykonanie prognozy funkcją fan() z przedziałami ufności
powtorzenia = forecast_horizon
simulations <- matrix(nrow=forecast_horizon, ncol=powtorzenia) # wiersz to prognoza dla kolejnego tygodnia
for (i in 1:powtorzenia) { simulations[,i] <- simulate(model_auto, nsim=forecast_horizon) }
plot(dane$Zamkniecie, type="l", xlim=c(900, nrow(dane)+forecast_horizon+50), ylim=c(40000,120000), main="Prognoza")
start = nrow(dane) # początek prognozy na osi odciętych
anchor = dane$Zamkniecie[start] # początek prognozy na osi rzędnych
fan(simulations, start=start, anchor=anchor, type="interval")#,  probs=seq(5, 95, 5), ln=c(50, 80))



##### Ekstra: Analiza szeregu przy pomocy modelu GARCH (niestacjonarność względem średniej jak i wariancji)

library(rugarch)

# model GARCH 1. (parametry dobrane metodą prób i błędów)
# https://rdrr.io/rforge/rgarch/man/ugarchspec-methods.html
garch_param <- ugarchspec(variance.model = list(model="sGARCH",
                                           garchOrder=c(1,1)),
                                           mean.model=list(armaOrder=c(1,1)),
                                           distribution.model="std")
garch_param

# https://rdrr.io/rforge/rgarch/man/ugarchfit-methods.html
modelGarch <- ugarchfit(spec=garch_param, data=dane$Zamkniecie)
modelGarch

plot(modelGarch@fit$residuals, type="l")
plot(modelGarch, which=10) # lista wykresów: plot(modelGarch)

# prognoza
modelGarcg.pred <- ugarchboot(modelGarch, n.ahead = forecast_horizon, method = c("Partial", "Full")[1])
print(modelGarcg.pred)
plot(modelGarcg.pred, which=2)


# model GARCH 2. (dla porównania)
garch_param2 <- ugarchspec(variance.model = list(model="sGARCH",
                                            garchOrder=c(1,1)),
                                            mean.model=list(armaOrder=c(0,0)),
                                            distribution.model="std")
modelGarch2 <- ugarchfit(spec=garch_param2, data=dane$Zamkniecie)
modelGarch2

# Model GARCH wykazuje lepsze dopasowanie niż model ARIMA:
# AIC_garch = 17082 < AIC_autoarima = 17949))


##### End