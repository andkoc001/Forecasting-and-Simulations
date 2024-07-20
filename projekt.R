# Prognozowanie i symulacje: 
# Przeprowadzenie analizy szeregów czasowych i prognozowania przyszłych wartości na podstawie modelowania ARIMA
# Wykładowca: dr hab. Elżbieta Kubińska, WSB-NLU, 2024
# Autor: Andrzej Kocielski, 22-06-2024


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

# graficzna prezentacja danych - wykres zmian indeksu
ggplot(dane, aes(x=Data, y=Zamkniecie)) +
  geom_line() +
  labs(title="Ceny zamknięcia WIG", x="Data", y="Zamknięcie")


##### 2. Sprawdzenie czy dane nadają się do modelu ARIMA

# test Augmented Dickey-Fuller (ADF), sprawdzający stacjonarność danych (założenie ARIMA); H0: szereg niestacjonarny jeżeli p-value > 0.05
adf.test(dane$Zamkniecie) 
# Wniosek: p-value > 0,05, czyli nie ma podstaw do odrzucenia H0, tj. szereg niestacjonarny -> wymaga dopasowania

# Autokorelacja (ACF) i częściowa autokorelacja (PACF)
acf(dane$Zamkniecie, main="ACF Ceny Zamknięcia") 
pacf(dane$Zamkniecie, main="PACF Ceny Zamknięcia")

# Interpretacja ACF - jak są ze sobą skorelowane wartości opóźnione (przeszłe): 
# Widzimy silne dodatnie korelacje przy małych opóźnieniach - wartości szeregu czasowego 
#  mają tendencję do bycia podobnymi do wartości poprzednich.
# Wartości ACF opadają stopniowo - wskazuje na obecność trendu w danych. 
# Z wykresu odczytuje obecność trendu lub sezonowości -> wymaga dopasowania

# Interpretacja PACF - przy eliminacji wpływu obserwacji pośrednich:
# Widzimy słabe naprzemiennie zmieniające się (dodatnie-ujemne) korelacje dla rosnących lagów.
# Znaczące wartości (lag 3, 15 oraz 23) sugeruje, że model AR być odpowiedni dla tych opóźnień.

# Sprawdzenie parametrów ARIMA dla naszych danych
summary(model_surowy <- arima(dane$Zamkniecie))
# AIC = 22690.16; RMSE = 12527.51

# Diagnostyka reszt (czy spełnia założania iid)
tsdisplay(residuals(model_surowy), main="Reszty Modelu ARIMA")
checkresiduals(model_surowy)
# średnio spełnia założenia iid



############## Ręczne dostosowanie szeregu 

##### 3. Stabilizacja (wybicie trendu) po przez różnicowanie

# analiza różnic między kolejnymi obserwacjami w szeregu czasowym
y1 <- diff(dane$Zamkniecie, lag=1, differences=1) #diff(x, lag = 1, differences = 1) 
cat(mean(y1), sd(y1))
adf.test(y1) # p-value < 0.05 -> sugeruje na szereg niestacjonarny 
plot(y1, type="l", main="różnicowanie lag=1")

y2 <- diff(dane$Zamkniecie, lag=1, differences=10) # wielokrotne różnicowanie
adf.test(y2)
#cat(mean(y2), sd(y2))
#plot(y2, type="l", main="różnicowanie wielokrotne")

y3 <- diff(log(dane$Zamkniecie), lag=1, differences=1) # logarytmowanie
adf.test(y3)
#cat(mean(y3), sd(y3))
#plot(y3, type="l", main="różnicowanie log(dane)")

# Diagnostyka reszt dla wybranego dostosowania szeregu
tsdisplay(residuals(arima(y1)), main="Reszty Modelu y1 (diff lag 1)")

# Wniosek: nawet jednokrotne różnicowanie wyraźnie stabilizuje szereg
# Na podstawie ACF i PACF, wydaje się, iż odpowiednim będzie AR(2) oraz MA(2)


##### 4. Analiza odwrotności różnicowania (a więc do postaci szeregu pierwotnego - integrowanie)
z1 <- diffinv(dane$Zamkniecie, lag=1, differences=1) # diff(log(dane)) pokazuje zmianę procentową
cat(mean(z1), sd(z1))
tsdisplay(residuals(arima(z1)), main="Integrowanie Modelu z1 (diffinv lag 1)")

z2 <- diffinv(log(dane$Zamkniecie), lag=1, differences=1) # logartmowane wartości
tsdisplay(residuals(arima(z1)), main="Integrowanie Modelu z2 (diffinv(log)) lag 1)")
adf.test(z2)

# Diagnostyka reszt
tsdisplay(residuals(arima(z2)), main="Reszty Modelu z2")


##### 5. Analiza z wyodrębnionym trendem deterministycznym

# Trend liniowy
dane$Trend_lin <- 1:nrow(dane)
x1 <- lm(Zamkniecie ~ Trend_lin, data=dane)
x1_residuals <- residuals(x1)
summary(arima(x1_residuals))

cat(mean(x1_residuals), sd(x1_residuals))
adf.test(x1_residuals) # p-value > 0.05 -> sugeruje na szereg nieniestacjonarny 
tsdisplay(residuals(arima(x1_residuals)), main="Reszty Modelu x1")



##### 6. Symulacjia - ręczny dobór parametrów do modelu ARIMA

# Symulacja dla AR(1), I(1) MA(2); wartości parametrów dobrane metodą prób i błędów
model112 <- arima.sim(model = list(order = c(1, 1, 2), # środkowa liczba dotyczy rzędu integrowania (I)
                                    ar = c(0.5), # jeden parametr phi, bo zadaliśmy AR(1)
                                    ma = c(0.4, -0.5)), # dwa parametry theta, bo zadaliśmy MA(2)
                       n = 100)

summary(model112)
tsdisplay(residuals(arima(model112)), main="Symulowane dane ARIMA(1, 1, 2)")


##### 7. Porównanie kilku modeli ARIMA oraz ocena dopasowania (AIC) 

# Walidacja modelu (dążymy do minimalizacji AIC)
model1 <- arima(dane$Zamkniecie, order = c(0, 0, 1)) 
summary(model1) # AIC = 21394.51

model2 <- arima(diff(dane$Zamkniecie), order = c(1, 0, 1)) 
summary(model2) # AIC = 17952.63

model3 <- arima(log(dane$Zamkniecie), order = c(1, 1, 2)) 
summary(model3) # AIC = -4545.95



##### 8. Model ARIMA za pomocą auto.arima()

# Automatyczne dopasowanie modelu ARIMA
model_auto <- auto.arima(dane$Zamkniecie)
summary(model_auto)
# ARIMA(0,1,2); AIC = 17949.21; RMSE = 1305.142

# Diagnostyka reszt (iid)
tsdisplay(residuals(model_auto), main="Reszty Dopasowanego Modelu ARIMA")
checkresiduals(model_auto)
# Automatyczne dopasowanie parametrów modelu ARIMA znacząco korzystne dla analizy



##### 9. Porównanie modeli za pomocą AIC i BIC
model3_aic <- AIC(model3)
model3_bic <- BIC(model3)
cat("Model dopasowany ręcznie AIC:", model3_aic, "BIC:", model3_bic, "\n")

model_auto_aic <- AIC(model_auto)
model_auto_bic <- BIC(model_auto)
cat("Model dopasowany automatycznie AIC:", model_auto_aic, "BIC:", model_auto_bic, "\n")

# Analiza reszt
checkresiduals(model3)
checkresiduals(model_auto)


##### 10. Prognoza przyszłych obserwacji

# Usuwamy kolumny
dane <- subset(dane, select = -c(Data, Wolumen))

x_lat <- 10
forecast_horizon <- x_lat * 52 # wyrażony w tygodniach
forecast_data <- forecast(model_auto, h=forecast_horizon)
autoplot(forecast_data)

# Wykonanie prognozy z przedziałami ufności
powtorzenia = forecast_horizon
simulations <- matrix(nrow=forecast_horizon*7, ncol=powtorzenia) # wiersz to prognoza dla kolejnego tygodnia
for (i in 1:powtorzenia) { simulations[,i] <- simulate(model_auto, nsim=forecast_horizon) }

plot(dane$Zamkniecie, type="l", xlim=c(800, nrow(dane)+forecast_horizon+50), ylim=c(40000,120000), main="Prognoza")

start = nrow(dane) # początek prognozy na osi odciętych
anchor = dane$Zamkniecie[start] # początek prognozy na osi rzędnych
fan(simulations, start=start, anchor=anchor, type="interval",  probs=seq(5, 95, 5), ln=c(50, 80))


##### End