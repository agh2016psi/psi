#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Siec dwukomorkowa Kohonena (kazde wejscie polączone do każdej komórki):
# 2 wejscia x1, x2 dla komorki 1, 1 wyjscie y1.
# Te same wejscia dla komorki 2. Komorka 2 ma 1 wyjscie y2.
# DEFINIUJEMY wektor wejść:

import numpy as np
import matplotlib.pyplot as plt

x1 = 0.5
x2 = 0.9
# Macierz wag W bedzie więc mieć 4 składowe, w11 i w12 dla komórki pierwszej
# oraz w21 i w22 dla komórki drugiej. Przyjmijmy wagi poczatkowe.
w11 = 0.5
w12 = 0.7
w21 = 0.3
w22 = 0.5
# Uogólniony Algorytm HEBBA
# Współczynnik uczenia eta niechaj wynosi 0.5. DEFINIUJEMY:
eta = 0.1
# Sumowanie S bedzie dane wzorami: y1 = w11*x1 + w12*x2, y2 = w21*x1 + w22*x2
# DEFINIUJEMY funkcje y. Niechaj bedzie ona dana ogólnie w ten sposób:


def y(w1, w2):
    return w1 * x1 + w2 * x2
# W modelu HEBBA w PIERWSZYM KROKU OBLICZEN przyjmujemy wartości y1, y2=1 dla obliczeń korekty wag
# y1=y(w11, w12);y2=y(w21, w22);
y1 = 1
y2 = 1
# Wypiszmy pierwszym razem wartości początkowe:
print ("w11 {}\t w12 {}\t w21 {}\t w22 {}\t y1 {}\t y2 {}".format(
    w11, w12, w21, w22, y1, y2))
# print(`w11`+"{}\t"+`w12`+"{}\t"+`w21`+"{}\t"+`w22`+"{}\t"+`y1`+"{}\t"+`y2`,w11,w12,w21,w22,y1,y2);
# KROK DRUGI OBLICZEN. Obliczamy nowe wartości macierzy wag w modelu HEBBA:
w11 = w11 + eta * y1 * (x1 - w11 * y1)
w12 = w12 + eta * y2 * (x1 - w11 * y1 - w12 * y2)
w21 = w21 + eta * y1 * (x2 - w21 * y1)
w22 = w22 + eta * y2 * (x2 - w21 * y1 - w22 * y2)
# KROK TRZECI OBLICZEN
# Zobaczmy, jakie sa nowe wartosci macierzy w oraz jaka jest nowa wartosc y:
print ("w11 {}\t w12 {}\t w21 {}\t w22 {}\t y1 {}\t y2 {}".format(
    w11, w12, w21, w22, y1, y2))
plt.plot(4)
# POWTARZAMY KROKI od 1 do 3 (w pętli), az do czasu gdy y staje sie
# dostatecznie bliskie z.
for i in range(0, 50, 1):
    y1 = y(w11, w12)
    y2 = y(w21, w22)
    w11 = w11 + eta * y1 * (x1 - w11 * y1)
    w12 = w12 + eta * y2 * (x1 - w11 * y1 - w12 * y2)
    w21 = w21 + eta * y1 * (x2 - w21 * y1)
    w22 = w22 + eta * y2 * (x2 - w21 * y1 - w22 * y2)
    print ("i {}\t w11 {}\t w12 {}\t w21 {}\t w22 {}\t y1 {}\t y2 {}".format(
        i, w11, w12, w21, w22, y1, y2))
