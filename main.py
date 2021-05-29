import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint


def GeometriaDefinicja():
    # Węzły zadane w sposób ręczny:
    WEZLY = np.array([[1, 0],
                      [2, 1],
                      [3, 0.5],
                      [4, 0.75]])

    # Elementy zadane w sposób ręczny:
    ELEMENTY = np.array([[1, 1, 3],
                      [2, 4, 2],
                      [3, 3, 4]])

    # Definicja warunków brzegowyvh
    # Warunki brzegowe zaimplementowane zostały w postaci słownika:
    #    nr_wezla, typ_warunku(D- Dirichleta), wartosc warunku brzegowego
    WB = [{"ind": 1, "typ": 'D', "wartosc": 1},
          {"ind": 2, "typ": 'D', "wartosc": 2}]

    return WEZLY, ELEMENTY, WB


def AutomatycznyGeneratorGeometrii(a, b, n):

    #przygotowujemy indeksy węzłów
    lp = np.arange(1, n + 1)
    #przygotowujemy współrzędne węzłów
    x = np.linspace(a, b, n);
    #tworzymy tablicę WEZLY o kolumnach: indeks, współrzędna
    WEZLY = (np.vstack((lp.T, x.T))).T  # [lp.T, x.T]

    #przygotowujemy indeksy elementów
    lp = np.arange(1, n)
    # przygotowujemy indeksy węzłów elementów
    C1 = np.arange(1, n)
    C2 = np.arange(2, n + 1)
    # tworzymy tablicę ELEMENTY o kolumnach: indeks elementu,
    # indeks pierwszego węzła, indeks drugiego węzła
    ELEMENTY = (np.block([[lp], [C1], [C2]])).T

    return WEZLY, ELEMENTY


def RysujGeometrie(WEZLY, ELEMENTY, WB):
    fh = plt.figure()
    #Rysujemy naszą geometrię
    plt.plot(WEZLY[:, 1], np.zeros((np.shape(WEZLY)[0], 1)), '-b|')
    nodeNo = np.shape(WEZLY)[0]

    for ii in np.arange(0, nodeNo):
        ind = WEZLY[ii, 0]
        x = WEZLY[ii, 1]
        #rysujemy indeksy węzłów
        plt.text(x, 0.01, str(int(ind)), c="b")
        # rysujemy współrzędne węzłów
        plt.text(x, -0.01, str(round(x,2)))

    elemNo = np.shape(ELEMENTY)[0]
    for ii in np.arange(0, elemNo):
        wp = ELEMENTY[ii, 1]
        wk = ELEMENTY[ii, 2]

        x = (WEZLY[wp - 1, 1] + WEZLY[wk - 1, 1]) / 2
        # rysujemy indeksy elementów
        plt.text(x, 0.01, str(ii + 1), c="r")
    plt.show()
    return fh


def Alokacja(n):

    # Alokujemy miejsce dla macierzy A oraz wektora b.
    # Początkowo inicjujemy je zerami.

    #Rozmiary macierzy A są równe liczbie węzłów
    A = np.zeros([n, n])

    #Wektor b ma rozmiar ilość węzłów x 1
    b = np.zeros([n, 1])

    return A, b


def FunkcjeBazowe(n):
    #Jeżeli interpolujemy wielomianami zerowego rzędu
    if n == 0:
        f = (lambda x: 1 + 0 * x)
        df = (lambda x: 0 * x)
    #Jeżeli interpolujemy wielomianami pierwszego rzędu
    elif n == 1:
        f = (lambda x: -1 / 2 * x + 1 / 2, lambda x: 0.5 * x + 0.5)
        df = (lambda x: -1 / 2 + 0 * x, lambda x: 0.5 + 0 * x)
    #Jeżeli interpolujemy wielomianami drugiego rzędu
    elif n == 2:
        f =  (lambda x: (1/2)*(x-1)*x , lambda x: 1-(x*x), lambda x: (1/2)*(x+1)*x )
        df = (lambda x: x-(1/2) , lambda x: -2*x, lambda x: x+(1/2) )
    #W przypadku podania innego, niż powyższe rzędu, zgłaszamy wyjątek
    else:
        raise Exception("Blad w funkcji FunkcjeBazowe().")
    return f, df


def Aij(df_i, df_j, c, f_i, f_j):
    #Jako parametry przyjmujemy funkcje
    # bazowe i-tego oraz j-ego węzła, a także ich pochodne
    #zwracamy funkcję podcałkową jako funkcję lambda
    return lambda x: -df_i(x) * df_j(x) + c * f_i(x) * f_j(x)


def RysujRozwiazanie(WEZLY, ELEMENTY, WB, u):
    #jako parametry przyjmujemy węzły, elementy, warunki brzegowe oraz rozwiązanie
    #Najpierw wywołujemy wcześniej zdefiniowaną funkcję rysującą geometrię
    RysujGeometrie(WEZLY, ELEMENTY, WB)
    x = WEZLY[:, 1]
    #Następnie na wcześniej utworzonym wykresie dorysowujemy rozwiązanie u

    plt.plot(x, u, 'm*')
    plt.show()

if __name__ == '__main__':

######################################## PRE - PROCESSING ##############################################################

    ## 1.a) parametry sterujace
    c = 0
    f = lambda x: 0 * x  # Przyjmujemy brak wymuszenia

    ## 1.b) Definicja geometrii

    # W przypadku ręcznego zadawania geometrii,
    # wykorzystywany jest ten fragment

    # WEZLY, ELEMENTY, WB = GeometriaDefinicja()
    # n = np.shape(WEZLY)[0]

    # W przypadku automatycznego generowania
    # geometrii, wykorzystywany jest ten fragment

    #Definiujemy krańce przedziału
    x_a = -4
    x_b = 4

    #Definiujemy liczbę węzłów
    n = 88

    #Wywołujemy funkcję automatycznie generującą geometrię.
    WEZLY, ELEMENTY = AutomatycznyGeneratorGeometrii(x_a, x_b, n)

    ## 1.c) Definicja warunków brzegowych

    # Definiujemy warunki brzegowe Dirichleta
    WB = [{"ind": 1, "typ": 'D', "wartosc": 100},
          {"ind": n, "typ": 'D', "wartosc": 22}]

    ## 1.d) Prezentacja geometrii zagadnienia
    #Wywołujemy funkcję tworzącą wykres utworzonej geometrii
    RysujGeometrie(WEZLY, ELEMENTY, WB)

    ## 1.e) utworzenie macierzy wypełnionych zerami
    # Wywołujemy funkcję Alokującą miejsce dla macierzy A oraz wektora b
    A, b = Alokacja(n)

    ## 1.f) Definicja funkcji bazowych
    #Będziemy stosować wielomiany 1 rzędu
    stopien_fun_bazowych = 1
    #Do zmiennej phi przypisujemy funkcje bazowe, natomiast do dphi ich pochodne
    phi, dphi = FunkcjeBazowe(stopien_fun_bazowych)



############################################### PROCESSING #############################################################

    #Sprawdzamy liczbę elementów na podstawie rozmiarów macierzy ELEMENTY
    liczbaElementow = np.shape(ELEMENTY)[0]

    #Pętla przechodząca przez wszystkie elementy geometrii
    for ee in np.arange(0, liczbaElementow):

        #Indeks w pętli bierzącego elementu
        elemRowInd = ee

        #Globalny indeks bieżącego elementu
        elemGlobalInd = ELEMENTY[ee, 0]

        #Indeks wezla początkowego bieżącego elementu w pętli
        elemWezel1 = ELEMENTY[ee, 1]

        #Indeks wezla koncowego bieżącego elementu w pętli
        elemWezel2 = ELEMENTY[ee, 2]

        #Indeksy węzłów początkowego oraz końcowego bieżącego elementu
        indGlobalneWezlow = np.array([elemWezel1, elemWezel2])

        #współrzędne węzłów bieżącego przedziału
        x_a = WEZLY[elemWezel1 - 1, 1]
        x_b = WEZLY[elemWezel2 - 1, 1]

        ## 2.a) Obliczenie macierzy lokalnej

        #Inicjalizacja macierzy lokalnej
        Ml = np.zeros([stopien_fun_bazowych + 1, stopien_fun_bazowych + 1])

        #Obliczenie Jakobianu
        J = (x_b - x_a) / 2


        #Obliczenie poszczególnych elementów macierzy lokalnej

        for m in range(2):
            for n in range(2):
                Ml[m, n] = J * spint.quad(Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]

        ## 2.b) Agregacja macierzy lokalnej w macierzy globalnej
        #Uzupełnienie macierzy A otrzymanymi macierzami lokalnymi
        A[np.ix_(indGlobalneWezlow - 1, indGlobalneWezlow - 1)] = \
            A[np.ix_(indGlobalneWezlow - 1, indGlobalneWezlow - 1)] + Ml


    ## 2.c) Uwzględnienie warunków brzegowych
    print(WB)
    #Jeżeli mamy warunek brzegowy Dirichleta,
    # do brzegowego węzła przypisujemy
    # wartość warunku brzegowego
    if WB[0]['typ'] == 'D':
        ind_wezla = WB[0]['ind']
        wart_war_brzeg = WB[0]['wartosc']
        iwp = ind_wezla - 1
        #Aby uprościć układ równań,
        # przemnażamy współczynnik przy elemencie
        # znanym z warunku brzegowego
        # przez bardzo dużą zmienną
        WZMACNIACZ = 10 ** 14
        b[iwp] = A[iwp, iwp] * WZMACNIACZ * wart_war_brzeg
        A[iwp, iwp] = A[iwp, iwp] * WZMACNIACZ
    #Jeżeli mamy warunek brzegowy Dirichleta,
    # do brzegowego węzła przypisujemy
    # wartość warunku brzegowego
    if WB[1]['typ'] == 'D':
        ind_wezla = WB[1]['ind']
        wart_war_brzeg = WB[1]['wartosc']
        iwp = ind_wezla - 1
        # Aby uprościć układ równań,
        # przemnażamy współczynnik przy elemencie
        # znanym z warunku brzegowego
        # przez bardzo dużą zmienną
        WZMACNIACZ = 10 ** 14
        b[iwp] = A[iwp, iwp] * WZMACNIACZ * wart_war_brzeg
        A[iwp, iwp] = A[iwp, iwp] * WZMACNIACZ

    ## 2.d) Rozwiązanie problemu
    # Rozwiązanie liniowego układu równań
    u = np.linalg.solve(A, b)

################################################ POST - PROCESSING #####################################################

    ## 3.a) Prezentacja graficzna rozwiązania
    #Rysujemy rozwiązanie funkcją RysujRozwiazanie
    RysujRozwiazanie(WEZLY, ELEMENTY, WB, u)
