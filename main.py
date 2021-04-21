# CW- 6.2

import numpy as np
import matplotlib.pyplot as plt
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# tworzy i wyświetla arr, czyli [1 2 3 4 5]

A=np.array([[1,2,3],[7,8,9]])
print(A)

#tworzy i  wyświetla A, czyli [1 2 3]
#                             [7 8 9]

A = np.array([[1, 2, 3],
              [7, 8, 9]])
print(A)
#tworzy i  wyświetla A, czyli [1 2 3]
#                             [7 8 9]


A=np.array([[1 , 2 , \
             3],
            [1 , 8 , 9]])
print(A)

#wyświetla A, ale \ pozwala na przeniesienie do następnej linii

v = np.arange(1,7)
print(v,"\n")
# generuje wektor w zakresie 1 do 6 co 1
v = np.arange(-2,7)
print(v,"\n")
# generuje wektor w zakresie -2 do 6 co 1
v = np.arange(1,10,3)
print(v,"\n")
# generuje wektor w zakresie 1 do 10 co 3
v = np.arange(1,10.1,3)
print(v,"\n")
# generuje wektor w zakresie 1.0 do 10.1 co 3, więc 10.0 też zostaje dodane
v = np.arange(1,11,3)
print(v,"\n")
# generuje wektor w zakresie 1 do 11 co 3, więc 10 też zostaje dodane
v = np.arange(1,2,0.1)
print(v,"\n")
# generuje wektor w zakresie 1 do 2 co 0.1

v = np.linspace(1,3,4)
print(v)

#tworzy 4 elementowy wektor obejmujący zakres 1-3

v = np.linspace(1,10,4)
print(v)

#tworzy 4 elementowy wektor obejmujący zakres 1-10

X = np.ones((2,3))
#tworzy macierz jedynek o rozmiarze 2x3
Y = np.zeros((2,3,4))
#tworzy macierz zer o rozmiarze 3x4
Z = np.eye(2) # np.eye(2,2)
#tworzy macierz jednostkową o rozmiarze 2
# np.eye(2,3) -> macierz jednostkowa musi być kwadratowa
Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))

#tworzy macierz losową o rozmiarze 2x5

print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)

#U = np.block([[A], [X,Z]])
#print(U)
# tak nie można skleić, bo wymiary się nie zgadzają

U = np.block([[A], [X]])
print(U)

#skleja macierze A i X

V = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)

#Zagnieżdżone sklejanie wielu macierzy

print( V[0,2] )
print( V[3,0] )
print( V[3,3] )
print( V[-1,-1] )
print( V[-4,-3] )
#Wyciąga odpowiedni element macierzy

print( V[3,:] )
print( V[:,2] )
print( V[3,0:3] )
print( V[np.ix_([0,2,3],[0,-1])] )
print( V[3] )

#wyciąga wiwle elementów macierzy

Q = np.delete(V, 2, 0)
print(Q)
Q = np.delete(V, 2, 1)
print(Q)
v = np.arange(1,7)
print( np.delete(v, 3, 0) )

#usuwa elementy macierzy pojedynczo, lub jako wektor
np.size(v)
np.shape(v)

#Wymiary macierzy

A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
print( A+B )
print( A-B )
print( A+2 )
print( 2*A )

#Dodawanie, odejmowanie, oraz mnożenie przez skalar macierzy.

MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)

#Mnożenie macierzowe

MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)

#Mnożenie tablicowe

DT1 = A/B
7
print(DT1)

#Dzielenie tablicowe, Rozwiązywanie układów równań liniowych

C = np.linalg.solve(A,MM1)
print(C) # porownaj z macierza B
x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)

#Potęgowanie macierzy

PM = np.linalg.matrix_power(A,2) # por. A@A
PT = A**2 # por. A*A

#Transpozycja

A.T # transpozycja
A.transpose()
A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
A.conj().transpose()

#Operatory porównania

np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)
print( np.all(A) )
print( np.any(A) )
print( v > 4 )
print( np.logical_or(v>4, v<2))
print( np.nonzero(v>4) )
print( v[np.nonzero(v>4) ] )
#Maximum, minimum

print(np.max(A))
print(np.min(A))
print(np.max(A,0))
print(np.max(A,1))
print( A.flatten() )
print( A.flatten('F') )

#Utworzenie wykresu

x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()

#Wykres Sinusa z opisem i pogrubieniem

x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y,'r:',linewidth=6)

plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)
plt.show()

#Wiele przebiegów na jednym wykresie

x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.plot(x,y1,'r:',x,y2,'g')
plt.legend(('dane y1','dane y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()
#Wiele przebiegów na jednym wykresie, inna legenda

x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,'b')
l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()

# CW- 6.3

T1=np.arange(1,6)
T2=np.arange(5,0,-1)
T3=2*np.ones((2,3))
T4=np.zeros((3,2))
T5=np.linspace(-90,-70,3)
T6=10*np.ones((5,1))

U1 = np.block([[T1], [T2]])
U2= np.block([[T3], [T5]])
U3= np.block([T4,U2])
U4= np.block([[U1],[U3]])
U5= np.block([U4,T6])

A=U5

print(A)


# CW- 6.4

B=np.array(A[2,:]+A[4,:])
print(B)


# CW- 6.5

C=np.ones((1,np.size(A,1)))

for i in range(0,np.size(A,1)):
    C[0,i]=max(A[:,i])
print(C)


# CW- 6.6

D=np.delete(B,[0,len(B)-1])
print(D)


# CW- 6.7

for i in range(0,len(D)-1):
    if D[i]==4:
        D[i]=0
print(D)


# CW- 6.8

C=max(C);# To jest specjalnie, żeby zamienić C z macierzy na wektor
print(C)
mac=max(C)
mic=min(C)
zakres=len(C)
tmp=max(np.ones((1,len(C))))
j=0

for i in range(0,zakres):
    print(C[i])
    if (C[i]!=mic and C[i]!=mac):
        tmp[j]=C[i]
        j=j+1
tmp=np.delete(tmp,[len(tmp)-2,len(tmp)-1])
E=np.array(tmp)
print(E)
print(A)


#CW - 6.9

for i in range (0,np.size(A,0) ):
    tmp=A[i,:]
    if max(tmp)==np.amax(A):
        print(tmp)

for i in range (0,np.size(A,0) ):
    tmp=A[i,:]
    if min(tmp)==np.amin(A):
        print(tmp)


#CW - 6.10
print(D*E)
print(E*D)
print(D@E)
print(E@D)

#CW - 6.11
def rand_sq(n):
    mat=np.random.randint(11,size=(n,n))
    print(mat)
    slad=np.trace(mat)
    print(slad)
    return mat,slad
ceb,sl=rand_sq(6)


#CW - 6.12

def zerowanie(tab):
    sy=np.size(tab,0)
    sx=np.size(tab,1)
    for i in range(0,sy):
        for j in range(0,sx):
            if i==j or i+j==sy-1:
                tab[i,j]=0
    print(tab)
zerowanie(ceb)


#CW - 6.13

def sumowanie(mat):
    suma=0
    sy=np.size(mat,0)
    sx=np.size(mat,1)
    for i in range(0,sy):
        for j in range(0,sx):
            if i%2==0:
                suma=suma+mat[i,j]
    print(suma)

print(ceb)
sumowanie(ceb)


#CW - 6.14

fun_cos=lambda x:np.cos(2*x)
x=np.arange(-10,10,0.1)
y=fun_cos(x)
plt.plot(x,y,color='red',linestyle='dashed')
plt.show()
print(fun_cos(np.pi/4))


#CW - 6.15


#CW - 6.16

fun_sin_sqrt=lambda z:np.where(z<0,np.sin(z),np.sqrt(z))
y2=fun_sin_sqrt(x)
plt.plot(x,y,color='red',linestyle='dashed')
plt.plot(x,y2,'g+')
plt.show()


#CW - 6.17

fun_3=lambda x:3*fun_cos(x)+fun_sin_sqrt(x)
y3=fun_3(x)
plt.plot(x,y,color='red',linestyle='dashed')
plt.plot(x,y2,'g+')
plt.plot(x,y3,'b*')
plt.show()


#CW - 6.18

a=np.array([[10, 5, 1, 7],[10, 9, 5, 5],[1, 6, 7, 3],[10, 0, 1, 5]])
b=np.array([34, 44, 25, 27])
xx=np.linalg.solve(a,b)
print(xx)


#CW - 6.19

x=np.linspace(0,2*np.pi,1000000)
fun_4=lambda x:np.sin(x)
y_4=fun_4(x)
int_fun=np.trapz(y_4)
plt.plot(x,y_4,color='green',linestyle='dotted')
plt.show()
print(int_fun)


#CW - 6.20

A=np.array([[2.0, 1.0, -1.0],[1.0, -1.0, 1.0],[-3.0, 2.0, 0.0]])
B=np.array([[2.0],[-5.0],[17.0]])
U=np.block([A,B])
sy=np.size(U,0)
sx=np.size(U,1)
print(U)
print(sy)
print(sx)

for i in range(0,sy):
    U[i,:]=U[i,:]/U[i,i]

    for j in range(0,sy):
        print(U)

        if j!=i:
            U[j, :] = U[j, :] - U[i, :] * (U[j, i] / U[i, i])
print(U)




