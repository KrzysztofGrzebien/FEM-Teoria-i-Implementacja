import numpy as np
import matplotlib.pyplot as plt

c=0
f=0

x_0=0
x_p=1
n=8

twb_L='D'
twb_P='D'

wwb_L=0
wwb_P=1


def generujTabliceGeometrii(x_0,x_p,n):
    tmp1=np.arange(1,n+1,1)
    tmp2=np.linspace(x_0,x_p,n)

    ZMN=np.block([[tmp1],
                    [tmp2]])
    WEZLY=ZMN.transpose()

    tmp3 = np.arange(1, n, 1)
    tmp4 = np.arange(1, n, 1)
    tmp5 = np.arange(2, n+1, 1)

    ZMN2 = np.block([[tmp3],
                         [tmp4],
                         [tmp5]])

    ELEMENTY=ZMN2.transpose()
    return WEZLY, ELEMENTY

def rysuj_geometrie(WEZLY,ELEMENTY,WAR_BRZEGOWE):
    x=WEZLY[:,1]
    y=0*WEZLY[:,1]

    x_pos_w=WEZLY[:,1]
    y_pos_w=x_pos_w*0-0.01
    text_w=WEZLY[:,0]

    x_pos_e=np.zeros((len(WEZLY[:,0])-1, 1))
    y_pos_e=np.zeros((len(WEZLY[:,0])-1, 1))
    text_e=np.zeros((len(WEZLY[:,0])-1, 1))

    for i in range(0,len(x_pos_w)):
        plt.text(x_pos_w[i],y_pos_w[i],int(text_w[i]))

    for j in range(0,len(x)-1):
        x_pos_e[j]=(x[j]+x[j+1])/2
        y_pos_e[j]=0.01
        text_e[j]=ELEMENTY[j,0]
        plt.text(x_pos_e[j], y_pos_e[j], int(text_e[j]))

    plt.plot(x,y,color='green',marker='o',linestyle='dashed')
    plt.grid(True)
    plt.show()

a,b= generujTabliceGeometrii(x_0,x_p,n)

print(a)
print(b)

rysuj_geometrie(a,b,np.array([wwb_L,wwb_P]))
