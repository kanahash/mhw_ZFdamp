import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import math
import scipy.fftpack as sf

from google.colab import drive
drive.mount('/content/drive')

# -*- coding: utf-8 -*-
"""1．MHW計算～データの書き出し  test

# Commented out IPython magic to ensure Python compatibility.
# %config InlineBackend.figure_format = 'retina'


"""##MHW方程式の実行"""

def MHW(nx,ny,lx,ly,nt,dt,kap,alph,nu,mu,phi,n,isav):
    global KX,KY,KX2,KY2,KXD,KYD

    dx=lx/nx; dy=ly/ny

    ### define grids ###
    kx =2*np.pi/lx*np.r_[np.arange(nx/2),np.arange(-nx/2,0)]
    ky =2*np.pi/ly*np.r_[np.arange(ny/2),np.arange(-ny/2,0)]
    kxd=np.r_[np.ones(nx//3),np.zeros(nx//3+nx%3),np.ones(nx//3)]   #for de-aliasing
    kyd=np.r_[np.ones(ny//3),np.zeros(ny//3+ny%3),np.ones(ny//3)]   #for de-aliasing
    kx2=kx**2; ky2=ky**2
    KX ,KY =np.meshgrid(kx ,ky )
    KX2,KY2=np.meshgrid(kx2,ky2)
    KXD,KYD=np.meshgrid(kxd,kyd)

    phif=sf.fft2(phi)
    nf  =sf.fft2(n)
    #zetaf=sf.fft2(np.random.randn(nx,ny)*2)
    zetaf=-(KX2+KY2)*phif

    phihst =np.zeros((nt//isav,nx,ny))
    nhst   =np.zeros((nt//isav,nx,ny))
    zetahst=np.zeros((nt//isav,nx,ny))
    phihst[0,:,:] =np.real(sf.ifft2(phif)).T
    nhst[0,:,:]   =np.real(sf.ifft2(nf)).T
    zetahst[0,:,:]=np.real(sf.ifft2(zetaf)).T

    phifhst =np.zeros((nt//isav,nx,ny),dtype=complex)
    nfhst   =np.zeros((nt//isav,nx,ny),dtype=complex)
    zetafhst=np.zeros((nt//isav,nx,ny),dtype=complex)

    count=0

    for it in range(1,nt):

        #---double steps with integrating factor method(4th-order Runge-Kutta)---#
        zetaf=np.exp(-mu*(KX2+KY2)**2*dt)*zetaf
        nf   =np.exp(-mu*(KX2+KY2)**2*dt)*nf

        gw1,ga1=adv(zetaf           ,nf           )
        gw2,ga2=adv(zetaf+0.5*dt*gw1,nf+0.5*dt*ga1)
        gw3,ga3=adv(zetaf+0.5*dt*gw2,nf+0.5*dt*ga2)
        gw4,ga4=adv(zetaf+    dt*gw3,nf+    dt*ga3)

        zetaf=zetaf+dt*(gw1+2*gw2+2*gw3+gw4)/6
        nf   =nf   +dt*(ga1+2*ga2+2*ga3+ga4)/6

        if(it%isav==0):
            phif=zetaf/(-(KX2+KY2)); phif[0,0]=0
            phi=np.real(sf.ifft2(phif))
            n   =np.real(sf.ifft2(nf))
            zeta=np.real(sf.ifft2(zetaf))
            phihst[it//isav,:,:]=phi.T
            nhst[it//isav,:,:]=n.T
            zetahst[it//isav,:,:]=zeta.T
            phifhst[it//isav,:,:]=phif.T
            nfhst[it//isav,:,:]=nf.T
            zetafhst[it//isav,:,:]=zetaf.T

            print(count)
            count+=1

   ## return locals()
    return phihst,nhst,zetahst,phifhst,nfhst,zetafhst

def adv(zetaf,nf):
    phif=zetaf/(-(KX2+KY2)); phif[0,0]=0

    phi=np.real(sf.ifft2(phif))
    n  =np.real(sf.ifft2(nf))

    phiz=np.sum(phi*dy,axis=0)/ly
    nz  =np.sum(n  *dy,axis=0)/ly

    phixf = 1j*KX*phif;  phix =np.real(sf.ifft2(phixf *KXD*KYD))
    phiyf = 1j*KY*phif;  phiy =np.real(sf.ifft2(phiyf *KXD*KYD))
    zetaxf= 1j*KX*zetaf; zetax=np.real(sf.ifft2(zetaxf*KXD*KYD))
    zetayf= 1j*KY*zetaf; zetay=np.real(sf.ifft2(zetayf*KXD*KYD))
    nxf   = 1j*KX*nf;    nnx   =np.real(sf.ifft2(nxf  *KXD*KYD))
    nyf   = 1j*KY*nf;    nny   =np.real(sf.ifft2(nyf  *KXD*KYD))
    zeta =np.real(sf.ifft2(zetaf *KXD*KYD))

    advf =-(phix*zetay-phiy*zetax)+alph*((phi-phiz)-(n-nz))
    advg =-(phix*nny  -phiy*nnx)  +alph*((phi-phiz)-(n-nz))-kap*np.real(sf.ifft2(phiyf))
    advff=sf.fft2(advf)
    advgf=sf.fft2(advg)

    return advff,advgf

"""## パラメーターの設定"""

nx=128; ny=256; nt=50000; isav=25
kap=1.0
alph=3.0
nu=0
mu=1e-4
dt=2e-4
lx=2*np.pi/0.15; ly=2*np.pi/0.15
dx=lx/nx; dy=ly/ny
x  =np.arange(nx)*dx
y  =np.arange(ny)*dy
X,Y=np.meshgrid(x,y)

s=2; s2=s**2
r1=(X-lx/2)**2+(Y-ly/2)**2
n  =np.exp(-r1/s2)
phi=n

"""## 計算"""

path="/content/drive/MyDrive/hw_simuration/mhw_ZFdamp/0/"
data=MHW(nx,ny,lx,ly,nt,dt,kap,alph,nu,mu,phi,n,isav)

from numpy.core.multiarray import ndarray
datanp = np.asarray(data)

datanp.shape

"""## 関数の定義"""

phi = datanp[0,:,:,:]
n = datanp[1,:,:,:]
zeta = datanp[2,:,:,:]
phif = datanp[3,:,:,:]
nf = datanp[4,:,:,:]
zetaf = datanp[5,:,:,:]



"""# データの書き出し"""

##np.savetxt("/content/drive/MyDrive/研究室/HW 書き直し/t=20000 data/MHW2 t=20000.txt",datanp2)

params = [nx,ny,nt,isav,kap,alph,nu,mu,dt,lx,ly]

#path="/content/drive/MyDrive/HWsimuration/run10/3/"
##params(パラメータ)をファイルへ保存
np.save(path+"params.npy",params)

np.save(path+"mhw.npy",datanp)

#path="/content/drive/MyDrive/HWsimuration/run/000/"
#params = np.load(path+"params t=100.npy")
#mhw = np.load(path+"mhw.npy")


"""# 時間ごとに分けて保存する"""

dt*isav

N_save = int(nt/isav)
t_max = dt*nt
t_save_arr = np.linspace(0,t_max,N_save)
np.save(path+"t_save_arr.npy",t_save_arr)
