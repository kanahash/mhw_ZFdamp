import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sf
from IPython import display
from IPython.display import HTML
import math as mt
import matplotlib.animation as animation
import os

from google.colab import drive
drive.mount('/content/drive')

shot_no=11
dir='/content/drive/MyDrive/hw_simuration/MHW_data/data{}/'.format(shot_no)

phi_evo=np.load(dir+'phi_evo.npy')
N_evo=np.load(dir+'N_evo.npy')
t_arr=np.load(dir+'t_arr.npy')
x_arr=np.load(dir+'x_arr.npy')
y_arr=np.load(dir+'y_arr.npy')

[nx,ny,lx,ly,nt,dt,kap,alph,mu,nu]=np.load(dir+'params.npy')

x_arr.shape

phi_evo = np.transpose(phi_evo, (0, 2, 1))  # [t, x, y] の形状に変換
N_evo = np.transpose(N_evo, (0, 2, 1))  # [t, x, y] の形状に変換

N_t=t_arr.size
N_x=x_arr.size
N_y=y_arr.size

phi_evo.shape

lx.shape

ly.shape

t_max = 2000
# nx_max = 256
# ny_max = 256

# delta_x = 256
# delta_y = 256

# lx = np.linspace(lx - delta_x, lx + delta_x, 256)
# ly = np.linspace(ly - delta_y, ly + delta_y, 256)

# tの範囲をphi_evoの形状に合わせる
t = np.linspace(0, t_max, 2000)  # 2000個の要素

lx = np.linspace(0, 2 * np.pi / 0.15, 256)  # 0から2π/0.15までの範囲
ly = np.linspace(0, 32 * np.pi, 256)  # 0から32πまでの範囲

# t = np.arange(t_max)

X,Y = np.meshgrid( t, lx)

contour = plt.contourf(X,Y,phi_evo[:,0,:].T,100,cmap = 'hot')

contour.set_clim(-0.03, 0.03)

plt.title("phi (nu=0.22)")
plt.xlabel('Time (t)')
plt.ylabel('Y')
plt.colorbar(contour, label='phihst_mean')

# グラフを表示
plt.show()

plt.figure(figsize=(15, 7))

X,Y = np.meshgrid( t, ly)

contour = plt.contourf(X,Y,phi_evo[:,:,0].T,100,cmap = 'hot')

contour.set_clim(-0.03, 0.03)

# plt.title("phi (nu=0.22)")
plt.xlabel('Time (t)')
plt.ylabel('X')
plt.rcParams["font.size"] = 15

plt.tick_params(axis='both', which='major', labelsize=25)

plt.colorbar(contour)
# グラフを表示
plt.show()

X,Y = np.meshgrid( lx, ly)

contour = plt.contourf(X,Y,phi_evo[1800,:,:].T,100,cmap = 'hot')

#contour.set_clim(-0.03, 0.03)

plt.title("Contour plot of phihist (nu=0.22)")
plt.ylabel('X (lx)')
plt.xlabel('Y (ly)')
plt.colorbar(contour, label='phihst_mean')

# グラフを表示
plt.show()

# # lx, lyがもともと(256, 0)の形なので、lxとlyの正しい範囲を設定
delta_x = 2 * np.pi / 0.15
delta_y = 32 * np.pi

lx = np.linspace(-delta_x, delta_x, 256)  # lxの範囲を256に設定
ly = np.linspace(-delta_y, delta_y, 256)  # lyの範囲を256に設定

plt.figure(figsize=(15, 7))

X,Y = np.meshgrid( t, ly)

contour = plt.contourf(X,Y,phi_evo[:,:,0].T,100,cmap = 'hot')

contour.set_clim(-0.03, 0.03)

# plt.title("phi (nu=0.22)")
plt.xlabel('t',fontsize=60)
plt.ylabel('y',fontsize=60)
plt.rcParams["font.size"] = 15

plt.tick_params(axis='both', which='major', labelsize=60)

#plt.colorbar(contour)
# グラフを表示
plt.show()

X,Y = np.meshgrid( lx, ly)

contour = plt.contourf(X,Y,phi_evo[1800,:,:].T,100,cmap = 'hot')

contour.set_clim(-0.03, 0.03)

plt.title("Contour plot of phihist (nu=0.22)")
plt.ylabel('X (lx)')
plt.xlabel('Y (ly)')
plt.colorbar(contour, label='phihst_mean')

# グラフを表示
plt.show()

# figureとaxesを作成
fig, axes = plt.subplots(2, 3, figsize=(16, 12))

# 各サブプロットに異なる時刻の等高線を描画
time_indices = [900,1000, 1200, 1500, 1800, 1999]  # 使用するphi_evoのインデックス
for i in range(2):
    for j in range(3):
        idx = i * 3 + j  # time_indicesのインデックスを計算
        contour = axes[i][j].contourf(X, Y, phi_evo[time_indices[idx], :, :].T, 100, cmap='hot')
        axes[i][j].set_xlim(-delta_x, delta_x)
        axes[i][j].set_ylim(-delta_y, delta_y)

# カラーバーを追加
fig.colorbar(contour, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05)

plt.show()

N_evo.shape

def cal_x_derivative(phi,x_arr):
    dx_phi=np.zeros_like(phi)
    dx=x_arr[2]-x_arr[0]
    N_x=x_arr.size
    dx_phi[:,1:N_x-2,:]=(phi[:,2:N_x-1,:]-phi[:,0:N_x-3,:])/dx
    dx_phi[:,0,:]=(phi[:,1,:]-phi[:,N_x-1,:])/dx
    dx_phi[:,N_x-1,:]=(phi[:,0,:]-phi[:,N_x-2,:])/dx
    dx_phi[:,N_x-2,:]=(phi[:,N_x-1,:]-phi[:,N_x-3,:])/dx
    return dx_phi

def cal_y_derivative(phi,y_arr):
    dy_phi=np.zeros_like(phi)
    dy=y_arr[2]-y_arr[0]
    N_y=y_arr.size
    dy_phi[:,:,1:N_y-2]=(phi[:,:,2:N_y-1]-phi[:,:,0:N_y-3])/dy
    dy_phi[:,:,0]=(phi[:,:,1]-phi[:,:,N_y-1])/dy
    dy_phi[:,:,N_y-1]=(phi[:,:,0]-phi[:,:,N_y-2])/dy
    dy_phi[:,:,N_y-2]=(phi[:,:,N_y-1]-phi[:,:,N_y-3])/dy
    return dy_phi

def cal_ZF_comp(phi):
    phi_ZF=np.mean(phi,axis=2)
    dum1, dum2, dum3 =phi.shape
    expanded_data = np.expand_dims(phi_ZF, axis=2)
    result = np.repeat(expanded_data, dum3, axis=2)
    return result
    #return phi_ZF

def cal_ST_comp(phi):
    phi_ST=np.mean(phi,axis=1)
    dum1, dum2, dum3 =phi.shape
    expanded_data = np.expand_dims(phi_ST, axis=1)
    result = np.repeat(expanded_data, dum2, axis=1)
    return result

phi_ZF=cal_ZF_comp(phi_evo)
phi_ST=cal_ST_comp(phi_evo)
phi_turb=phi_evo-phi_ZF-phi_ST

N_ZF=cal_ZF_comp(N_evo)
N_ST=cal_ST_comp(N_evo)
N_turb=N_evo-N_ZF-N_ST

phi_turb[:, ::2, :].shape

V_y_ZF=cal_x_derivative(phi_ZF,x_arr)

V_x_ZF=cal_x_derivative(phi_ZF,x_arr)

x_arr.shape

dx_phi_turb=cal_x_derivative(phi_turb[::3,::3,::3],x_arr[::3])

dy_phi_turb=cal_y_derivative(phi_turb[::3,::3,::3],y_arr[::3])

Gamma_x=-N_evo[::3,::3,::3]*dy_phi_turb
Gama_x_ZF=cal_ZF_comp(Gamma_x)

plt.subplots(figsize = (3, 5))
plt.contourf(x_arr[::3],t_arr[::3],Gama_x_ZF[:,:,1], 50, cmap='hot')
plt.colorbar(shrink=0.8, pad=0.1)
plt.subplots_adjust(right=1.2)
contour.set_clim(-0.000003, 0.000003)

plt.xlabel('x')
plt.ylabel('t')
plt.show()

dum=N_turb[::3,::3,::3]**2+dx_phi_turb**2+dy_phi_turb**2
I_turb=np.mean(dum,axis=2)

bt=256

plt.plot(x_arr[::3],dx_phi_turb[bt,:,1]**2)

print("Gamma_x の形状:", Gamma_x.shape)

Gamma_x_y_avg = np.mean(Gamma_x, axis=1)

plt.contourf(dx_phi_turb[:,1,:]**2)
plt.colorbar(shrink=0.8, pad=0.1)
plt.subplots_adjust(right=1.2)
contour.set_clim(-0.000003, 0.000003)

plt.xlabel('x')
plt.ylabel('t')
plt.show()

bt=1800
plt.contourf(x_arr,y_arr,phi_ZF[bt,:,:].T)
plt.show()
plt.contourf(x_arr,y_arr,np.squeeze(phi_ST[bt,:,:].T))
plt.show()
plt.contourf(x_arr,y_arr,np.squeeze(phi_turb[bt,:,:].T))
plt.show()

def cal_x_derivative(phi,x_arr):
    dx_phi=np.zeros_like(phi)
    dx=x_arr[2]-x_arr[0]
    N_x=x_arr.size
    dx_phi[:,1:N_x-2,:]=(phi[:,2:N_x-1,:]-phi[:,0:N_x-3,:])/dx
    dx_phi[:,0,:]=(phi[:,1,:]-phi[:,N_x-1,:])/dx
    dx_phi[:,N_x-1,:]=(phi[:,0,:]-phi[:,N_x-2,:])/dx
    dx_phi[:,N_x-2,:]=(phi[:,N_x-1,:]-phi[:,N_x-3,:])/dx
    return dx_phi

def cal_y_derivative(phi,y_arr):
    dy_phi=np.zeros_like(phi)
    dy=y_arr[2]-y_arr[0]
    N_y=y_arr.size
    dy_phi[:,:,1:N_y-2]=(phi[:,:,2:N_y-1]-phi[:,:,0:N_y-3])/dy
    dy_phi[:,:,0]=(phi[:,:,1]-phi[:,:,N_y-1])/dy
    dy_phi[:,:,N_y-1]=(phi[:,:,0]-phi[:,:,N_y-2])/dy
    dy_phi[:,:,N_y-2]=(phi[:,:,N_y-1]-phi[:,:,N_y-3])/dy
    return dy_phi

def cal_energy(N,phi,x_arr,y_arr):
    phi_ZF=cal_ZF_comp(phi)
    N_ZF=cal_ZF_comp(N)
    phi_ST=cal_ST_comp(phi)
    N_ST=cal_ST_comp(N)

    phi_turb=phi-phi_ZF-phi_ST
    N_turb=N-N_ZF-N_ST

    dx_phi_ZF=cal_x_derivative(phi_ZF,x_arr)
    dx_phi_turb=cal_x_derivative(phi_turb,x_arr)
    dy_phi_turb=cal_y_derivative(phi_turb,y_arr)
    dy_phi_ST=cal_y_derivative(phi_ST,y_arr)

    E_turb=1/2*np.sum(np.sum(dx_phi_turb**2+dy_phi_turb**2+N_turb**2,axis=2),axis=1)
    E_ZF=1_2*np.sum(np.sum(dx_phi_ZF**2,axis=2),axis=1)
    E_ST=1_2*np.sum(np.sum(dy_phi_ST**2,axis=2),axis=1)
    return E_turb, E_ZF, E_ST

V_y_ZF=cal_x_derivative(phi_ZF[::3,::3,::3],x_arr[::3])

bt=500
plt.plot(x_arr[::3],V_y_ZF[bt,:,1],'b')

#bt=1000
plt.contourf(x_arr,y_arr,np.squeeze(phi_turb[bt,:,:].T),100,cmap='hot')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

dum=0
output_dir=dir+'/cont_phi_turb_Vy'

if dum==1:
    for bt in range(0,N_t,10):
        plt.subplots(figsize=(5, 3))
        cf=plt.contourf(x_arr,y_arr,np.squeeze(phi_turb[bt,:,:].T), levels=np.linspace(-7, 7, 100),cmap='hot')
        plt.plot(x_arr,V_y_ZF[bt,:,1]*2+20,'w')
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar(cf)
        cbar.set_ticks(np.linspace(-7, 7, 5))
        plt.title(f't= {t_arr[bt]}')
        if dum==1:
            filename = os.path.join(output_dir, f'cont_phi_Vy_{bt:03d}.png')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        plt.show()

plt.plot(x_arr,N_ZF[bt,:,1]*1.0-kap*x_arr+30.0)
plt.xlabel('x')
plt.ylabel('<N>')

N_evo.shape

phi_evo.shape

E_turb,E_ZF,E_ST=cal_energy(N_evo[::2,::2,::2],phi_evo[::2,::2,::2],x_arr[::2],y_arr[::2])

plt.semilogy(t_arr[::2],E_turb,'r',label='turb')
plt.semilogy(t_arr[::2],E_ZF,'b',label='ZF')
plt.semilogy(t_arr[::2],E_ST,'k',label='ST')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.xlabel('t')
plt.ylabel('Energy')
plt.show()

dx_phi_turb=cal_x_derivative(phi_turb[::2],x_arr[::2])
dy_phi_turb=cal_y_derivative(phi_turb[::2],y_arr[::2])
Gamma_x=-N_evo[::2]*dy_phi_turb
Gama_x_ZF=cal_ZF_comp(Gamma_x)

plt.subplots(figsize = (3, 5))
plt.contourf(x_arr,t_arr[::2],Gama_x_ZF[:,:,1], 50, cmap='hot')
plt.colorbar(shrink=0.8, pad=0.1)
plt.subplots_adjust(right=1.2)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

N_turb.shape

dx_phi_turb.shape

dy_phi_turb.shape

E_turb_last_t = E_turb[599:999]

E_turb_mean = np.mean(E_turb_last_t)

E_turb_mean

E_ZF_last_t = E_ZF[599:999]

E_ZF_mean = np.mean(E_ZF_last_t)

E_ZF_mean

E_ST_last_t = E_ST[599:999]

E_ST_mean = np.mean(E_ST_last_t)

E_ST_mean

nu1 = [0, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.2, 0.21, 0.22,0.23, 0.24, 0.25, 0.26, 0.27, 0.28,0.29]

E_turb1 = [3625.265, 7582.445, 6792.131, 5102.548, 3741.816, 2524.78, 1953.1153,1610.7582, 1188.957, 0.706408,0.10341, 0.07955, 0.077395,0.07652, 0.07587, 0.07535, 0.0749, 0.07457, 0.07429]
E_ZF1 = [482341.39, 5562.757, 3012.9859, 1545.5553, 1139.442, 726.0297, 424.158693450116, 174.895054902459, 101.9696, 3.28907, 6.84397, 1.3139, 4.09845, 2.62724, 1.8287, 1.29645, 9.294598,6.7285, 4.91471]
E_ST1 = [1133.48696, 4933.627, 4733.946, 4106.5877,2611.1628, 2513.7015, 1605.79791482747, 2326.25982668884, 1668.35387079238, 12.0718, 0.4392, 0.03314, 0.01288, 0.01084, 0.01002, 0.00937, 0.00879,0.008274, 0.007808]

plt.plot(nu1, E_turb1)
plt.xlabel('nu')
plt.ylabel('Energy_turb')
plt.show()

plt.plot(nu1, E_ZF1)
plt.xlabel('nu')
plt.ylabel('Energy_ZF')
plt.show()

plt.plot(nu1, E_ST1)
plt.xlabel('nu')
plt.ylabel('Energy_ST')
plt.show()

plt.plot(nu1, E_turb1, label='E_turb1')
plt.plot(nu1, E_ST1, label='E_ST1')
plt.plot(nu1, E_ZF1, label='E_ZF1')

# 軸ラベル
plt.xlabel('nu')
plt.ylabel('Energy')

# 凡例の追加
plt.legend()

# グラフの表示
plt.show()

