#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


dt = 0.0005
T = 1
num_times = int(T/dt)
times = np.linspace(0,T,num_times)
M = 200 #Nombre de neurones
spikes = []
V_thr = np.zeros(M)-10.0+np.random.uniform(-15.0,15.0,M) # mV
V_rest = -5.0 # mV
V_reset = -80.0 #
V_initial = -70.0 # mV
S0 = -10.0
tau_s = 0.050
s_initial = S0*tau_s
tau = 0.01 # 10ms
Delta = 0.01
Delta_num = int(Delta/dt)
refractory_time = 0.001


D = np.identity(M)*0.0 #Matrice des coefficients des influences des neurones entre eux
for i in range(0,M):
    for j in range(0,M):
        if i == j:
            D[i,j]=0
        else:
            D[i,j]=0*np.random.normal(0.0,1.0)

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def model_synapse(s,spikenumber):
    help0 = D@spikenumber
    help1 = np.zeros(M)
    for n in range(M):
        help1[n] = -s[n]/tau_s+S0+help0[n]
    return help1

def model(V,I):
     dV = -(V-V_rest)+I
     return dV


L_MF=[] #liste qui va stocker les moyennes de firing rate
V = np.zeros((num_times,M))
s = np.zeros((num_times,M))
f = np.zeros((num_times,M))
F = np.zeros(num_times)

spikenumber = np.zeros(M,dtype=int)

spikes=[]
for j in range(0,M):
    V[0,j]=V_initial+np.random.uniform(-100.0,100.0)
    s[0,j]=s_initial
for i in range(M):
    spikes.append([])
counter_spike = np.zeros(M,dtype=int)
refractory_counter = np.zeros(M,dtype=int)
for i in range(len(times)-1):
    counter_spike[:] = 0
    for j in range(0,M):
        if refractory_counter[j]>0:
            refractory_counter[j] -= 1
            V[i+1,j]=V[i,j]
        else:
            I=s[i,j]
            dV = model(V[i,j],I)
            #print(np.shape(dV),np.shape(I))
            help = V[i,j]+dt*dV/tau
            #print("help:%f thr:%f self.V:%f   %f"%(help,V_thr,V[i],Y[i]))
            #print(help)
            if help > V_thr[j]:
                counter_spike[j] += 1
                spikes[j].append(times[i])
                #if counter_spike <20:
                    #print("#%d spike at time %f"%(counter_spike,times[i]))
                help = V_reset
                refractory_counter[j] = int(refractory_time/dt)
            V[i+1,j] = help
        #if counter_spike[j]>0:
        #    print("i=%f j:%d spike_counter:%d"%(times[i],j,counter_spike[j]))



    for j in range(0,M):
        if i>=Delta_num:
            x=float(i-Delta_num)*dt
            y=float(i)*dt
            #print("check whether spikes in previous windows: x=%f  y=%f"%(x,y))
            counter_spikes0=0
            for t in spikes[:][j]:
                #print("j=%d  x:%f y=%f t=%f "%(j,x,y,t))
                if x <= t <= y:
                    counter_spikes0 +=1
                    #print("##### j=%d   x:%f y=%f t=%f  counter=%d"%(j,x,y,t,counter_spikes0))
                    #quit()
            spikenumber[j]=counter_spikes0
            #if spikenumber[j]>0:
            #    print("spikenumber[%d,%d]=%d"%(i,j,spikenumber[j]))
    #print(spikenumber)
    f[i,:] = spikenumber/Delta
    F[i]=np.mean(spikenumber)
    ds = model_synapse(s[i,:],f[i,:])
    s[i+1,:]=s[i,:]+dt*ds[:]+np.sqrt(dt)*np.random.normal(0.0,1.0,M)


f[num_times-1,:]=f[num_times-2,:]
F[num_times-1]=F[num_times-2]
Vbar=np.mean(V,axis=1)

fig1 = plt.figure("time series")
plt.subplot(3,2,1)
plt.plot(times,V[:,0],'k')
plt.plot(times,V[:,25],'r')
plt.subplot(3,2,3)
plt.plot(times,s[:,0],'k')
plt.plot(times,s[:,25],'r')
plt.subplot(3,2,5)
plt.plot(times,f[:,0],'k')
plt.plot(times,f[:,25],'r')
plt.subplot(3,2,2)
plt.plot(times,F,'g')
plt.subplot(3,2,4)
plt.plot(times,Vbar,'g')

fig1 = plt.figure("network")
ax=plt.subplot(2,1,1)
im=ax.imshow(V.transpose(), interpolation='bilinear', cmap=cm.RdYlGn,
                    origin='lower', extent=[times[0], times[-1], 0, M],
                    #vmax=abs(Z).max(), vmin=-abs(Z).max()
                )
plt.colorbar(im)
forceAspect(ax,aspect=2)

ax=plt.subplot(2,1,2)
im=ax.imshow(f.transpose(), interpolation='bilinear', cmap=cm.RdYlGn,
                    origin='lower', extent=[times[0], times[-1], 0, M],
                    #vmax=abs(Z).max(), vmin=-abs(Z).max()
                )
plt.colorbar(im)
forceAspect(ax,aspect=2)
plt.show()

