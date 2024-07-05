import numpy as np
import matplotlib.pyplot as plt


dt = 0.005
T = 2.0
refractory_time = 0.001
num_times = int(T/dt)
times = np.linspace(0,T,num_times) 
spikes = []
M = 400 #Nombre de neurones
V_thr = np.zeros(M)-10.0+1.0*np.random.uniform(-5.0,5.0,M) # mV
V_rest = -5.0 # mV
V_reset = -80.0 #
V_initial = -70.0 # mV
tau = 0.01 # 10ms




# Fonction palier mais continue
def G(t):
    t= t%0.4
    if t < 0.1:
        y=1
    else:
        if t < 0.2:
            y=3-20*t
        else:
            if t < 0.3:
                y=-1
            else:
                y = -7+20*t
    return y

#Fonction plate avec des pointes
def P(t):
    t=t%0.2
    if t < 0.07:
        y=0
    else:
        if t > 0.13:
            y=0
        else:
            if 0.07 <= t <= 0.1:
                y=100*(t-0.07)
            else:
                if 0.1 < t <= 0.13:
                    y=100*(t-0.13)
    return y

def gauss_noise(t):
    return np.random.normal(loc=0, scale=1)


##Modeles d'input synaptique
alpha = 1
J=np.zeros(num_times)
J[0]=10000
for i in range(len(J)-1):
    J[i+1]=J[i]+(-alpha*J[i]+10**4)*dt +np.random.normal(loc=0, scale=10)

# plt.clf()
# plt.plot(times, J)
# plt.show()
# plt.hist(J)
# plt.show()



Y=[]
for i in range(len(times)):
    Y.append(50*G(times[i])+gauss_noise(times[i])) #Changer la fonction I pour étudier d'autres cas

plt.clf()
plt.plot(times, Y)
plt.title('Input I(t) qui stimule le neurone')
plt.xlabel('t [s]')
plt.ylabel('I(t)')
plt.grid(True)
plt.show()



def model(V,I):
     dV = -(V-V_rest)+I
     return dV
 

D = np.zeros((M,M)) #Matrice des coefficients des influences des neurones entre eux

for i in range(0,M):
    for j in range(0,M):
        if i == j:
            D[i,j]=0
        else:
            D[i,j]=np.random.normal(loc=0,scale=1)*0.0



V=np.zeros((num_times,M))
spikes=[]
for j in range(0,M):
  V[0,j]=V_initial+0*np.random.uniform(-100.0,100.0)
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
          dV = model(V[i,j],Y[i] + sum(D[:,j]*V[j,:]))
          #print(np.shape(dV),np.shape(I))
          help = V[i,j]+dt*dV/tau
          #print("help:%f thr:%f self.V:%f   %f"%(help,V_thr[j],V[i],Y[i]))
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
      
plt.clf()
for j in range(0,M):
    O=[j]*len(spikes[j])
    plt.plot(spikes[j], O, '.')
plt.xlabel('t [s]')
plt.ylabel('Indice m du neurone')
plt.show()




#Fonction fenetre glissante
def cellule_glissante(times,k,p):
    ## k: points in sliding window
    ## p: number of shift points of window
    #print("number of times:%d  number of window points:%d number of shifts:%d"%(len(times),k,p))
    J=[]
    l=0
    times_new = len(times)-k+1
    imax = np.floor(times_new/p).astype(int)
    for i in range(0,imax):
        L=[]
        for j in range(l,l+k):
            L.append(times[j])
        J.append(L)
        l=l+p
    return J



window_duration = 0.1
k = int(window_duration/dt)
time_shift = 0.1
p = int(time_shift/dt)
#k, p = 100, 1
#cell_potential = cellule_glissante(V, k, p)
cell_times = cellule_glissante(times, k, p)


MV = np.zeros((3,M))
L_F, L_ISI = [], []
for j in range(0,M):
  S, count, ISI =[], [], []   #S est la liste qui stocke les listes des spikes pour chaque intervalle, count compte le nombre de spikes a chaque intervalle
  R = spikes[j]
  counter = 0
  for inter in cell_times:
      x=min(inter)
      y=max(inter)
      L=[]
      for t in R:
          if x <= t <= y:
              L.append(t)
              if counter<3:
                  print("interval %f - %f  spike at %f"%(x,y,t),len(L))
      S.append(L)
      count.append(len(L))
      counter += 1
  
  F=[]
  tau=dt*k
  for i in range(0,len(cell_times)):
      F.append(count[i]/tau)    
  L_F.append(F)
  m=np.mean(F)
  v=np.var(F)
  for i in range(0,len(R)-1):
      ISI.append(R[i+1]-R[i])
  L_ISI.append(ISI)
  q = np.mean(ISI)
  MV[:,j]=((m,v,q))


plt.clf()
plt.plot(np.linspace(0, M-1, M), MV[0,:], marker='.', linestyle='-')
plt.title('Moyennes des firing rate sur les M neurones')
plt.grid(True)
plt.show()


plt.plot(np.linspace(0, M-1, M), MV[1,:], marker='.', linestyle='-')
plt.title('Variances des firing rate sur les M neurones')
plt.grid(True)
plt.show()


plt.plot(np.linspace(0, M-1, M), MV[2,:], marker='.', linestyle='-')
plt.title('Moyenne des ISI sur les M neurones')
plt.grid(True)
plt.show()




##Etude des firing rates selon différentes valeurs de I constants
refractory_time = 0.001
Imin = -1000.0#/dt
Imax = 4000.0#/dt
dI = 100.0#/dt
Imin_num = int(Imin/dI)
Imax_num = int(Imax/dI)
num_I = Imax_num - Imin_num
I = np.linspace(Imin,Imax,num_I)
V_thr=np.zeros(M)-10.0+np.random.uniform(-5.0,05.0,M) # mV

def model2(V,I, V_rest):
     dV = -(V-V_rest)+I
     return dV


L_MF=[]
for y in I:
    input = y.copy()
    Y = np.random.normal(y,200.0,(num_times,M))
    V=np.zeros((num_times,M))
    spikes=[]
    for j in range(0,M):
        V[0,j]=V_initial+np.random.uniform(-10.0,10.0)
    for i in range(M):
        spikes.append([])
    counter_spike = np.zeros(M,dtype=int)
    counter_spike_all = np.zeros(M,dtype=int)
    refractory_counter = np.zeros(M,dtype=int)
    #print("V_thr[0]:%f [50]:%f [100]:%f "%(V_thr[0],V_thr[50],V_thr[100]))
    #print("input variance:",np.var(Y[:,0]),np.var(Y[:,50]),np.var(Y[:,100]))
    for i in range(len(times)-1):
        counter_spike[:] = 0
        for j in range(0,M):
            if refractory_counter[j]>0:
                refractory_counter[j] -= 1
                dV = 0.0#V[i+1,j]=V[i,j]
            else:            
                dV = model(V[i,j],Y[i,j] + (D@V[i,:])[j])
                #if i>10 and i<15 and j==10:
                #    print("input=%f"%Y[i,j])
                #dV = model(V[i,j],y+0.1*10**4*np.random.normal(loc=0, scale=1))
                #print(np.shape(dV),np.shape(I))
            help = V[i,j]+dt*dV/tau
            #print("help:%f thr:%f self.V:%f   %f"%(help,V_thr[j],V[i],Y[i]))
            #print(help)
            if i>10 and i<15 and j==10:
                print("input=%f dV=%f  V=%f  help=%f"%(Y[i,j],dV,V[i,j],help))
            if help > V_thr[j]:
                counter_spike[j] += 1
                counter_spike_all[j] += 1
                spikes[j].append(times[i])
                #if counter_spike <20:
                    #print("#%d spike at time %f"%(counter_spike,times[i]))
                help = V_reset
                refractory_counter[j] = int(refractory_time/dt)
            V[i+1,j] = help
        #if counter_spike[j]>0:
        #    print("i=%f j:%d spike_counter:%d"%(times[i],j,counter_spike[j]))
    #plt.plot(times,V[:,0],'k')
    #plt.plot(times,V[:,50],'r')
    #plt.plot(times,V[:,100],'b')
    #plt.show()
    print("I=%f  spiking rate: %f"%(input,np.mean(counter_spike_all)/(num_times*tau)))    
    cell_potential = cellule_glissante(V, k, p)
    for j in range(0,M):
        S, count = [], []   #S est la liste qui stocke les listes des spikes pour chaque intervalle, count compte le nombre de spikes a chaque intervalle
        R = spikes[j]
        counter = 0
        for inter in cell_times:
            x=min(inter)
            y=max(inter)
            L=[]
            for t in R:
                if x <= t <= y:
                    L.append(t)
                    #if counter<3:
                        #print("interval %f - %f  spike at %f"%(x,y,t),len(L))
            S.append(L)
            count.append(len(L))
            counter += 1
        
        F=[]
        tau=dt*k
        for i in range(0,len(cell_times)):
            F.append(count[i]/tau)    
        m=np.mean(F)
    w=np.mean(F)
    L_MF.append(w)  
    #print("I=%f F:%f"%(input,w))

plt.plot(I, L_MF)
plt.xlabel('I')
plt.ylabel('Moyenne des F(I)')
plt.title("Moyennes des F(I) en fonction de l'input I")
plt.show()

