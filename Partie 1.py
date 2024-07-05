import numpy as np
import matplotlib.pyplot as plt


dt = 0.0005
T = 1
num_times = int(T/dt)
V = np.zeros(num_times)
times = np.linspace(0,T,num_times) 
spikes = []
V_thr = -10.0 # mV
V_rest = -5.0 # mV
V_reset = -80.0 #
V_initial = -70.0 # mV
tau = 0.01 # 10ms


def model(V,I):
     dV = -(V-V_rest)+I
     return dV
 
    
#Somme de sinusoide (bruit)
def Ib(t,N):
    J, L=[], []
    for i in range(0,N):
        J.append(np.random.uniform(0,1))
        L.append(20*np.sin(J[i]*t))
    return sum(L)

def A(t):
    return np.random.uniform(-40,40)

#Fonction escalier
def H(t):
    t= t%0.2
    if t < 0.1:
        y=1
    else:
        y=-1
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


#Sinus
def K(t):
    return (1/(0.25*np.pi))*np.sin(t/0.25*np.pi)


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


def gauss_noise(t):
    return np.random.normal(loc=0, scale=1)

Y=[]
for i in range(len(times)):
    Y.append(50*G(times[i])+gauss_noise(times[i])) #Changer la fonction I pour étudier d'autres cas

plt.plot(times, Y)
#plt.title('Input I(t) qui stimule le neurone')
plt.xlabel('t [s]')
plt.ylabel('I(t)')
plt.grid(True)
plt.show()

counter_spike = 0
V[0]=V_initial
for i in range(len(times)-1):
    dV = model(V[i],Y[i])
    help = V[i]+dt*dV/tau
    #print("help:%f thr:%f self.V:%f   %f"%(help,V_thr,V[i],Y[i]))
    if help > V_thr:
        counter_spike += 1
        spikes.append(times[i])
        if counter_spike <20:
            print("#%d spike at time %f"%(counter_spike,times[i]))
        help = V_reset
    V[i+1] = help

val0 = 5.0
vals = [val0]*len(spikes)

    
plt.plot(times, V,'.-')
plt.plot(spikes,vals,'r*')
plt.xlabel('t [s]')
plt.ylabel('u(t)')
#plt.title('Potentiel u(t) du neurone stimulé par I(t)')
plt.show()


#Fonction fenetre glissante
def cellule_glissante(times,k,p):
    ## k: points in sliding window
    ## p: number of shift points of window
    print("number of times:%d  number of window points:%d number of shifts:%d"%(len(times),k,p))
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
#Retourne une liste de petits intervalles obtenus à partir d'un intervalle times


time_shift = 100*dt
window_duration = 0.1*T
k = int(window_duration/dt)
p = int(time_shift/dt)
#k, p = 100, 1
cell_potential = cellule_glissante(V, k, p)
cell_times = cellule_glissante(times, k, p)




S, count=[], []
counter = 0
for inter in cell_times:
    x=min(inter)
    y=max(inter)
    L=[]
    for t in spikes:
        if x <= t <= y:
            L.append(t)
            if counter<3:
                print("interval %f - %f  spike at %f"%(x,y,t),len(L))
    S.append(L)
    count.append(len(L))
    counter += 1


plt.plot(np.linspace(0, T-k*dt, len(count)), count)
plt.axis([0, T-k*dt, 0, max(count)*(1+0.1)])
plt.title("Nombre de spikes par intervalle")
plt.xlabel('t [s]')
#plt.show()


F=[]
tau=dt*k
for i in range(0,len(cell_times)):
    F.append(count[i]/tau)

plt.clf()
plt.plot(np.linspace(0, T-k*dt, len(count)), F)
#plt.title('Firing rate')
plt.axis([0,T-k*dt,0,max(F)*(1+0.1)])
plt.xlabel('t [s]')
plt.ylabel('F(t)')
plt.show()

m=np.mean(F)
v=np.var(F)
print("Moyenne du firing rate:%d  Variance du firing rate:%d"%(m,v))

#Etude des espaces entre les spikes
ISI=[]
for i in range(0,len(spikes)-1):
    ISI.append(spikes[i+1]-spikes[i])

DeltaT=0.005
x, l = [0], 0
while x[l] < max(ISI):
    l=l+1
    x.append(max(x)+DeltaT)
else:
    x.append(max(x)+DeltaT)

plt.hist(ISI, x)
plt.xlabel('t [s]')
#plt.title('ISI (temps entre les spikes)')
plt.show()



##Etude pour différentes valeurs de I constantes
I0=1
I=[]
x=I0
Imin = -10.0/dt
Imax = 40.0/dt
dI = 100.0
Imin_num = int(Imin/dI)
Imax_num = int(Imax/dI)
num_I = Imax_num - Imin_num
I = np.linspace(Imin,Imax,num_I)


def model2(V,I, V_rest):
     dV = -(V-V_rest)+I
     return dV
 
    
MF=[] #liste qui va stocker les moyennes de firing rate
counter_loop = 0
refractory_time = 0.001
for y in I:
    Y, Z=[], []
    for i in range(len(times)):
        Y.append(y+5*10**4*gauss_noise(times[i]))
        Z.append(np.random.normal(loc=-5, scale=10))
    spikes=[]
    V = np.zeros(num_times)
    counter_spike = 0
    V[0]=V_initial
    print("counter: %d Y:"%counter_loop,Y[0])
    refractory_counter = 0
    for i in range(len(times)-1):
        if refractory_counter>0:
            refractory_counter -= 1
            V[i+1]=V[i]
        else:
            dV = model2(V[i],Y[i],Z[i])
            help = V[i]+dt*dV/tau
            if counter_loop == num_I-1:
                print("help:%f thr:%f self.V:%f   y=%f    dV=%f"%(help,V_thr,V[i],Y[i],dV))
            if help > V_thr:
                counter_spike += 1
                spikes.append(times[i])
                if counter_loop == num_I-1:
                    print("#%d spike at time %f"%(counter_spike,times[i]))
                #if counter_spike <20:
                #    print("#%d spike at time %f"%(counter_spike,times[i]))
                help = V_reset
                refractory_counter = int(refractory_time/dt)
            V[i+1] = help
        #print("i+1=%d V=%f"%(i+1,V[i+1]))
    
    count=[]
    counter = 0
    for inter in cell_times:
        x=min(inter)
        y=max(inter)
        L=[] # spikes per time window
        for t in spikes:
            if x <= t <= y:
                L.append(t)
                #if counter<3:
                    #print("interval %f - %f  spike at %f"%(x,y,t),len(L))
        count.append(len(L)) # count over all time windows
        counter += 1
    print("counter:%d count=%d"%(counter_loop,np.mean(count)))
            
    #F=[]
    #tau=dt*k
    #for i in range(0,len(cell_times)):
    m=np.mean(count)/window_duration
    MF.append(m)
    
    if counter_loop == 300:
        val0 = 5.0
        vals = [val0]*len(spikes)
        plt.plot(times[0:50],V[0:50],'ko-')
        #plt.plot(spikes,vals,'r*')
        #plt.show()
    counter_loop += 1

plt.clf()
plt.plot(I, MF)
plt.title("Moyennes des Firing rate en fonction de l'input I")
plt.xlabel('I')
plt.ylabel('F(I)')
plt.show()
