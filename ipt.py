import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import cmath
import math
from scipy import signal
from scipy import integrate

one=complex(1.0,0.0)
ii=complex(0.0,1.0)
zero=complex(0.0,0.0)


#------------------------------------------------------
#CONVOLUTION FUNCTION
#------------------------------------------------------
#          infty
#y3(t)=int^      y1(tau)*y2(t-tau)
#         -infty

def convolve(y1,y2):
  n=len(y1)
  y1p=np.pad(y1,(n,n),'constant',constant_values=(0,0))
  y2p=np.pad(y2,(n,n),'constant',constant_values=(0,0))
  y3p=signal.fftconvolve(y1p,y2p,mode='same')
  y3=y3p[n:2*n]*step_size
  return y3
#------------------------------------------------------
  

#------------------------------------------------------
#Kramers-Kronig transform
#------------------------------------------------------
def kkt(w,rho,ss):
  wl=w+np.ones(len(w))*ss/2.0
  owl=1.0/wl
  return(convolve(rho,owl))
#------------------------------------------------------

#------------------------------------------------------
def gauss(z,t):
  G=-ii*np.sqrt(np.pi)*special.wofz(z/t)/t
  return G
#------------------------------------------------------

#------------------------------------------------------
def fermi_integral(E, fermi, T):
    if T > 0.0:
      if E < fermi:
        return 1.0 / (1.0 + np.exp((E - fermi)/T))
      else:
        return np.exp(-(E - fermi) /(T))/(1.0 + np.exp(-(E - fermi)/T))
    elif T==0.0:
      if E < fermi:
        return 1.0
      elif E==fermi:
        return 0.5
      else:
        return 0.0
#------------------------------------------------------

#------------------------------------------------------
#SELF-ENERGY
#------------------------------------------------------
def selfenergy(w,rho,ferm,ss): 
  F1=np.flip(rho)*ferm
  F2=rho*ferm
  chi2=convolve(F1,F2)
  chi1=convolve(np.flip(F2),np.flip(F1))
  rho_SE=U**2*(convolve(np.flip(F1),chi1)+convolve(F2,chi2))
  #Impose p-h symmetry
  rho_SE=0.5*(rho_SE+np.flip(rho_SE))
  Re_SE=kkt(w,rho_SE,ss)
  SelfE=Re_SE-ii*np.pi*rho_SE
  return SelfE

#------------------------------------------------------#
#------------------------------------------------------#
#	MAIN PROGRAM STARTS HERE
#------------------------------------------------------#
#------------------------------------------------------#

t=1.0
U=6.5
eta=1.0e-03
mu0=0.0
init=0
sol='M'
Temp=0.0

#Read input parameters from file par.dat
f=open('par.dat','r')
content=f.readlines()
nums=content[1].split()
wm=float(nums[0]); step_size=float(nums[1])
nums=content[3].split()
U=float(nums[0]); init=int(nums[1]); sol=nums[2]; mu0=float(nums[3])
Temp=float(nums[4])

#Frequency Grid - Uniform from -10 to 10 with step_size=0.001
#wm=10.0
#step_size=1.0e-03
w=np.arange(start=-wm,stop=wm+step_size,step=step_size,dtype=float)
nw=len(w)

fermi=0.0
ferm=np.zeros(nw)
for i in range(nw):
  ferm[i]=fermi_integral(w[i],fermi,Temp)


print('wm=',wm,' step_size=',step_size,' nw=',nw)
print('U=',U,' init=',init,' sol=',sol)


#Prepare the initial Green's functions, self-energy and hybridization
#Start Prepare
z=w*one
if(init==0):
  if(sol=='M'):
    SelfE=-ii*eta*np.ones(nw)
  else:
    SelfE=U/2  #U**2*one/(4.0*(z+ii*10.0*eta))
  gamma=z-SelfE
  G_new=gauss(gamma,t)
  hyb=gamma-one/G_new
else:
  se_data = np.loadtxt("SE.dat", dtype=float,skiprows=1)
  SelfE=se_data[:,1] -ii*se_data[:,2]
  gamma=z-SelfE
  G_new=gauss(gamma,t)
  hyb=gamma-one/G_new

G_host=one/(z+mu0-hyb) #Initial Hartree corrected Green's function
G_old=G_new
#End Prepare


plt.title("Host Green's function",fontsize=20,color='black',fontweight='bold')
plt.plot(w,-G_host.imag/np.pi,lw=2,color='black',label='Host-DoS')
plt.plot(w,G_host.real,lw=2,color='red',label='Real(Host-G)')
plt.legend()
plt.xlabel('Frequency',fontsize=15,fontweight='bold')
plt.show()

#------------------------------------------------------
#               DMFT Iteration
#------------------------------------------------------
conv=1.0
iter=1
n=0.5
n0=0.5
ef=-U/2
while conv > 0.001 and iter < 101:

  print('-----------------------')
  print('Iteration number=',iter)

  G_host=one/(z+mu0-hyb) #Hartree corrected Host Green's function
  rho=-G_host.imag/np.pi

  A = n*(1-n)/(n0*(1-n0)) 
  B = U*(1-n)+ef+mu0/(n0*(1-n0)*U**2)
  
  SelfE_2=selfenergy(w,rho,ferm,step_size)

  SelfE = (A*SelfE_2/(1-B*SelfE_2)) #IPT Ansatz from Kajueter & Kotlier PRL 1996

  gamma=w-SelfE
  G_new=gauss(gamma,t)

  hyb=gamma-one/G_new

  conv=np.sum(abs(G_new-G_old))/np.sum(abs(G_new))
  norm=integrate.trapz(-G_new.imag/np.pi,w,dx=step_size)
  print('Norm=',norm,' Conv=',conv)

#  plt.plot(w,-G_old.imag/np.pi,lw=2,color='black',label='Old-DoS')
#  plt.plot(w,-G_new.imag/np.pi,lw=2,color='red',label='New-DoS')
#  plt.legend()
#  plt.xlabel('Frequency',fontsize=15,fontweight='bold')
#  plt.show()

  G_old=G_new
  iter+=1

#------------------------------------------------------
plt.title("Spectral function-Kajueter IPT",fontsize=20,color='black',fontweight='bold')
#plt.plot(w,-G_host.imag/np.pi,lw=2,color='black',label='Host-DoS')
plt.plot(w,-G_new.imag/np.pi,lw=2,color='red',label='New-DoS')
plt.legend()
plt.xlabel('Frequency',fontsize=15,fontweight='bold')
plt.show()

hstr=' w   Re(hyb)    Im(hyb)'
h2D=np.vstack((w, hyb.real, hyb.imag)).T
np.savetxt('hyb.dat', h2D, delimiter=' ', fmt='%e',header=hstr)

hstr=' w   Re(SE)    -Im(SE)'
h2D=np.vstack((w, SelfE.real, -SelfE.imag)).T
np.savetxt('SE.dat', h2D, delimiter=' ', fmt='%e',header=hstr)
