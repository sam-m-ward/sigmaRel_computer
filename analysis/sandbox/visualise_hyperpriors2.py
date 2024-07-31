import sys
rootpath = '../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *

lw = 3 ; alph=1
plotting_args = {'linewidth':lw, 'alpha':alph}
def get_step(x,Nb):
	counts,bins = np.histogram(x,bins=Nb)
	counts = np.concatenate((counts[:1],counts))
	smooth_ind = 10
	return bins,counts/np.average(counts[int(len(counts)/2)-smooth_ind:int(len(counts)/2)+smooth_ind])

fac = 100#0
Ns = int(10000000/fac)
Nb = int(1000/fac)
'''#1/x on either axis
lnX = np.random.uniform(-100,2,Ns)
lnY = np.random.uniform(-100,2,Ns)
X = np.exp(lnX)
Y = np.exp(lnY)

#'''

'''#Uniform on either axis
X = np.random.uniform(0,1.5,Ns)
Y = np.random.uniform(0,1.5,Ns)
#'''

#
'''#P(x) = x
X = np.sqrt(np.random.uniform(0,1,Ns))
Y = np.sqrt(np.random.uniform(0,1,Ns))
#'''

'''#Arbitrary
aX = np.random.uniform(0,1,Ns)
aY = np.random.uniform(0,1,Ns)
X = 6*np.power(aX,3)+ 3.5*np.power(aX,3/2) -2*np.power(aX,-1/2)
Y = 6*np.power(aY,3)+ 3.5*np.power(aY,3/2) -2*np.power(aY,-1/2)
#'''

'''#Arbitrary2
aX = np.random.uniform(0,1,Ns)
aY = np.random.uniform(0,1,Ns)
X = np.sin(aX)+5*aX-np.log(aX+1)
Y = np.sin(aY)+5*aY-np.log(aY+1)
#'''

#'''#Arbitrary3
lnX = np.random.uniform(-20,2,Ns)
lnY = np.random.uniform(-20,2,Ns)
X = np.exp(lnX) + 1.3**(np.cos(lnX))-1/1.3
Y = np.exp(lnY) + 1.3**(np.cos(lnY))-1/1.3
#X = np.exp(lnX) + 1.3**(np.cos(np.exp(lnX)))-1/1.3
#Y = np.exp(lnY) + 1.3**(np.cos(np.exp(lnY)))-1/1.3
#'''

#rmin = 1#0#1.1
#rmax = 1.1#100#1.15

rmin = 1#05
rmax = 2#.5

def get_Xk_Yk(rmin=0.5,rmax=1):
	keep_i  = []

	for i,x,y in zip(np.arange(Ns),X,Y):
		r = np.sqrt(np.square(x)+np.square(y))
		if rmin<=r and r<=rmax and x>=0 and y>=0:
			keep_i.append(i)
	#keep_i = np.arange(Ns)#Keep all

	Xk = np.take(X,keep_i)
	Yk = np.take(Y,keep_i)

	Rk  = np.sqrt(np.square(Xk)+np.square(Yk))
	y1  = Yk/Rk
	rho = np.square(y1)
	r   = np.sqrt(Xk**2+Yk**2)
	return Xk, Yk, r, rho

r_pairs = [[1,1.1],[1.3,1.5],[1.8,1.9]]

r_pairs = [[0.01,0.05],[0.1,0.2],[0.3,0.4]]
r_pairs = [[0.5,1],[1.5,2],[2,2.5]]


pl.figure()
pl.scatter(X,Y,alpha=0.01,color=f'C0')
pl.show()

pl.figure()
for i,rp in enumerate(r_pairs):
	Xk,Yk,r,rho = get_Xk_Yk(rmin=rp[0],rmax=rp[1])
	pl.scatter(Xk,Yk,alpha=0.01,color=f'C{i}')
pl.show()


pl.figure()
for i,rp in enumerate(r_pairs):
	Xk,Yk,r,rho = get_Xk_Yk(rmin=rp[0],rmax=rp[1])
	pl.plot(*get_step(rho,Nb),color=f'C{i}',**plotting_args)
pl.xlabel(r'$\rho$')
pl.show()

pl.figure()
for i,rp in enumerate(r_pairs):
	Xk,Yk,r,rho = get_Xk_Yk(rmin=rp[0],rmax=rp[1])
	pl.plot(*get_step(r,Nb),color=f'C{i}',**plotting_args)
pl.xlabel(r'$r$')
pl.show()





Xk, Yk, r, rho = get_Xk_Yk(rmin,rmax)

#'''
pl.figure()
#pl.scatter(X,Y)
pl.scatter(Xk,Yk,alpha=0.01)
pl.show()
#'''

pl.figure()
pl.plot(*get_step(r,Nb),color='C0',**plotting_args)
rr = np.linspace(rmin,rmax,len(r))

#ff = 1/rr
#pl.plot(rr,ff/ff[int(len(ff)/2)],color='C1')

#rr = np.linspace(np.log(rmin),np.log(rmax),len(r))
#m = 0.1#(1-2*np.log10(np.e))
#print (m)
#ff = m*rr
#pl.plot(rr,ff/np.abs(ff[int(len(ff)/2)]),color='C1')

#ff = rr
#pl.plot(rr,ff/ff[int(len(ff)/2)],color='C1')

#ff = rr**3
#pl.plot(rr,ff/ff[int(len(ff)/2)],color='C1')

pl.show() #; err=1/0


Nr = len(rho)
rho2 = np.random.beta(0.5,0.5,Nr)



pl.figure()

pl.plot(*get_step(rho,Nb),color='C0',label=r'$P(z) \propto 1/z$',**plotting_args)
pl.plot(*get_step(rho2,Nb),color='C1',label=r'$\rho \sim \rm{Arcsine}(0,1)$',**plotting_args)

rr = np.linspace(0,1,Nr)[1:-1]
ff = 1/np.sqrt(rr*(1-rr))
pl.plot(rr,ff/ff[int(len(ff)/2)],color='C2',label='arcsine')

ff = np.sqrt(np.power(rr,-2) + np.power(1-rr,-2))
pl.plot(rr,ff/ff[int(len(ff)/2)],color='C3',label='p=0 sp')

ff = 1/(rr*(1-rr))
pl.plot(rr,ff/ff[int(len(ff)/2)],color='C4',label='p=0 thetap')

ff = np.sqrt(np.power(rr,0.5-2) + np.power(1-rr,0.5-2))
pl.plot(rr,ff/ff[int(len(ff)/2)],color='C5',label='p=0.5 sp')

pl.ylim([0,10])
pl.legend()
pl.show()
