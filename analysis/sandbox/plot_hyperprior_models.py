import sys
rootpath = '../../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *
#from simulate_distances import *

def plot_rel_com_ratios(flip=False,alt=False):
    FS=14
    rel_rat = np.linspace(0,1,10000)
    com_rat = np.sqrt(1-np.square(rel_rat))
    fig = pl.figure()
    pl.plot(rel_rat,com_rat,color='black',linewidth=3)
    pl.plot(rel_rat,1-rel_rat,color='black',linewidth=3)
    pl.xlabel(r'$\sigma_{\rm{Rel}}/\sigma_0$',fontsize=FS)
    pl.ylabel(r'$\sigma_{\rm{Common}}/\sigma_0$',fontsize=FS)
    pl.gca().set_aspect('equal')
    major_ticks = np.arange(0, 1.01, 0.2)
    minor_ticks = np.arange(0, 1.01, 0.1)
    pl.xticks(major_ticks)
    pl.xticks(minor_ticks, minor=True)
    pl.yticks(major_ticks)
    pl.yticks(minor_ticks, minor=True)
    pl.grid(which='major', alpha=1,     color='black')
    pl.grid(which='minor', alpha=0.5,   color='black')
    if not alt:
        xs = np.arange(0,1.11,0.1)#-0.05
    if alt:
        xs = np.arange(0,1.11,0.1)
        xs = xs[::-1][1:]
    for ix,xi in enumerate(xs):
        if flip:
            if not alt:
                ii = np.argmin(np.abs(com_rat-xi))
            elif alt:
                ii = np.argmin(np.abs(rel_rat-xi))
            x = rel_rat[:ii+1] ; y = com_rat[:ii+1]
            pl.fill_betweenx(y,x,np.zeros(len(x)),alpha=0.5)
        else:
            if not alt:
                ii = np.argmin(np.abs(rel_rat-xi))
            elif alt:
                ii = np.argmin(np.abs(com_rat-xi))
            x = rel_rat[ii:] ; y = com_rat[ii:]
            pl.fill_between(x,np.zeros(len(y)),y,alpha=0.5)

    pl.xlim([0,1])
    pl.ylim([0,1])
    pl.tick_params(labelsize=FS)
    pl.savefig(rootpath+f"plots/hyperprior/rel_com_ratios{'_flip' if flip else ''}{'_alt' if alt else ''}.pdf",bbox_inches='tight')
    #pl.show()

'''
for flip in [False,True]:
    for alt in [False,True]:
        plot_rel_com_ratios(flip=flip,alt=alt)
#'''

FS=14
x = np.linspace(0,1,10000)

def get_y(x,p):
    return np.power(1-np.power(x,2/p),p/2)


def get_g(x,p):
    return np.sqrt(1+np.power(x,4/p-2)/np.power(1-np.power(x,2/p),2-p))

def get_f(x,p):
    return x/np.sqrt(np.square(x)+np.power(1-np.power(x,2/p),p))

def get_f_prime(f):
    df      = f[1:]-f[:-1]
    shifted = (df[1:]+df[:-1])/2#Shifted in x-axis

    starter_df = np.array([np.nan])
    ender_df   = np.array([np.nan])

    full_df = np.concatenate((starter_df,shifted,ender_df))
    return full_df#Start and end gradients are left blank

def get_dsdtheta(x,p):
    #Get Objs
    g = get_g(x,p)
    f = get_f(x,p)
    fprime = get_f_prime(f)

    #Combine
    num = g*np.sqrt(1-np.square(f))
    den = fprime
    dsdtheta = num/den

    #Return
    dsdtheta = dsdtheta[1:-1]#Removes first and last points where gradients are unknown
    return dsdtheta


def get_dsdtheta1(x,p):
    #Get Objs
    g = get_g(x,p)

    num = g*np.sqrt(1-np.power(x,2/p))*p
    den = np.power(x,1/p-1)

    #Combine
    dsdtheta1 = num/den

    #Return
    dsdtheta1 = dsdtheta1[1:-1]#Removes first and last points where gradients are unknown
    return dsdtheta1

def get_theta(x,p):
    sintheta = get_f(x,p)
    theta = np.arcsin(sintheta)
    return theta

def get_theta1(x,p):
    return np.arcsin(np.power(x,1/p))


def get_ptheta1(theta1,p):
    return np.power(np.sin(theta1),p-1) * np.cos(theta1) * np.sqrt(1+np.power(np.tan(theta1),4-2*p))#times p(sp) which we set to be uniform

def get_prho(rho,p):
    #return np.power(np.sin(theta1),p-2) * np.sqrt(1+np.power(np.tan(theta1),4-2*p))
    return np.sqrt(np.power(rho,p-2)+np.power(1-rho,p-2))

def get_norm(x,p):
    norm = (x[1]-x[0])*np.sum(get_g(x,p))
    return norm

def get_pthetap(thetap,p):
    secsquare = np.square(1/np.cos(thetap))
    num = 1+np.power(np.tan(thetap),4/p-2)
    den = np.power(1+np.power(np.tan(thetap),2/p),p+2)
    return secsquare*np.sqrt(num/den)

def get_ds_dthetap_full(xp,p):
    yp     = get_y(xp,p)
    rp     = np.sqrt(np.square(xp)+np.square(yp))
    sin_thetap = xp/rp
    cos_thetap = yp/rp
    Rtilde = xp/yp
    A = np.sqrt(1+np.power(Rtilde,4/p-2))
    B = np.square(rp)*cos_thetap
    C = rp - xp*sin_thetap*(1-np.power(Rtilde,2/p-2))

    return A*B/C


ps = [0.5,0.75,0.95,1,1.05,1.25,1.5,2]
ps = [0.1,0.5,0.95,1,1.05,2,3,10]
ps = [0.1,0.5,1,1.5,2,2.5,3,4,10]

fig = pl.figure()
for ip,p in enumerate(ps):
    pl.plot(x,get_y(x,p),color=f'C{ip}',linewidth=3,label=r'$p=%s$'%p)
pl.legend(framealpha=0.9,fontsize=FS)
pl.xlabel(r'$\sigma_{\rm{Rel}}^p/\sigma^p_0$',fontsize=FS)
pl.ylabel(r'$\sigma_{\rm{Common}}^p/\sigma^p_0$',fontsize=FS)
pl.gca().set_aspect('equal')
major_ticks = np.arange(0, 1.01, 0.2)
minor_ticks = np.arange(0, 1.01, 0.1)
pl.xticks(major_ticks)
pl.xticks(minor_ticks, minor=True)
pl.yticks(major_ticks)
pl.yticks(minor_ticks, minor=True)
pl.grid(which='major', alpha=1,     color='black')
pl.grid(which='minor', alpha=0.5,   color='black')
pl.xlim([0,1])
pl.ylim([0,1])
pl.tick_params(labelsize=FS)
pl.savefig(rootpath+f"plots/hyperprior/p_sigRp_sigCp.pdf",bbox_inches='tight')
pl.show()

def get_weighted_sum(px,x):
    dx_i  = x[1:]-x[:-1]
    px_av = (px[1:]+px[:-1])/2
    weighted_sum = sum(px_av*dx_i)
    return weighted_sum

theta1mode = False
rhomode    = False
thetapmode = False
pl.figure()
for ip,p in enumerate(ps):
    '''
    theta1mode = True
    theta1  = np.linspace(0,np.pi/2,len(x))[1:-1]
    ptheta1 = get_ptheta1(theta1,p)
    pl.plot(theta1*2/np.pi,ptheta1/(sum(ptheta1)*(theta1[1]-theta1[0])),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')
    #'''

    #'''
    thetapmode = True
    #xp = x[1:-1]
    #thetap = np.arcsin(xp/np.sqrt(np.square(xp)+np.power(1-np.power(xp,2/p),p)))
    #pthetap = get_ds_dthetap_full(xp,p)
    thetap  = np.linspace(0,np.pi/2,len(x))[1:-1]
    pthetap = get_pthetap(thetap,p)
    pl.plot(thetap*2/np.pi,pthetap/get_weighted_sum(pthetap,thetap),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')
    #'''

    '''
    rhomode=True
    #theta1  = np.linspace(0,np.pi/2,len(x))[1:-1]
    #ptheta1 = get_ptheta1(theta1,p)
    rho  = np.linspace(0,1,len(x))[1:-1]#np.square(np.sin(theta1))
    #prho = ptheta1/(2*np.sin(theta1)*np.cos(theta1))
    prho = get_prho(rho,p)
    pl.plot(rho,prho/get_weighted_sum(prho,rho),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')#Should be weighted sum, simple approx
    #'''

    '''
    xs    = x[1:-1]
    dfac = get_dsdtheta(x,p)
    pl.plot(xs,dfac,label=r'$p=%s$'%p)
    #'''

    #For thetap
    '''
    thetapmode = True
    dfac   = get_dsdtheta(x,p)
    theta  = get_theta(x,p)[1:-1]
    itheta_half = np.argmin(np.abs(theta-np.pi/4))
    theta = np.concatenate((theta[:itheta_half],np.pi/2-theta[:itheta_half][::-1]))
    dfac  = np.concatenate((dfac[:itheta_half],dfac[:itheta_half][::-1]))
    pl.plot(theta*2/np.pi,dfac/get_weighted_sum(dfac,theta),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')
    #'''

    '''#For theta1
    dfac1 = get_dsdtheta1(x,p)
    theta1 = get_theta1(x,p)[1:-1]
    itheta_half = np.argmin(np.abs(theta1-np.pi/4))
    theta1 = np.concatenate((theta1[:itheta_half],np.pi/2-theta1[:itheta_half][::-1]))
    dfac1   = np.concatenate((dfac1[:itheta_half],dfac1[:itheta_half][::-1]))
    if p<10 and p>0.1:
        pl.plot(theta1,dfac1/get_norm(x[1:-1],p),label=r'$p=%s$'%p)
    #'''
pl.legend(framealpha=0.9,fontsize=FS,loc='upper center',ncol=3)#int(len(ps)/2))#bbox_to_anchor=(0.5,1)
if theta1mode:
    pl.xlabel(r'$2\theta_1/\pi$',fontsize=FS)
    pl.ylabel(r'$P_p(\theta_1)$',fontsize=FS)
if thetapmode:
    pl.xlabel(r'$2\theta_p/\pi$',fontsize=FS)
    pl.ylabel(r'$P_p(\theta_p)$',fontsize=FS)
if rhomode:
    pl.xlabel(r'$\rho$',fontsize=FS)
    pl.ylabel(r'$P_p(\rho)$',fontsize=FS)
if theta1mode or rhomode or thetapmode:
    pl.xlim([0,1])
    pl.ylim([0,3])
pl.tick_params(labelsize=FS)
if theta1mode:
    pl.savefig(rootpath+f"plots/hyperprior/p_theta1.pdf",bbox_inches='tight')
if thetapmode:
    pl.savefig(rootpath+f"plots/hyperprior/p_thetap.pdf",bbox_inches='tight')
if rhomode:
    pl.savefig(rootpath+f"plots/hyperprior/p_rho.pdf",bbox_inches='tight')
pl.show()
