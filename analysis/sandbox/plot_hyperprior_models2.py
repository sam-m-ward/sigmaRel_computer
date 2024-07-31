import sys
rootpath = '../../'
sys.path.append(rootpath+'model_files/')
from sigmaRel import *


FS=14
x = np.linspace(0,1,10000)
ps = [0.1,0.5,1,1.5,2,2.5,3,4,10]
#x = np.linspace(1,10,1000)
#ps = [-3,-2,-2.5,-1,-0.5,-0.1][::-1]
#ps = [0.0482,0.5,1,1.5,2,2.5,3,4,10]
#ps = [1.5,1.6,1.7,1.8,1.9,2]
#ps = np.linspace(1.9,2.1,9)

def get_y(x,p):
    return np.power(1-np.power(x,2/p),p/2)

#'''#ppaths
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
#pl.xlim([1,10])#pl.ylim([1,10])
pl.tick_params(labelsize=FS)
pl.savefig(rootpath+f"plots/hyperprior/p_sigRp_sigCp.pdf",bbox_inches='tight')
#pl.show() #; err=1/0
#'''

###
def get_ptheta1_usamp_sp(theta1,p):
    return p*np.cos(theta1)*np.power(np.sin(theta1),p-1)*np.sqrt(1+np.power(np.tan(theta1),4-2*p))

def get_ptheta1_usamp_thetap(theta1,p):
    return p*np.power(np.tan(theta1),p-1)*(1+np.square(np.tan(theta1)))/(1+np.power(np.tan(theta1),2*p))

###
def get_prho_usamp_sp(rho,p):
    return (p/2)*np.sqrt(np.power(rho,p-2)+np.power(1-rho,p-2))

def get_prho_usamp_thetap(rho,p):
    A = p/(2*np.sqrt(rho*(1-rho)))
    B = np.power(rho,(1+p)/2)*np.power(1-rho,(1-p)/2)
    C = np.power(rho,(1-p)/2)*np.power(1-rho,(1+p)/2)
    return A/(B+C)
###
def get_pthetap_usamp_sp(thetap,p):
    secsquare = np.square(1/np.cos(thetap))
    num = 1+np.power(np.tan(thetap),4/p-2)
    den = np.power(1+np.power(np.tan(thetap),2/p),p+2)
    return secsquare*np.sqrt(num/den)

def get_pthetap_usamp_thetap(thetap,p):
    return 2/np.pi*np.ones(len(thetap))


######
def get_pxp_usamp_sp(xp,p):
    return np.sqrt(1+np.power(xp,4/p-2)/np.power(1-np.power(xp,2/p),2-p))

def get_px1_usamp_sp(x1,p):
    return get_pxp_usamp_sp(x1**p,p)*p*np.power(x1,p-1)

def get_px1_usamp_sp_mod(x1,p):
    return np.power(x1,p-1)*np.sqrt(1+np.power(x1,4-2*p)/np.power(1-np.power(x1,2),2-p))

def get_pxq_usamp_sp(xq,p,q=1):
    return get_pxp_usamp_sp(np.power(xq,p/q),p)*(p/q)*np.power(xq,p/q-1)


def general(x,func,xlabel=r'$x$',ylabel=r'$P_p(x)$'):
    x = x[1:-1]
    pl.figure()
    for ip,p in enumerate(ps):
        px = func(x,p)
        pl.plot(x,px/(sum(px)*(x[1]-x[0])),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')
    pl.legend(framealpha=0.9,fontsize=FS,loc='upper center',ncol=3)#int(len(ps)/2))#bbox_to_anchor=(0.5,1)
    pl.xlabel(xlabel,fontsize=FS)
    pl.ylabel(ylabel,fontsize=FS)
    pl.ylim([0,3])
    pl.tick_params(labelsize=FS)
    #pl.plot(x,(1/(x))*1/sum(1/x)*1/(x[1]-x[0]))
    #pl.savefig(rootpath+f"plots/hyperprior/p_x.pdf",bbox_inches='tight')
    pl.show()
    err=1/0

#ps = [1e-5,0.01,0.2,1,2]
#general(x,get_pxp_usamp_sp)
#general(x,get_px1_usamp_sp,xlabel=r'$x_1$',ylabel=r'$P_p(x_1)$')
#general(x,get_pxq_usamp_sp)

#'''
def get_px1_usamp_sp_pzero(x1):
    rho = 1-np.square(x1)
    A = 1/np.square(rho) + 1/np.square(1-rho)
    return np.sqrt(A)*x1

def get_px1_usamp_thetap_pzero(x1):
    rho = 1-np.square(x1)
    return x1/(rho*(1-rho))

x = x[10:-10]
pl.figure()

px = get_px1_usamp_sp_pzero(x); #px = get_px1_usamp_sp_mod(x,0)
pl.plot(x,px/(sum(px)*(x[1]-x[0])),label=r'$p=0$ $s_p$-sampling',linewidth=3)
pq = get_px1_usamp_thetap_pzero(x)
pl.plot(x,px/(sum(pq)*(x[1]-x[0])),label=r'$p=0$ $\theta_p$-sampling',linewidth=3,linestyle='--')
pl.plot(x,(1/(x))*1/sum(1/x)*1/(x[1]-x[0]),label=r'$1/x_1$',linewidth=3)
#px = get_px1_usamp_sp(x,0.01)
#pl.plot(x,px/(sum(px)*(x[1]-x[0])),label=r'p=0.01 $s_p$-sampling',linewidth=3)
px = get_px1_usamp_sp(x,1)
pl.plot(x,px/(sum(px)*(x[1]-x[0])),label=r'Arcsine $\rho$-Hyperprior',linewidth=3)
px = get_px1_usamp_sp(x,2)
pl.plot(x,px/(sum(px)*(x[1]-x[0])),label=r'Uniform $\rho$-Hyperprior',linewidth=3)
pl.legend(framealpha=0.9,fontsize=FS,loc='upper center',ncol=1)#int(len(ps)/2))#bbox_to_anchor=(0.5,1)
pl.xlabel(r'$x_1$',fontsize=FS)
pl.ylabel(r'$P(x_1)$',fontsize=FS)
pl.ylim([0,10])
pl.xlim([0,1])
pl.tick_params(labelsize=FS)
pl.savefig(rootpath+f"plots/hyperprior/p_p0solns.pdf",bbox_inches='tight')
pl.show()
#'''

'''
from matplotlib import cm
import matplotlib.collections as mcoll
import matplotlib.path as mpath

Ns = 1000
Ns = 50000
x  = np.linspace(0,1,Ns)[1:-1]
px = get_px1_usamp_sp(x,1)

y  = np.sqrt(1-x**2)
py = get_px1_usamp_sp(y,1)

MAP = 'jet'
ff  = 2 ; Nslice = 10
NPOINTS = int(len(x)/ff)

fig = pl.figure()
ax  = fig.add_subplot(111)
ax.set_title(r'$P(\rho)$, $P(x)$ and $P(y)$ densities from Arcsine $\rho$-Hyperprior' +'\n'+r'for equal intervals of $y$ with increasing $\rho$',fontsize=FS)
cm  = pl.get_cmap(MAP)

pl.plot(x,1/(np.pi*np.sqrt(x*(1-x))),color='black',label=r'$z=\rho$',linestyle=':')
pl.legend(fontsize=FS,loc='center',bbox_to_anchor=(0.5,0.6),facecolor='white',framealpha=0.5,
            edgecolor='black')#, boxstyle='round,pad=0.3')
#bbox=dict(facecolor='white',alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))
pl.plot(1-x,px,linewidth=1,color='black')
pl.plot(x,px,linewidth=1,color='black')

ax.set_prop_cycle('color', [cm(1.0*_/(NPOINTS)) for _ in range(NPOINTS)])
for i in np.arange(len(x)):
    if i%int(len(x)/Nslice)>int(len(x)/(Nslice*ff)):
        pl.plot(x[i:i+2],px[i:i+2],linewidth=5)

ax.set_prop_cycle('color', [cm(1.0*_/(NPOINTS)) for _ in range(NPOINTS)])
for i in np.arange(len(x)):
    if i%int(len(x)/Nslice)>int(len(x)/(Nslice*ff)):
        pl.plot(1-y[i:i+2],py[i:i+2],linewidth=5)

pl.ylim([0,8])
pl.xlim([0,1])
drho=0.2
pl.annotate(r'$\rho \rightarrow 1$',xy=(0.75,0.7),xycoords='axes fraction', weight='bold',fontsize=FS+2,color='red',ha='left',bbox=dict(facecolor='white',alpha=0.5, edgecolor='black'))#, boxstyle='round,pad=0.3'))
pl.annotate(r'$\rightarrow$',xy=(0.425+drho,0.75),xycoords='axes fraction', weight='bold',fontsize=100,color='red',ha='left')

pl.annotate(r'$\rho \rightarrow 0$',xy=(0.25,0.7),xycoords='axes fraction', weight='bold',fontsize=FS+2,color='blue',ha='right',bbox=dict(facecolor='white',alpha=0.5, edgecolor='black'))#, boxstyle='round,pad=0.3'))
pl.annotate(r'$\leftarrow$',xy=(0.5-drho,0.75),xycoords='axes fraction', weight='bold',fontsize=100,color='blue',ha='right')

pl.annotate(r'$z = 1-x$',xy=(0.5,4), weight='bold',fontsize=FS+2,color='black',ha='center',va='center',bbox=dict(facecolor='white',alpha=0.5, edgecolor='black'))#, boxstyle='round,pad=0.3'))
pl.annotate(r'$z = y$',  xy=(0.5,3.25), weight='bold',fontsize=FS+2,color='black',ha='center',va='center',bbox=dict(facecolor='white',alpha=0.5, edgecolor='black'))#, boxstyle='round,pad=0.3'))

dx = 0.07
x0 = 0.385
pl.arrow(x0,4,dx-x0,0,head_width=0.5,head_length=0.05,overhang=1,length_includes_head=True)
x0 = 0.569
pl.arrow(x0,3.25,(1-dx-x0),0,head_width=0.5,head_length=0.05,overhang=1,length_includes_head=True)
pl.ylabel(r'$P(z)$',fontsize=FS)
pl.xlabel(r'$z$',fontsize=FS)
pl.tick_params(labelsize=FS)
pl.savefig('../plots/hyperprior/quad_rho_densities.pdf',bbox_inches='tight')
pl.show()
err=1/0
#'''




FS=20
uniform = 'thetap'
uniform = 'sp'


plot_coord = 'theta1'
plot_coord = 'rho'
plot_coord = 'thetap'

uniforms    = ['sp','thetap','both']
plot_coords = ['theta1','rho','thetap']

Ns = 10000
def get_coord(coord):
    if 'theta' in coord:
        return np.linspace(0,np.pi/2,Ns)[1:-1]
    if coord=='rho':
        return np.linspace(0,1,Ns)[1:-1]

coords = ['theta1','rho','thetap']
xparlabels = [r'$2\theta_1/\pi$',r'$\rho$',r'$2\theta_p/\pi$']
yparlabels = [r'$P_p(\theta_1)$',r'$P_p(\rho)$',r'$P(\theta_p)$']

for iu,uniform in enumerate(uniforms):
    fig,axs = pl.subplots(1,len(plot_coords),figsize=(8*len(plot_coords),8))
    ax = fig.axes
    if uniform not in ['both']:
        fig.suptitle(r'Hyperprior Densities from uniformly sampling %s'%dict(zip(uniforms,['$s_p$','$\\theta_p$']))[uniform]+'\n\n'*(iu==0),weight='bold',fontsize=FS+2)
    elif uniform=='both':
        fig.suptitle(r'Differences in these Hyperprior Densities'+'\n',weight='bold',fontsize=FS+2)

    for iax,plot_coord in enumerate(plot_coords):

        X = get_coord(plot_coord)

        for ip,p in enumerate(ps):
            if uniform=='sp':
                if plot_coord=='theta1':
                    pX = get_ptheta1_usamp_sp(X,p)
                if plot_coord=='rho':
                    pX = get_prho_usamp_sp(X,p)
                if plot_coord=='thetap':
                    pX = get_pthetap_usamp_sp(X,p)

            elif uniform=='thetap':
                if plot_coord=='theta1':
                    pX = get_ptheta1_usamp_thetap(X,p)
                if plot_coord=='rho':
                    pX = get_prho_usamp_thetap(X,p)
                if plot_coord=='thetap':
                    pX = get_pthetap_usamp_thetap(X,p)

            elif uniform=='both':
                if plot_coord=='theta1':
                    pX1 = get_ptheta1_usamp_sp(X,p)
                    pX2 = get_ptheta1_usamp_thetap(X,p)
                if plot_coord=='rho':
                    pX1 = get_prho_usamp_sp(X,p)
                    pX2 = get_prho_usamp_thetap(X,p)
                if plot_coord=='thetap':
                    pX1 = get_pthetap_usamp_sp(X,p)
                    pX2 = get_pthetap_usamp_thetap(X,p)
                pX = (pX1/(sum(pX1))-pX2/(sum(pX2)))
                #if p==1 and uniform=='both':
                #    print (pX)

            a = 2/np.pi if 'theta' in plot_coord else 1
            if uniform=='both':
                ax[iax].plot(X*a,pX/(X[1]-X[0]),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')
            else:
                ax[iax].plot(X*a,pX/(sum(pX)*(X[1]-X[0])),label=r'$p=%s$'%p,linewidth=3,linestyle='-' if p!=3 else '--')

            if iax==1:
                ax[iax].legend(framealpha=1,fontsize=FS-2,loc='upper center',ncol=len(ps),bbox_to_anchor=(0.5,1.1))
                #ax[iax].legend(framealpha=0.9,fontsize=FS,loc='upper center',ncol=3)#int(len(ps)/2))#bbox_to_anchor=(0.5,1)
            xlabel = xparlabels[coords.index(plot_coord)]
            ylabel = yparlabels[coords.index(plot_coord)]
            if uniform=='both':
                ylabel = ylabel.replace('$P','$\\Delta P')
            ax[iax].set_xlabel(xlabel,fontsize=FS)
            ax[iax].set_ylabel(ylabel,fontsize=FS)

            ax[iax].set_xlim([0,1])
            '''
            if uniform=='sp':
                if plot_coord=='theta1':
                    ax[iax].set_ylim([0,1.5])
                elif plot_coord=='rho':
                    ax[iax].set_ylim([0,3])
                elif plot_coord=='thetap':
                    ax[iax].set_ylim([0,3])
            elif uniform=='thetap':
                if plot_coord=='theta1':
                    ax[iax].set_ylim([0,8])
                elif plot_coord=='rho':
                    ax[iax].set_ylim([0,8])
            '''
            if uniform!='both':
                ax[iax].set_ylim([0,2.9])
            else:
                ax[iax].set_ylim([-1,1.1])
                ax[iax].set_yticks(np.arange(-1,1.01,1))

            ax[iax].tick_params(labelsize=FS)
    fig.subplots_adjust(wspace=0.2,hspace=0.2)
    pl.savefig(rootpath+f"plots/hyperprior/p_{plot_coords}_usamp{uniform}.pdf",bbox_inches='tight')
    #if uniform=='both':
    #    pl.show()
