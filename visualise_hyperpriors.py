import matplotlib.pyplot as pl
import numpy as np
'''
rho_root  = np.concatenate((np.square(np.random.uniform(0,1,int(Ns/2))),1-np.square(np.random.uniform(0,1,int(Ns/2))) ))
sig0_root = np.random.uniform(0,1,Ns)
sigC_root = np.sqrt(rho_root)*sig0_root
sigR_root = np.sqrt(1-np.square(rho_root))*sig0_root
'''

'''
Ns = 100000
x = np.random.uniform(0,1,Ns)
y = np.random.uniform(0,1,Ns)

#y = np.random.uniform(0,np.sqrt(1-np.square(x)))

r = np.sqrt(np.square(x)+np.square(y))




rcirc = [ri for ri in r if ri<1]
circ_inds = [i for i,ri in enumerate(r) if ri<1 ]
rho = np.square(y)/np.square(r)
rhocirc = [rhoi for i,rhoi in enumerate(rho) if i in circ_inds]
#rho_out = [rhoi for i,rhoi in enumerate(rho) if i not in circ_inds]
#theta_out = [np.arctan(yi/xi) for xi,yi,i in zip(x,y,np.arange(Ns)) if i not in circ_inds]

theta  = np.random.uniform(0,np.pi/2,len(circ_inds))
#theta  = np.random.uniform(0,np.pi/2,Ns)
rhosin = np.square(np.sin(theta))
#pl.scatter(theta_out,rho_out,alpha=0.5,marker='o')
#pl.show()

#pl.hist(rho_out,bins=100,alpha=0.5,color='g')
#pl.hist(rho,bins=500,alpha=0.5,color='b')
#pl.hist(rho[:len(circ_inds)],bins=1000,alpha=0.5,color='g')
pl.hist(rhocirc,bins=1000,alpha=0.5,color='b')
pl.hist(rhosin,bins=1000,alpha=0.5,color='r')
pl.show()
err=1/0
'''

'''
#Analytic

R =1
Ns = 10000000
x = np.random.uniform(0,R,Ns)
y = np.random.uniform(0,R,Ns)
r = np.sqrt(np.square(x)+np.square(y))
pl.hist(r,bins=1000)
pl.show()

err=1/0
rho = np.linspace(0.01,0.99,Ns)

z = (rho*(1-rho))**0.5

x = z/4 + 1/(6*z)

#num = np.sqrt(rho)+np.sqrt(1-rho)
#den = rho*(1-rho)
pl.figure()
pl.plot(rho,x)#num/den)
pl.show()

err=1/0
'''

Ns = 10000000

#Default Prior
start = 1
dflabel   = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim \rm{Arcsine}(0,1)$"][start:]
altlabel  = [r"$\sigma_{\rm{Rel}} \sim U(0,1)$", r"$\sigma_{\rm{Common}} \sim U(0,1)$"][start:]
Alabel    = [r"$\sigma_{0} \sim U(0,1)$", r"$\sigma_{\rm{Rel}} \sim U(0,\sigma_0)$"][start:]
Blabel    = [r"$\sigma_{0} \sim U(0,1)$", r"$\sigma_{\rm{Common}} \sim U(0,\sigma_0)$"][start:]
Clabel    = [r"$\sigma_{0} \sim U(0,1)$", r"$\rho \sim U(0,1)$"][start:]
Dlabel    = [r"$\sigma^2_{\rm{Rel}} \sim U(0,1)$", r"$\sigma^2_{\rm{Common}} \sim U(0,1)$"][start:]
p = 2
Elabel    = [r"$\sqrt{\sigma_{\rm{Rel}}} \sim U(0,1)$", r"$\sqrt{\sigma_{\rm{Common}}} \sim U(0,1)$"][start:]
p = 1.5
Elabel    = [r"$\sigma^{1/%s}_{\rm{Rel}} \sim U(0,1)$"%p, r"$\sigma^{1/%s}_{\rm{Common}} \sim U(0,1)$"%p][start:]

rho_df  = np.random.beta(0.5,0.5,Ns)
sig0_df = np.ones(Ns)
sig0_df = np.random.uniform(0,1,Ns)
sigC_df = np.sqrt(rho_df)*sig0_df
sigR_df = np.sqrt(1-rho_df)*sig0_df

#Alt Prior
sigR_alt = np.random.uniform(0,1,Ns)
sigC_alt = np.random.uniform(0,1,Ns)
sig0_alt = (sigR_alt**2 + sigC_alt**2)**0.5
rho_alt  = (sigC_alt**2)/(sig0_alt**2)

#Prior A
sig0_A = np.ones(Ns)
sig0_A = np.random.uniform(0,1,Ns)
sigR_A = np.random.uniform(0,sig0_A)
sigC_A = (sig0_A**2 - sigR_A**2)**0.5
rho_A  = (sigC_A**2)/(sig0_A**2)

#Prior B
sig0_B = np.ones(Ns)
sig0_B = np.random.uniform(0,1,Ns)
sigC_B = np.random.uniform(0,sig0_B)
sigR_B = (sig0_B**2 - sigC_B**2)**0.5
rho_B  = (sigC_B**2)/(sig0_B**2)

#Prior C
sig0_C = np.ones(Ns)
sig0_C = np.random.uniform(0,1,Ns)
rho_C  = np.random.uniform(0,1,Ns)
sigC_C = np.sqrt(rho_C)*sig0_C
sigR_C = np.sqrt(1-rho_C)*sig0_C

#Prior D
sigR_D = np.sqrt(np.random.uniform(0,1,Ns))
sigC_D = np.sqrt(np.random.uniform(0,1,Ns))
sig0_D = np.sqrt(np.square(sigR_D)+np.square(sigC_D))
rho_D  = np.square(sigC_D)/np.square(sig0_D)

#Prior E
sigR_E = np.random.uniform(0,1,Ns)**p
sigC_E = np.random.uniform(0,1,Ns)**p
sig0_E = np.sqrt(np.square(sigR_E)+np.square(sigC_E))
rho_E  = np.square(sigC_E)/np.square(sig0_E)

Nb = 1000

def get_step(x,Nb):
	counts,bins = np.histogram(x,bins=Nb)
	counts = np.concatenate((counts[:1],counts))
	return bins,counts

FS = 14
lw = 3 ; alph=1
plotting_args = {'linewidth':lw, 'alpha':alph}
#objs    = ['df','alt','A','B','C']
objs    = ['df','A','B','C']
#objs    = ['df','A']
rhos    = [eval(f'rho_{x}') for x in objs]
sigRs   = [eval(f'sigR_{x}') for x in objs]
sigCs   = [eval(f'sigC_{x}') for x in objs]
sig0s   = [eval(f'sig0_{x}') for x in objs]

labs    = [eval(f'{x}label') for x in objs]
flags   = [1,0,0,0,1]

def rho_plot():
	zipper  = list(zip(rhos,[f'C{_}' for _ in range(len(rhos))],labs,flags))
	pl.figure()
	for rho,color,label,flag in zipper:#Get legend label order correct
		pl.step(*get_step(rho,Nb),color=color,label='\n'.join(label[flag:]),**plotting_args)
	for rho,color,label,flag in zipper[::-1]:#Get plot overlay order correct
		pl.step(*get_step(rho,Nb),color=color,label=None                   ,**plotting_args)
	pl.xlabel(r'$\rho$',fontsize=FS)
	pl.ylabel(r'No. of Samples',fontsize=FS)
	pl.legend(fontsize=FS,title='Hyperpriors',title_fontsize=FS)
	pl.tick_params(labelsize=FS)
	pl.xlim([0,1])
	pl.yscale('log')
	pl.tight_layout()
	pl.savefig('plots/hyperprior/rho.pdf',bbox_inches='tight')
	#pl.show()

def sig0_plot():
	zipper  = list(zip(sig0s,[f'C{_}' for _ in range(len(rhos))],labs))
	pl.figure()#figsize=(8,6))
	for x,color,label in zipper:#Get legend label order correct
		pl.plot(*get_step(x,Nb),color=color,label='\n'.join(label),**plotting_args)
	for x,color,label in zipper[::-1]:#Get plot overlay order correct
		pl.plot(*get_step(x,Nb),color=color,label=None,**plotting_args)
	pl.xlabel(r'$\sigma_{0}$ (mag)',fontsize=FS)
	pl.ylabel(r'No. of Samples',fontsize=FS)
	pl.legend(fontsize=FS,title='Hyperpriors',title_fontsize=FS)
	pl.tick_params(labelsize=FS)
	pl.xlim([0,1])
	pl.ylim([0,None])
	pl.tight_layout()
	pl.savefig('plots/hyperprior/sigma0.pdf',bbox_inches='tight')
	#pl.show()

def sigR_plot():
	zipper  = list(zip(sigRs,[f'C{_}' for _ in range(len(rhos))],labs,objs))
	pl.figure()#figsize=(8,6))
	for x,color,label,obj in zipper:#Get legend label order correct
		plotting_args['linewidth'] = lw*2 if obj in ['C'] else lw
		pl.plot(*get_step(x,Nb),color=color,label='\n'.join(label),**plotting_args)
	for x,color,label,obj in zipper[::-1]:#Get plot overlay order correct
		plotting_args['linewidth'] = lw*2 if obj in ['C'] else lw
		pl.plot(*get_step(x,Nb),color=color,label=None,**plotting_args)
	pl.xlabel(r'$\sigma_{\rm{Rel}}$ (mag)',fontsize=FS)
	pl.ylabel(r'No. of Samples',fontsize=FS)
	pl.legend(fontsize=FS,title='Hyperpriors',title_fontsize=FS)
	pl.tick_params(labelsize=FS)
	pl.xlim([0,1])
	pl.ylim([0,None])
	pl.tight_layout()
	pl.savefig('plots/hyperprior/sigmaRel.pdf',bbox_inches='tight')
	#pl.show()


def sigC_plot():
	zipper  = list(zip(sigCs,[f'C{_}' for _ in range(len(rhos))],labs))
	pl.figure()#figsize=(8,6))
	for x,color,label in zipper:#Get legend label order correct
		pl.plot(*get_step(x,Nb),color=color,label='\n'.join(label),**plotting_args)
	for x,color,label in zipper[::-1]:#Get plot overlay order correct
		pl.plot(*get_step(x,Nb),color=color,label=None,**plotting_args)
	pl.xlabel(r'$\sigma_{\rm{Common}}$ (mag)',fontsize=FS)
	pl.ylabel(r'No. of Samples',fontsize=FS)
	pl.legend(fontsize=FS,title='Hyperpriors',title_fontsize=FS)
	pl.tick_params(labelsize=FS)
	pl.xlim([0,1])
	pl.ylim([0,None])
	pl.tight_layout()
	pl.savefig('plots/hyperprior/sigmaCommon.pdf',bbox_inches='tight')
	#pl.show()

def combined_plot():
	fig,axs = pl.subplots(2,2,figsize=(8,4))#,sharex='col')
	ax = fig.axes
	parlabs = [r'$\rho$',r'$\sigma_0$ (mag)',r'$\sigma_{\rm{Rel}}$ (mag)',r'$\sigma_{\rm{Common}}$ (mag)']
	for ii,par in enumerate(['rho','sig0','sigR','sigC']):
		zipper  = list(zip(eval(f'{par}s'),[f'C{_}' for _ in range(len(rhos))],labs,objs))
		for x,color,label,obj in zipper:#Get legend label order correct
			#if par=='sigR': plotting_args['linewidth'] = lw*2 if obj in ['C'] else lw
			ax[ii].plot(*get_step(x,Nb),color=color,label='\n'.join(label),**plotting_args)
		for x,color,label,obj in zipper[::-1]:#Get plot overlay order correct
			#if par=='sigR': plotting_args['linewidth'] = lw*2 if obj in ['C'] else lw
			ax[ii].plot(*get_step(x,Nb),color=color,label=None,**plotting_args)

		if par=='rho':  ax[ii].set_yscale('log')
		else:
			if par!='sig0' or (par=='sig0' and 'alt' in objs):
				ax[ii].set_ylim([0,None])
			else:
				ax[ii].set_ylim([0,ax[ii].get_ylim()[1]*2])



		ax[ii].set_xlim([0,1])
		ax[ii].annotate(parlabs[ii],xy=(0.5,0.7),xycoords='axes fraction',ha='center',fontsize=FS+1,bbox=dict(facecolor='white',alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))#ax[ii].set_xlabel(parlabs[ii],fontsize=FS)
		ax[ii].tick_params(labelsize=FS)
		if par!='sig0':
			yy = ax[ii].get_ylim()
			ax[ii].set_ylim([yy[0],yy[1]*0.5])
		if ii<2:
			ax[ii].set_xticklabels([])
		ax[ii].set_yticklabels([])
		if ii==0:   ax[ii].legend(fontsize=FS,title='Hyperprior w/ \n'+r'$\sigma_0 \sim U(0,1)$',title_fontsize=FS,bbox_to_anchor=(2.1,0),loc='center left')
	fig.suptitle(r'Choice of Hyperpriors',fontsize=FS+2)
	fig.supylabel(r'Hyperprior Density',fontsize=FS)
	pl.tight_layout()
	fig.subplots_adjust(wspace=0.1,hspace=0.05)
	pl.savefig('plots/hyperprior/hyperpriors.pdf',bbox_inches='tight')
	#pl.show()

'''
rho_plot()
sig0_plot()
sigR_plot()
sigC_plot()
'''
combined_plot()



'''
#LKJ prior is equivalent to uniform prior on rho!
from cmdstanpy import CmdStanModel
import pickle
import pandas as pd
try:
	df = pd.read_csv('products/LKJ/lkj.csv').iloc[:100000]
except:
	generator_file = 'model_files/stan_files/LKJ/generator.stan'
	n_chains = 1 ; n_sampling = Ns ; n_warmup = 1000 ; Nc = 2
	model       = CmdStanModel(stan_file=generator_file)
	fit         = model.sample(data={'Nc':Nc},chains=n_chains, iter_sampling=n_sampling, iter_warmup = n_warmup, seed=42)
	df          = fit.draws_pd()
	df.to_csv('products/LKJ/lkj.csv')

pl.figure()
pl.hist(df['L_cint_eta[2,1]'],bins=100)
pl.show()
'''

'''
Lint_eta_samples = []
for ind in df.index:
	if ind%100==0:
		print (ind,'/',n_sampling)
	L_int_eta = np.zeros((Nc,Nc))
	for i in range(Nc):
		for j in range(Nc):
			L_int_eta[i,j] = df[f'L_cint_eta[{i+1},{j+1}]'].loc[ind]
	Lint_eta_samples.append(L_int_eta)
Lint_sims_path = 'products/LKJ/';filename = 'LKJsims.pkl'
with open(Lint_sims_path+filename,'wb') as f:
	pickle.dump(Lint_eta_samples,f)
'''
