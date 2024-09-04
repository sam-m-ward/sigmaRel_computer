"""
Residual Model

Module containing ResModelLoader class
ResModelLoader class used to perform multiple linear regression between parameters

Contains:
--------------------
ResModelLoader class:
	inputs: dfmus,rootpath='./',samplename='full',FS=18,verbose=True

	Methods are:
		plot_res_pairs(px='theta',py='mu',show=False,save=True)
		print_table(PARS=['mu','AV','theta','RV'],verbose=False,returner=False)
		plot_res(px='theta',py='mu',zerox=True,zeroy=True,show=False,save=True,markersize=10,capsize=2,alpha=0.9,elw=1.5,mew=1.5,text_index=3,annotate_mode=False,args_legend={'loc':'upper left','ncols':2,'bbox_to_anchor':(1,1.02)})
		get_data()
		get_inits()
		prepare_stan()
		reorder_pars(pars)
		get_pars(cosmo=False)
		get_modelkey()
		sample_posterior(py='mu',pxs=['theta'], alpha_prior=10, beta_prior=10, sigint_prior=10, alpha_const=0, sigint_const=0, alpha=False, beta=None, overwrite=True, run=True, common_beta=None)
		get_parlabels()
		get_res_lines(stan_data)
		plot_posterior_samples(FS=None,paperstyle=True,quick=True,save=True,show=False, returner=False)
--------------------

Written by Sam M. Ward: smw92@cam.ac.uk
"""
import arviz as az
from cmdstanpy import CmdStanModel
import json, os, pickle, re, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from plotting_script import *
from matplotlib import container


class ResModelLoader:
	def __init__(self,dfmus,rootpath='./',samplename='full',FS=18):
		"""
		Initialisation

		Parameters
		----------
		dfmus : pandas df
			dataframe of parameters with columns Galaxy, SN, {PAR}_samps

		rootpath : str
			path/to/sigmaRel/rootpath

		samplename : str (optional; default='full')
			name of multi-galaxy sample of siblings

		FS : float (optional; default=18)
			fontsize for plots
		"""
		#Data
		self.dfmus       = dfmus
		self.samplename  = samplename

		#Paths
		self.rootpath    = rootpath
		self.modelpath   = self.rootpath  + 'model_files/'
		self.stanpath    = self.modelpath + 'stan_files/linear_reg/'
		self.plotpath    = self.rootpath  + 'plots/linear_reg/'
		self.productpath = self.rootpath  + 'products/linear_reg/'

		self.FS = FS

		#Create residual samples
		self.PARS = [par for par in ['mu','theta','etaAV','AV','RV'] if f'{par}_samps' in self.dfmus.columns]
		def mean_mapper(x):
			mean_samps = x.mean(axis=0)
			x = x.apply(lambda x: x-mean_samps)
			return x
		for PAR in self.PARS:
			self.dfmus[f'{PAR}res_samps'] = self.dfmus.groupby('Galaxy')[f'{PAR}_samps'].transform(lambda x: mean_mapper(x))
			self.dfmus[f'{PAR}_respoint'] = self.dfmus[f'{PAR}res_samps'].apply(np.median)
			self.dfmus[f'{PAR}_reserr']   = self.dfmus[f'{PAR}res_samps'].apply(np.std)
		#print (self.dfmus[[col for col in dfmus.columns if 'respoint' in col or 'reserr' in col] + ['Galaxy']])

		#Input Posterior Parameters
		self.master_parnames  = ['mu','theta','etaAV','AV','RV']
		self.master_parlabels = ['$\\Delta \\mu$ (mag)','$\\Delta \\theta$','$\\Delta \\eta_{A_V}$','$\\Delta A_V$ (mag)','$\\Delta \\R_V$']
		self.master_bounds    = [[None,None],[None,None],[None,None],[0,None],[None,None]]
		self.plabdict         = {'theta':'\\theta','AV':'A_{V}','etaAV':'\\eta_{A_V}','mu':'\\mu'}

		#Posterior Configuration
		self.n_warmup   = 10000
		self.n_sampling = 5000
		self.n_chains   = 4
		self.n_thin     = 1000

	def plot_res_pairs(self,px='theta',py='mu',show=False,save=True):
		"""
		Plot Residuals for Sibling Pairs

		Plots the residuals of two parameters against one another

		Parameters
		----------
		px,py : strs (optional; default='theta', 'mu')
			the x and y parameters, respectively

		show, save: bools (optional; default=False,True)
			whether to show/save plot

		End Product
		----------
		Plot of residuals
		"""
		dfmus = self.dfmus.copy()
		dfmus = dfmus.iloc[::2].copy()
		colours = [f'C{s%10}' for s in range(dfmus['Galaxy'].nunique())]
		markers = ['o','s','p','^','x','P','d','*']
		markers = [markers[int(ig%len(markers))] for ig in range(dfmus['Galaxy'].nunique())]
		pl.figure()
		for ig,g in enumerate(dfmus['Galaxy'].unique()):
			xm = dfmus[dfmus['Galaxy']==g][f'{px}res_samps'].apply(np.median)
			ym = dfmus[dfmus['Galaxy']==g][f'{py}res_samps'].apply(np.median)
			xe = dfmus[dfmus['Galaxy']==g][f'{px}res_samps'].apply(np.std)
			ye = dfmus[dfmus['Galaxy']==g][f'{py}res_samps'].apply(np.std)
			#if z1m.values[0]<0: z1m *= -1; z2m *= -1
			pl.errorbar(xm,ym,xe,ye,linestyle='none',marker=markers[ig],color=colours[ig])
		pl.xlabel(self.master_parlabels[self.master_parnames.index(px)], fontsize=self.FS)
		pl.ylabel(self.master_parlabels[self.master_parnames.index(py)], fontsize=self.FS)
		pl.tight_layout()
		pl.tick_params(labelsize=self.FS)
		if save:
			pl.savefig(self.plotpath+f'respairplot_{px}_{py}_samp{self.samplename}.pdf',bbox_inches="tight")
		if show:
			pl.show()

	def print_table(self, PARS=['mu','AV','theta','RV'],verbose=False,returner=False):
		"""
		Print Tables

		Method to take posterior chains and print summaries

		Parameters
		----------
		PARS: list (optional; default=['mu','AV','theta','RV'])
			total list that is then trimmed to match that in dfmus

		verbose : bool (optional; default=False)
			option to print out summaries for each plot

		returner: bool (optional; default=False)
			if True, return list of summaries

		End Product(s)
		----------
		Prints out string
		"""
		all_PARS   = copy.deepcopy(PARS)
		pPARS      = ['mu','AV','theta','RV']
		pars       = [p for p in self.master_parnames if p in PARS]
		samp_cols  = [f'{p}_samps' for p in pars if f'{p}_samps' in self.dfmus.columns]

		keep        = [i for i,p in enumerate(self.master_parnames) if p in pars]
		parlabels   = [x for i,x in enumerate(self.master_parlabels) if i in keep]
		bounds      = [x for i,x in enumerate(self.master_bounds)    if i in keep]

		plotting_parameters = {'FS':self.FS,'paperstyle':False,'quick':True,'save':False,'show':False}
		lines = [] ; old_g = self.dfmus['Galaxy'].iloc[0]
		for isn,sn in enumerate(self.dfmus.index):
			NSN_g = str(self.dfmus[self.dfmus.Galaxy==self.dfmus.Galaxy.loc[sn]].shape[0])
			if self.dfmus['Galaxy'].loc[sn]!=old_g:
				lines.append('\midrule')
				old_g = self.dfmus['Galaxy'].loc[sn]
				new_g = True
			else:
				new_g = False if isn!=0 else True
			try:
				samples = {par:self.dfmus[samp_col].loc[sn] for par,samp_col in zip(pars,samp_cols)}
				try:	Rhats   = {par:self.dfmus[f'{par}_Rhats'].loc[sn] for par in pars}
				except:	Rhats   = {par:-1.0 for par in pars}
				postplot = POSTERIOR_PLOTTER(samples, pars, parlabels, bounds, Rhats, plotting_parameters)
				Summary_Strs = postplot.corner_plot(verbose=verbose)#Table Summary
			except:
				Summary_Strs = {par:f"${self.dfmus[f'{par}s'].loc[sn]:.2f}\pm{self.dfmus[f'{par}_errs'].loc[sn]:.2f}$" for par in pars}
			#Create Line
			line = self.dfmus['SN'].loc[sn].replace('_','\_') + ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,self.dfmus['Galaxy'].loc[sn].astype(str)) if new_g else '')
			for par in all_PARS:
				if par in pars:
					line += ' & ' + Summary_Strs[par]
				elif par not in pars and par not in pPARS:
					if par in ['zcmb_hats','zHD_hats']:
						line += ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,str(round(self.dfmus[par].loc[sn],6))) if new_g else '')#int(-np.log10(self.dfmus['zhelio_errs'].loc[sn]))))
					elif par in ['cosmo_sample']:
						mark  = {True:"\\cmark",False:"\\xmark"}[self.dfmus[par].loc[sn]]
						line += ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,{True:"\\cmark",False:"\\xmark"}[self.dfmus[par].loc[sn]]) if new_g else '')
					else:
						line += ' & ' + (r"\multirow{%s}{*}{%s}"%(NSN_g,self.dfmus[par].astype(str).loc[sn]) if new_g else '')
			line +=  ' \\\\ '
			lines.append(line)
		if returner:
			return lines
		else:
			print ('\n'.join(lines))

	def plot_res(self,px='theta',py='mu',zerox=True,zeroy=True,show=False,save=True,markersize=10,capsize=2,alpha=0.9,elw=1.5,mew=1.5,text_index=3,annotate_mode=False,args_legend={'loc':'upper left','ncols':2,'bbox_to_anchor':(1,1.02)}):
		"""
		Plot Residuals

		Plots the residuals of two parameters against one another

		Parameters
		----------
		px,py : strs (optional; default='theta', 'mu')
			the x and y parameters, respectively

		zerox,zeroy : bools (optional; default=True,True)
			choice to subtract off galaxy mean

		show, save: bools (optional; default=False,True)
			whether to show/save plot

		markersize,capsize,alpha,elw,mew: float,float,bool,foat (optional;default=14,5,0.9,3,3)
			size of markers and cap., alpha, elinewidth and markeredgewidth

		text_index, annotate_mode, args_legend: int, str (optional; default=3, 'legend' or 'text', {'loc':'upper left','ncols':2,'bbox_to_anchor':(1,1.02)})
			for example if 'ZTF18abcxyz', specifying 3 returns label '18abcxyz'
			if legend annotate, put SN names in legend, otherwise, put SN name next to data points
			legend details

		End Product
		----------
		Plot of residuals
		"""
		dfmus   = self.dfmus.copy()
		colours = [f'C{s%10}' for s in range(dfmus['Galaxy'].nunique())]
		markers = ['o','s','p','^','x','P','d','*']
		markers = [markers[int(ig%len(markers))] for ig in range(dfmus['Galaxy'].nunique())]
		PARS    = [px,py]

		#Chromatic parameter posteriors bunching up
		if [pp for pp in ['AV','RV'] if pp in PARS]!=[]:#To get 68,95 intervals when required
			PPARS = [pp for pp in ['AV','RV'] if pp in PARS]
			lines = self.print_table(PARS=PPARS,returner=True)
			lines = [ll.split('\\\\')[0] for ll in lines if ll!='\\midrule']
			dfsummaries = pd.DataFrame([re.sub(r'\s+','',ll).split('&') for ll in lines],columns=['SN','GalaxyID']+PPARS)
			dfsummaries['SN'] = dfsummaries['SN'].apply(lambda x: x.replace('\_','_'))
			dfsummaries[[f'{pp}bunched' for pp in PPARS]] = dfsummaries[PPARS].apply(lambda x: ['\pm' not in xi for xi in x],axis=1,result_type='expand')
			for pp in PPARS:
				dfsummaries.loc[dfsummaries[f'{pp}bunched'],f'{pp}68'] = dfsummaries.loc[dfsummaries[f'{pp}bunched'],pp].apply(lambda x: x.split('(')[0].split('$')[1][1:])
				dfsummaries.loc[dfsummaries[f'{pp}bunched'],f'{pp}95'] = dfsummaries.loc[dfsummaries[f'{pp}bunched'],pp].apply(lambda x: x.split('(')[1].split(')')[0])
				dfsummaries.loc[dfsummaries[f'{pp}bunched'],f'{pp}bunchindex'] = dfsummaries.loc[dfsummaries[f'{pp}bunched'],pp].apply(lambda x: {'<':0,'>':1}[x[1]])
		else:
			dfsummaries = pd.DataFrame()

		#Axis labels
		def get_axlab(pz,zeroz):
			if pz=='mu':		zlabel = r"$\hat{\mu}_s - \overline{\hat{\mu}_s}$ (mag)"  if zeroz else r"$\hat{\mu}_s$ (mag)"	;	pword = 'Distance'
			if pz=='AV':		zlabel = r"$\hat{A}^s_V - \overline{\hat{A}^s_V}$ (mag)"  if zeroz else r"$\hat{A}^s_V$ (mag)"	;	pword = 'Dust Extinction'
			if pz=='RV':		zlabel = r"$\hat{R}^s_V - \overline{\hat{R}^s_V}$" 		  if zeroz else r"$\hat{R}^s_V$"		;	pword = 'Dust Law Shape'
			if pz=='theta':		zlabel = r"$\hat{\theta}_s - \overline{\hat{\theta}_s}$"  if zeroz else r"$\hat{\theta}_s$"		;	pword = 'Light Curve Shape'
			if pz=='etaAV':		zlabel = r"$\hat{\eta}^s_{A_V} - \overline{\hat{\eta}^s_{A_V}}$ (mag)" if zeroz else r"$\hat{\eta}^s_{A_V}$ (mag)"	;	pword = 'Repar. Dust Extinction'
			return zlabel
		xlabel = get_axlab(px,zerox)
		ylabel = get_axlab(py,zeroy)

		#Get parameter estimates, zeroed by point estimate of average, and/or with posterior bunching up at bound
		def get_zs(pz,zeroz,dfgal,dfsummaries,SNe):
			zs,zerrs = dfgal[f'{pz}s'].values,dfgal[f'{pz}_errs'].tolist()
			if zeroz:	zs += - np.average(zs)
			for isn,sn in enumerate(SNe):
				if pz in ['AV','RV'] and dfsummaries[dfsummaries['SN']==sn][f'{pz}bunched'].values[0]:
					bunchindex  = int(dfsummaries[dfsummaries['SN']==sn][f'{pz}bunchindex'].values[0])
					point       = self.master_bounds[self.master_parnames.index(pz)][bunchindex]
					e68         = float(dfsummaries[dfsummaries['SN']==sn][f'{pz}68'].values[0])
					zerrs[isn] = [[e68*bunchindex],[e68*(1-bunchindex)]]
			return zs, zerrs

		#if annotate_mode=='legend':
		fig = pl.figure(figsize=(9.6*(1+0.25*args_legend['ncols']*(annotate_mode=='legend')+0.5*(annotate_mode is None)),7.2))
		#elif annotate_mode is None:
		#	fig = pl.figure()

		#For each galaxy, plot distances
		for igal,gal in enumerate(self.dfmus['Galaxy'].unique()):
			dfgal  = self.dfmus[self.dfmus['Galaxy']==gal]
			SNe      = dfgal['SN'].values
			xs,xerrs = get_zs(px,zerox,dfgal,dfsummaries,SNe)
			ys,yerrs = get_zs(py,zeroy,dfgal,dfsummaries,SNe)
			#For each SN
			for x,xerr,y,yerr,ss in zip(xs,xerrs,ys,yerrs,np.arange(dfgal.shape[0])):
				#Choice of labelling for legend/SNe
				lab = '\n'.join([sn[text_index:] for sn in SNe])
				lab = lab if ss==0 and annotate_mode=='legend' else None

				pl.errorbar(x,y,xerr=xerr,yerr=yerr,
					color=colours[igal],marker=markers[igal],markersize=markersize,alpha=alpha,elinewidth=elw,markeredgewidth=mew,linestyle='none',capsize=capsize, label=lab)

		#Legend (needs work)
		if annotate_mode=='legend':
			lines, labels = fig.axes[-1].get_legend_handles_labels()
			lines = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in lines]
			fig.axes[0].legend(lines,labels,fontsize=self.FS-2,title='SN Siblings', **args_legend,title_fontsize=self.FS)

		pl.xlabel(xlabel, fontsize=self.FS)
		pl.ylabel(ylabel, fontsize=self.FS)
		pl.tight_layout()
		pl.tick_params(labelsize=self.FS)
		if save:
			pl.savefig(self.plotpath+f"resplot_{px}{'nozero' if not zerox else ''}_{py}{'nozero' if not zeroy else ''}_samp{self.samplename}.pdf",bbox_inches="tight")
		if show:
			pl.show()

	def get_data(self):
		"""
		Get Data

		Method for getting stan data

		End Products
		----------
		stan_data dictionary
		"""
		Ns   = self.dfmus.shape[0]
		Nf   = len(self.pxs)
		Ng   = self.dfmus['Galaxy'].nunique()
		S_g  = self.dfmus.groupby('Galaxy',sort=False)['SN'].agg('count').values
		X    = self.dfmus[[f'{px}s' for px in self.pxs]].to_numpy()
		Xerr = self.dfmus[[f'{px}_errs' for px in self.pxs]].to_numpy()
		Y    = self.dfmus[f'{self.py}s'].to_numpy()
		Yerr = self.dfmus[f'{self.py}_errs'].to_numpy()

		'''#Simulated data for testing stan model
		Nd     = 100
		betas  = np.array([1 for _ in range(Nf)])
		sigint = 0.01
		x      = np.asarray([np.random.normal(0,5,Nd) for _ in range(Nf)]).T
		y      = x@betas + alpha + np.random.normal(0,sigint,Nd)
		X      = x+np.asarray([np.random.normal(0,0.01,Nd) for _ in range(Nf)]).T
		Y      = y+np.random.normal(0,0.01,Nd)
		Xerr   = np.asarray([np.ones(Nd)*0.01 for _ in range(Nf)]).T
		Yerr   = np.ones(Nd)*0.01
		#'''

		stan_data = dict(zip(
			['Ns','Nf','Ng','S_g','X','Xerr','Y','Yerr','alpha_prior',     'beta_prior',	 'sigint_prior',    'alpha_const',     'sigint_const'],
			[ Ns , Nf , Ng , S_g , X , Xerr , Y , Yerr , self.alpha_prior , self.beta_prior , self.sigint_prior, self.alpha_const , self.sigint_const]
		))
		#For cosmo-dep wrapper
		stan_data['Nb'] = stan_data['Nf']*2 if self.split_beta else stan_data['Nf']
		stan_data['zero_beta_common'] = 0 if self.common_beta is None else 1

		if self.beta is not None:#Sometimes freeze betas so input these as data
			stan_data['beta'] = self.beta

		return stan_data

	def get_inits(self):
		"""
		Get inits

		Simple initialisations for stan model

		End Product
		----------
		list of json files with initialisations that cmdstan can read
		"""
		#json_file = {**{'sigmaRel':1e-6}, **{f'beta[{_+1}]':0 for _ in range(len(self.pxs))}}
		#json_file = {**{f'etabeta[{_+1}]':0.5 for _ in range(len(self.pxs))}}
		Ng = self.dfmus['Galaxy'].nunique()
		json_file = {	**{f'y[{isn+1}]':y   for isn,y in enumerate(self.dfmus[f'{self.py}s'].values)},
						**{}#,
						#**{f'rho[{_+1}]':0.5 		for _ in range(len(self.pxs)+1)},
						#**{f'etasigint[{_+1}]':0.5 	for _ in range(len(self.pxs)+1)},
						#**{f'eta_y_common[{ig+1}]':self.dfmus[self.dfmus['Galaxy']==g][f'{self.py}_respoint'].values[0]*2*np.sqrt(2)/self.sigint_prior[0] for ig,g in enumerate(self.dfmus['Galaxy'].unique())},
						#**{f'eta_x_common[{Ng*f+(ig+1)}]':self.dfmus[self.dfmus['Galaxy']==g][f'{self.pxs[f]}_respoint'].values[0]*2*np.sqrt(2)/self.sigint_prior[0] for f in range(len(self.pxs)) for ig,g in enumerate(self.dfmus['Galaxy'].unique())},
						#**{f'eta_x_rel[{Ng*f+(ig+1)}]':0 for f in range(len(self.pxs)) for ig,g in enumerate(self.dfmus['Galaxy'].unique())}
					}
		for ff,px in enumerate(self.pxs):
			json_file = {**json_file, **{f'x[{isn+1},{ff+1}]':x for isn,x in enumerate(self.dfmus[f'{px}s'].values)}}

		with open(f"{self.productpath}inits.json", "w") as f:
			json.dump(json_file, f)

		stan_init  = [f"{self.productpath}inits.json" for _ in range(self.n_chains)]
		return stan_init

	def prepare_stan(self):
		"""
		Prepare stan

		Get the .stan file

		End Products
		----------
		Prints out model and writes to current_model.stan
		"""
		#Update stan file according to choices, create temporary 'current_model.stan'
		self.stan_filename = 'linear_regression.stan'#{True:'linear_regression_QR.stan',False:'linear_regression.stan'}[self.QR]
		if self.beta is not None:
			self.stan_filename = 'linear_regression_fixedB.stan'

		with open(self.stanpath+self.stan_filename,'r') as f:
			stan_file = f.read().splitlines()
		stan_file = '\n'.join(stan_file)
		print (stan_file)
		with open(self.stanpath+'current_model.stan','w') as f:
			f.write(stan_file)

	def reorder_pars(self,pars):
		"""
		Re-order pars

		For plotting, orders by chromatic param first, then the object param
			- mu->theta->etaAV...
			- and for each, do beta->alpha->rho->sigint->sigR->sigC

		Parameters
		----------
			pars : list of str
				the un-ordered names of the Stan parameters

		Returns
		----------
			ordered_pars : list of str
				ordered pars
		"""
		ordered_pars = []
		for par in [pp for pp in self.PARS if pp in self.pxs + [self.py]]:
			for x in ['beta','alpha','rho','sigmaint','sigmaRel','sigmaCommon']:
				if par in self.pxs:
					_ = self.pxs.index(par)
					if x!='beta': _ += 1
				elif par==self.py and x!='beta':
					_ = 0
				else:
					_ = None

				if _!=None:
					if self.split_beta and x=='beta':
						opar = [f"{x}[{_+1}]",f"{x}[{len(self.pxs)+_+1}]"]
					else:
						opar = [f"{x}[{_+1}]"]
					for op in opar:
						if op in pars:
							ordered_pars.append(op)

		return ordered_pars

	def get_pars(self,cosmo=False):
		"""
		Get Pars

		Gets the parameters used by model

		Parameters
		----------
		cosmo: bool (optional; default=False)
			if True, used in tandem with sigmaRel cosmo-dep, so don't remove e.g. sigmaCommon, sigma0, rho, for distances

		Returns
		----------
		pars : list of str
			the parameter names used/saved in model
		"""
		pars  = [f'beta[{_+1}]' for _ in range(len(self.pxs)*(2 if self.split_beta else 1))]
		pars += [f'alpha[{_+1}]' 		for _ in range(len(self.pxs)+1)]
		pars += [f'sigmaRel[{_+1}]' 	for _ in range(len(self.pxs)+1)]
		pars += [f'sigmaCommon[{_+1}]' 	for _ in range(len(self.pxs)+1)]
		pars += [f'sigmaint[{_+1}]' 	for _ in range(len(self.pxs)+1)]
		pars += [f'rho[{_+1}]' 			for _ in range(len(self.pxs)+1)]
		if self.beta is not None:
			for _ in range(len(self.pxs)*(2 if self.split_beta else 1)):
				pars.remove(f'beta[{_+1}]')
		if self.mu_index is not None and not cosmo:
			pars.remove(f'alpha[{self.mu_index+1}]')
			pars.remove(f'sigmaint[{self.mu_index+1}]')
			pars.remove(f'rho[{self.mu_index+1}]')
			pars.remove(f'sigmaCommon[{self.mu_index+1}]')
		if self.alpha is False: pars = [par for par in pars if 'alpha' not in par]
		pars = self.reorder_pars(pars)
		return pars

	def get_modelkey(self):
		"""
		Get Model Key

		Takes input choices and builds a filename called modelkey

		End Product(s)
		----------
		self.modelkey : str
			descriptive filename of model choice
		"""

		print ('###'*10)
		self.modelkey = f"Y{self.py}__X{'_'.join(self.pxs)}_"
		if self.alpha is False:
			print (f'Setting intercepts to zero')
		else:
			print (f'alpha={self.alpha}')
			print (f'Setting intercepts to {self.alpha_const} with prior width {self.alpha_prior} for {self.py} and {self.pxs}')

		if self.beta is None:
			print ('Fitting beta parameters')
			self.modelkey += '_fitbeta'
		else:
			print (f'Fixing beta parameters of {self.pxs} to:', self.beta)
			self.modelkey += f"_fixbeta{'_'.join([str(b) for b in self.beta])}"
		print ('###'*10)
		self.filename = f"FIT{self.samplename}{self.modelkey}.pkl"

	def sample_posterior(self,py='mu',pxs=['theta'], alpha_prior=10, beta_prior=10, sigint_prior=10, alpha_const=0, sigint_const=0, alpha=False, beta=None, overwrite=True, run=True, common_beta=None):
		"""
		Sample Posterior

		Method for running posterior sampling

		Parameters
		----------
		py, pxs : str, list of str (optional; default='mu', ['theta'])
			predictor and features for multiple linear regression

		alpha_prior,beta_prior,sigint_prior,alpha_const,sigint_const: floats (optional; default=10,10,10,0,0)
			width on priors and added constant to sigma hyperparameters

		alpha: None or False (optional; default is False)
			if False, set all alpha hyperparameters to zero

		beta : None or list of floats (optional; default is None)
			if None, beta is fitted for, else, it is fixed at values given in list e.g. beta = [0.1]

		overwrite : bool (optional; default=True)
			if True, run new fit, else, load up previous fit

		run : bool (optional; default=True)
			option to run intialisations without actually doing fit

		common_beta :  None or bool (optional; default=None)
			Only applicable for when wrapping with cosmo-dep sigmaRel_computer
			if True, beta==beta_rel==beta_common; if False, beta_rel!=beta_common; if None, beta_common=0

		End Product(s)
		----------
		runs stan fit, saves FIT in productpath
		assigns self.FIT
		"""
		self.py    = py
		self.pxs   = pxs
		self.mu_index = 0 if self.py=='mu' else None if 'mu' not in self.pxs else self.pxs.index('mu')+1
		self.common_beta  = common_beta
		self.split_beta   = True if self.common_beta==False else False
		self.alpha        = alpha
		self.beta         = [beta 			for _ in range(len(self.pxs)*(2 if self.split_beta else 1))] if type(beta) 			in [float,int] else beta
		self.beta_prior   = [beta_prior 	for _ in range(len(self.pxs)*(2 if self.split_beta else 1))] if type(beta_prior)	in [float,int] else beta_prior
		self.alpha_prior  = [alpha_prior 	for _ in range(len(self.pxs)+1)] 	if type(alpha_prior)  in [float,int] else alpha_prior
		self.sigint_prior = [sigint_prior 	for _ in range(len(self.pxs)+1)] 	if type(sigint_prior) in [float,int] else sigint_prior
		self.alpha_const  = [alpha_const 	for _ in range(len(self.pxs)+1)] 	if type(alpha_const)  in [float,int] else alpha_const
		self.sigint_const = [sigint_const 	for _ in range(len(self.pxs)+1)] 	if type(sigint_const) in [float,int] else sigint_const
		if self.mu_index is not None:#Remove constraining alpha_mu,sigma_mu_tot
			self.alpha_prior[self.mu_index]  = 1e-6
			self.alpha_const[self.mu_index]  = 0
			self.sigint_prior[self.mu_index] = 1e-6
			self.sigint_const[self.mu_index] = 100#Prior width on mus
		if self.alpha is False:
			self.alpha_prior = [1e-6 	for _ in range(len(self.pxs)+1)]
			self.alpha_const = [0 		for _ in range(len(self.pxs)+1)]

		self.get_modelkey()

		if run:
			#If files don't exist or overwrite, do stan fits
			if not os.path.exists(self.productpath+self.filename) or overwrite:
				stan_data = self.get_data()
				stan_init = self.get_inits()
				self.prepare_stan()
				self.pars = self.get_pars()

				#Initialise and fit model
				model       = CmdStanModel(stan_file=self.stanpath+'current_model.stan')
				fit         = model.sample(data=stan_data,chains=self.n_chains, iter_sampling=self.n_sampling, iter_warmup = self.n_warmup, inits=stan_init, seed=42)
				df          = fit.draws_pd()

				#Thin samples
				if df.shape[0]>self.n_thin*self.n_chains:#Thin samples
					print (f'Thinning samples down to {self.n_thin} per chain to save on space complexity')
					Nthinsize = int(self.n_thin*self.n_chains)
					df = df.iloc[0:df.shape[0]:int(df.shape[0]/Nthinsize)]						#thin to e.g. 1000 samples per chain
					dfdict = df.to_dict(orient='list')											#change to dictionary so key,value is parameter name and samples
					fit = {key:np.array_split(value,self.n_chains) for key,value in dfdict.items()}	#change samples so split into chains

				#Fitsummary including Rhat valuess
				fitsummary = az.summary(fit)#feed dictionary into arviz to get summary stats of thinned samples

				#Save samples
				FIT = dict(zip(['data','summary','chains'],[stan_data,fitsummary,df]))
				with open(self.productpath+self.filename,'wb') as f:
					pickle.dump(FIT,f)

				#Print posterior summaries
				print (f"Fit Completed; Summary is:")
				for col in self.pars:
					print ('###'*10)
					print (f'{col}, median, std, 16, 84 // 68%, 95%')
					print (df[col].median().round(3),df[col].std().round(3),df[col].quantile(0.16).round(3),df[col].quantile(0.84).round(3))
					print (df[col].quantile(0.68).round(3),df[col].quantile(0.95).round(3))
					print ('Rhat:', fitsummary.loc[col]['r_hat'])

			else:#Else load up
				with open(self.productpath+self.filename,'rb') as f:
					FIT = pickle.load(f)
				self.pars = self.get_pars()#For plotting

			#Print distance summaries
			self.FIT  = FIT

	def get_parlabels(self):
		"""
		Get Parlabels

		Simple Method for posterior plotting, gets the appropriate labels and filters down based on model fitting choices

		End Product(s)
		----------
		assigns: parnames, dfparnames, parlabels, bounds
		parnames are names in data, dfparnames are names in model, parlabels are those for plotting, bounds are hard prior bounds on parameters for reflection KDEs
		"""
		#Initialisation
		parnames   = [f'beta[{_+1}]' for _ in range(len(self.pxs)*(2 if self.split_beta else 1))] + [f'alpha[{_+1}]' for _ in range(len(self.pxs)+1)] + [f'rho[{_+1}]' for _ in range(len(self.pxs)+1)]
		dfparnames = [f'beta[{_+1}]' for _ in range(len(self.pxs)*(2 if self.split_beta else 1))] + [f'alpha[{_+1}]' for _ in range(len(self.pxs)+1)] + [f'rho[{_+1}]' for _ in range(len(self.pxs)+1)]
		if self.common_beta is True:
			parlabels  = ['$\\beta_{%s}$'%(self.plabdict[px]) for px in self.pxs]
		else:
			assert(self.common_beta in [None,False])
			parlabels  = ['$\\beta_{\\rm{Rel}}^{%s}$'%(self.plabdict[px]) for px in self.pxs]
			if self.common_beta is False:	parlabels += ['$\\beta_{\\rm{Common}}^{%s}$'%(self.plabdict[px]) for px in self.pxs]
		parlabels += ['$\\alpha_{%s}$'%(self.plabdict[pxy]) for pxy in [self.py]+self.pxs] + ['$\\rho^{%s}$'%(self.plabdict[pxy]) for pxy in [self.py]+self.pxs]
		bounds     = [[None,None] for _ in range(len(self.pxs)*(2 if self.split_beta else 1))] + [[None,None] for _ in range(len(self.pxs)+1)] + [[0,1] for _ in range(len(self.pxs)+1)]
		for intword in ['Rel','Common','int']:
			parnames   += [f'sigma{intword}[{_+1}]' for _ in range(len(self.pxs)+1)]
			dfparnames += [f'sigma{intword}[{_+1}]' for _ in range(len(self.pxs)+1)]
			parlabels  += ['$\\sigma_{\\rm{%s}}^{%s}$'%(intword if intword!='int' else 'Tot' if ii!=self.mu_index else '0',self.plabdict[pxy]) for ii,pxy in enumerate([self.py]+self.pxs)]
			bounds     += [[0,None] for _ in range(len(self.pxs)+1)]
		parlabels  = [x+' (mag)' if ('sigma' in x and 'mu' in x) else x for x in parlabels]
		parlabels  = [x.replace('\mu','') for x in parlabels]

		#Filter on pars
		PARS  = pd.DataFrame(data=dict(zip(['dfparnames','parlabels','bounds'],[dfparnames,parlabels,bounds])),index=parnames)
		PARS = PARS.loc[self.pars]

		#Set class attributes
		self.parnames   = self.pars
		self.dfparnames = PARS['dfparnames'].values
		self.parlabels  = PARS['parlabels'].values
		self.bounds     = PARS['bounds'].values

	def get_res_lines(self,stan_data):
		"""
		Get Res Lines

		Simple Method to get descriptive lines for corner plot

		Parameters
		----------
		stan_data : dict
			dictionary of data used in fit

		Returns
		----------
		Lines :  list of str
			descriptive lines
		"""
		S_g,Ng = stan_data['S_g'], stan_data['Ng']
		Lines  = add_siblings_galaxies(Ng,S_g)
		def modder(x,word='Rel'):
			x = x.replace('mu','delta M')
			if word!='':
				if '}' in x: 	return x.split('}')[0] + ';\,\\rm{%s}}'%word
				else: 			return x + '_{\\rm{%s}}'%word
			else:
				return x
		ypar  = self.plabdict[self.py]
		xpars = [self.plabdict[px] for px in self.pxs]
		assert(stan_data['Nf'] == int(sum([ss==self.sigint_prior[-1] for ss in self.sigint_prior[1:]])) )#Ensure sigint priors are same for all X params
		if stan_data['Nf']==1:
			#Lines += [r"$Y \equiv %s$ ; $X \equiv %s$"%(ypar,xpars[0])]
			if self.split_beta is True:
				Lines += [r"$%s^s = \beta_{\rm{Rel}}^{%s} %s^s + \beta_{\rm{Common}}^{%s} %s^g + \epsilon^s_{0}$"%(modder(ypar,''),xpars[0],modder(xpars[0]),xpars[0],modder(xpars[0],'Common'))]
			if self.common_beta is True:
				Lines += [r"$%s^s = \beta^{%s} %s^s + \epsilon^s_{0}$"%(modder(ypar,''),xpars[0],modder(xpars[0],''))]
			if self.common_beta is None:
				Lines += [r"$%s^s = \beta_{\rm{Rel}}^{%s} %s^s + \epsilon^s_{\rm{Rel}}$"%(modder(ypar),xpars[0],modder(xpars[0]))]
			if self.beta is not None:
				Lines[-1] = Lines[-1].replace(r'\beta',r'\hat{\beta}')
				if self.split_beta is True:
					Lines += [r"$\{ \hat{\beta}_{\rm{Rel}}^{%s},\hat{\beta}_{\rm{Common}}^{%s} \} = \{%s\}$"%(xpars[0],xpars[0],','.join([str(b) for b in self.beta]))]
				if self.common_beta is True:
					Lines += [r"$\hat{\beta}^{%s}=%s$"%(xpars[0],self.beta[0])]
				if self.common_beta is None:
					Lines += [r"$\hat{\beta}_{\rm{Rel}}^{%s}=%s$"%(xpars[0],self.beta[0])]

			Lines += [r"$\sigma^{%s}_{\rm{Tot}} \sim U(0,%s)$"%(xpars[0],self.sigint_prior[-1])]
			Lines += [r"$\rho^{%s} \sim \rm{Arcsine}(0,1)$"%(xpars[0])]
		else:
			#Lines += [r"$Y \equiv %s$ ; $X \equiv \{%s\}$"%(ypar,','.join(xpars))]
			if self.beta is None:
				Lines += [r"$X \equiv \{%s\}$"%(','.join(xpars))]
			else:
				if self.split_beta is True:
					rel_betas = ','.join([str(b) for b in self.beta[:len(self.pxs)]]) ; common_betas = ','.join([str(b) for b in self.beta[len(self.pxs):]])
					Lines += [r"$X \equiv \{%s\} ; \hat{\beta}_{\rm{Rel}} \equiv \{%s\} ; \hat{\beta}_{\rm{Common}} \equiv \{%s\}$"%(','.join(xpars),rel_betas,common_betas)]
				if self.common_beta is True:
					Lines += [r"$X \equiv \{%s\} ; \hat{\beta} \equiv \{%s\}$"%(','.join(xpars),','.join([str(b) for b in self.beta]))]
				if self.common_beta is None:
					Lines += [r"$X \equiv \{%s\} ; \hat{\beta}_{\rm{Rel}} \equiv \{%s\}$"%(','.join(xpars),','.join([str(b) for b in self.beta]))]

			#Lines += [r"$%s^s = \beta_{X_i} X^s_{i;\,\rm{Rel}} + \epsilon^s_{\rm{Rel}}$"%(modder(ypar))]
			if self.split_beta is True:
				Lines += [r"$%s^s = \beta_{\rm{Rel}}^{X_i} X^s_{i;\,\rm{Rel}} + \beta_{\rm{Common}}^{X_i} X^g_{i;\,\rm{Common}} + \epsilon^s_{0}$"%(modder(ypar,''))]
			if self.common_beta is True:
				Lines += [r"$%s^s = \beta^{X_i} X^s_{i} + \epsilon^s_{0}$"%(modder(ypar,''))]
			if self.common_beta is None:
				Lines += [r"$%s^s = \beta_{\rm{Rel}}^{X_i} X^s_{i;\,\rm{Rel}} + \epsilon^s_{\rm{Rel}}$"%(modder(ypar))]
			if self.beta is not None:
				Lines[-1] = Lines[-1].replace(r'\beta',r'\hat{\beta}')
			Lines += [r"$\sigma^{X_i}_{\rm{Tot}} \sim U(0,%s)$"%self.sigint_prior[-1]]#Lines += [r"$\mathbf{\sigma}^X_{\rm{Tot}} \sim U(0,%s)$"%self.sigint_prior[-1]]
			Lines += [r"$\rho^{X_i} \sim \rm{Arcsine}(0,1)$"]#Lines += [r"$\mathbf{\rho}_X \sim \rm{Arcsine}(0,1)$"]
		return Lines

	def plot_posterior_samples(self, FS=None,paperstyle=True,quick=True,save=True,show=False, returner=False):
		"""
		Plot Posterior Samples

		Creates corner plot of fit

		Parameters
		----------
		FS : float or None (optional; default=None)
			fontsize for plotting, if None use self.FS

		paperstyle : bool (optional; default=True)
			if False, plot up Rhat values

		quick : bool (optional; default=True)
			if True, plot samples in 2D panels, if False, plot KDEs in 2D panel

		show : bool (optional; default=False)
			bool to show plots or not

		save : bool (optional; default=True)
			bool to save plots or not

		returner : bool (optional; default=False)
			if True, return posterior summaries for use in a Table

		End Product(s)
		----------
		Plot
		"""
		if FS is None: FS = self.FS

		#Get pars to plot
		if 'modelkey' not in self.__dict__:
			raise Exception('Please run sample_posterior() with overwrite=False to initialise')
		self.get_parlabels()

		#Get posterior samples
		if 'FIT' in self.__dict__.keys():
			FIT = self.FIT
		else:
			with open(self.productpath+self.filename,'rb') as f:
				FIT = pickle.load(f)

		#Get posterior data
		df         = FIT['chains'] ; fitsummary = FIT['summary'] ; stan_data  = FIT['data']
		Rhats      = {par:fitsummary.loc[par]['r_hat'] for par in self.dfparnames}
		samples    = {par:np.asarray(df[par].values)   for par in self.dfparnames}
		print ('Rhats:',Rhats)

		#Corner Plot
		self.plotting_parameters = {'FS':FS,'paperstyle':paperstyle,'quick':quick,'save':save,'show':show}
		postplot = POSTERIOR_PLOTTER(samples, self.parnames, self.parlabels, self.bounds, Rhats, self.plotting_parameters)
		Summary_Strs = postplot.corner_plot()#Table Summary
		savekey         = self.samplename+self.modelkey+'_FullKDE'*bool(not self.plotting_parameters['quick'])+'_NotPaperstyle'*bool(not self.plotting_parameters['paperstyle'])
		save,quick,show = [self.plotting_parameters[x] for x in ['save','quick','show']][:]
		#finish_corner_plot(postplot.fig,postplot.ax,get_Lines(stan_data,self.c_light,modelloader.alt_prior),save,show,self.plotpath,savekey)
		finish_corner_plot(postplot.fig,postplot.ax,self.get_res_lines(stan_data),save,show,self.plotpath,savekey)

		#Return posterior summaries
		if returner: return Summary_Strs
