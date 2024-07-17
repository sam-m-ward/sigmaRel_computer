"""
Simulation-Based Calibration Plotting Functions

Module contains functions useful for plotting SBC

Contains
----------
SBC_FITS_PLOTTER class
	inputs : self,iax,ax,PAR,FITS,bounds,path_to_sigmaRel_rootpath,quantilemode=True,Quantiles=[0,0.025,0.05,0.16,0.5,0.84,0.95,0.975,1],FS=18

	Methods are:
		get_SAMPS()
		get_QUANTILES()
		plot_sbc_panel(Ncred=True,Parcred=False,sap=None,annotate_true=True,real_data_samps=False,plot_ind=True,plot_true=True,plot_medians=True,include_pmedian=True,dress_figure=True,fill_between=True,color='C0',linestyle='-',Lside=False,FAC=None,XGRID=None,line_sap_title=None,line_rec_title=None)

Functions are:
	get_KEEPERS(GLOB_FITS,Nsim_keep,Rhat_threshold,loop_par,dfpar)
	trim_to_KEEPERS(GLOB_FITS,KEEPERS)
	get_sap_Ncred(loop_par,true_rho,ALT_PRIOR)

--------------------

Written by Sam M. Ward: smw92@cam.ac.uk

"""
import pandas as pd
import numpy as np
from contextlib import suppress
import pickle

class SBC_FITS_PLOTTER:

	def __init__(self,iax,ax,PAR,FITS,bounds,path_to_sigmaRel_rootpath,quantilemode=True,Quantiles=[0,0.025,0.05,0.16,0.32,0.5,0.68,0.84,0.95,0.975,1],FS=18):
		self.iax = iax
		self.ax  = ax
		true_par,par,dfpar,parlabel = PAR[:]
		self.true_par = true_par
		self.par      = par
		self.dfpar    = dfpar
		self.parlabel = parlabel
		self.FITS     = FITS
		self.bounds   = bounds
		self.path_to_sigmaRel_rootpath = path_to_sigmaRel_rootpath
		self.quantilemode = quantilemode
		self.Quantiles    = Quantiles
		self.FS = 18


	def get_SAMPS(self):
		"""
		Get Samples

		Method to get pandas df of samples across simulations

		Returns
		----------
		SAMPS : pandas df
			each row is one simulation; samples for the (hyper)parameter
		"""
		SAMPS = pd.DataFrame(data={ISIM:self.FITS[ISIM]['chains'][self.dfpar].values for ISIM in self.FITS})
		return SAMPS

	def get_QUANTILES(self):
		"""
		Get Quantiles

		Method to get pandas df of quantiles across simulations

		Returns
		----------
		QUANTILES : pandas df
			quantiles across simulations
		"""
		QUANTILES = pd.DataFrame(data={q:[self.FITS[ISIM]['chains'][self.dfpar].quantile(q) for ISIM in self.FITS] for q in self.Quantiles})
		return QUANTILES

	def plot_sbc_panel(self,Ncred=True,Parcred=False,sap=None,annotate_true=True,real_data_samps=False,plot_ind=True,plot_true=True,plot_medians=True,include_pmedian=True,dress_figure=True,fill_between=True,color='C0',linestyle='-',Lside=False,FAC=None,XGRID=None,line_sap_title=None,line_rec_title=None):
		"""
		Plot SBC Panel

		Method to plot a single panel of hyperparameter recovery

		Parameters
		----------
		Ncred : bool (optional; default=True)
			if True, plot N68, N95
		Parcred : bool (optional; default=False)
			if True, plot 68 and 95 quantiles of parameter
		sap : None or int (optional; default=None)
			if None, summarise using normal singly peaked Gaussian like posterior;
			otherwise, summarise as peaking at lower or upper prior boundary, indicated by sap = 0 or 1, respectively
		annotate_true : bool (optional; default=True)
			if True, plot 'True par = value'
		real_data_samps : array (optional; default is None)
			if True, plot up samples from real-data fit
		plot_ind : bool (optional; default=True)
			if True, plot faint line posteriors, one for each sim
		plot_true: bool (optional; default=True)
			if True, plot line for True parameter value
		plot_medians : bool (optional; default=True)
			plot medians and include in legend
		include_pmedian : bool (optional; default=True)
			if True include p(median<Truth) line in legend
		dress_figure : bool (optional; default=True)
			apply e.g. lims, set yticks etc.
		fill_between : bool (optional; default=True)
			if True, include shaded regions for 68% credible interval in simulation averaged posterior
		color : str
			color of panel
		linestyle : str
			linestyle for simulation-averaged posterior
		Lside : bool (optional; default=False)
			if True, put annotations on LHS of panel
		FAC : float (optional; default=None)
			factor to reduce KDE grid for simulation-averaged posterior by compared to No.of samples
		XGRID : list (optional; default=None)
			[xmin,xmax,Nsteps] for .xgrid
		line_sap_title : str (optional; default=None)
			string used in legend for simulation-averaged posterior, defaults to 'Simulation-Averaged Posterior'
		line_rec_title : str (optional; default=None)
			string used in legend for real-data posterior, defaults to 'Real-Data Posterior'

		End Product(s)
		----------
		ax[iax] panel with:
			faint lines for per-Sim posterior
			thick line for simulation-averaged posterior
			orange band for simulation medians
			legend with median and sap summaries
			line for true parameter and annotation
			N68,N95 annotations
		"""
		#Imports from SigmaRel
		import sys
		sys.path.append(self.path_to_sigmaRel_rootpath+'model_files/')
		from plotting_script import PARAMETER

		true_par  = self.true_par
		loop_par  = self.par
		dfpar     = self.dfpar
		parlabel  = self.parlabel
		FITS      = self.FITS
		Quantiles = self.Quantiles
		iax,ax    = self.iax,self.ax
		FS = self.FS
		HA = {True:'left',False:'right'}[Lside]

		#Get SAMPS, QUANTILES
		SAMPS     = self.get_SAMPS()
		QUANTILES = self.get_QUANTILES()
		lims      = {loop_par:[SAMPS.min().min(),SAMPS.max().max()]}
		self.lims   = lims
		#self.bounds = bounds
		roundint = 2 if loop_par not in ['sigmaRel','sigmaCommon'] else 3
		if sap is not None:
			Ncred=False;Parcred=True
			assert(sap in [0,1])
			x68 = {0:0.68,1:0.32}[sap]
			x95 = {0:0.95,1:0.05}[sap]

		###Plot and Simulation Averaged Posterior
		samps = PARAMETER(SAMPS.stack(),dfpar,parlabel,lims[loop_par],self.bounds[loop_par],None,iax,{},XGRID=XGRID)
		sap_chain = samps.chain
		#Plot and label
		if XGRID is None:
			if FAC is None:	samps.Nsamps /= 10
			else:			samps.Nsamps /= FAC
		samps.get_xgrid_KDE()
		if line_sap_title is None: line_sap_title = 'Simulation-Averaged Posterior'
		if sap is None:
			if self.quantilemode:	line_sap_summary = r'$%s = %s ^{+%s}_{-%s}$'%(parlabel,sap_chain.quantile(0.5).round(roundint),round(sap_chain.quantile(0.84)-sap_chain.quantile(0.5),roundint),round(sap_chain.quantile(0.5)-sap_chain.quantile(0.16),roundint))
			else:					line_sap_summary = r'$%s = %s \pm %s$'%(parlabel,sap_chain.quantile(0.5).round(roundint),sap_chain.std().round(roundint))
		else:
			lg = {0:'<',1:'>'}[sap]
			line_sap_summary = r'$%s %s %s (%s)$'%(parlabel,lg,sap_chain.quantile(x68).round(roundint),sap_chain.quantile(x95).round(roundint))
		print (line_sap_title+line_sap_summary)
		ax[iax].plot(samps.xgrid,samps.KDE,alpha=1,color=color,linewidth=3,label='\n'.join([line_sap_title,line_sap_summary]) if line_sap_title is not '' else line_sap_summary,linestyle=linestyle)
		KDEmax      = np.amax(samps.KDE)
		simavheight = samps.KDE[np.argmin(np.abs(sap_chain.quantile(0.5)-samps.xgrid))]
		if sap in [0,1]:
			ax[iax].plot(sap_chain.quantile(x68)*np.ones(2),[0,samps.KDE[np.argmin(np.abs(sap_chain.quantile(x68)-samps.xgrid))]],c=color,linewidth=2,linestyle=linestyle)
			ax[iax].plot(sap_chain.quantile(x95)*np.ones(2),[0,samps.KDE[np.argmin(np.abs(sap_chain.quantile(x95)-samps.xgrid))]],c=color,linewidth=2,linestyle=linestyle)
		elif sap is None:
			ax[iax].plot(sap_chain.quantile(0.5)*np.ones(2),[0,simavheight],c=color,linewidth=2,linestyle=linestyle)
		#Fill between with quantiles
		if fill_between:
			if sap is None:
				for qlo,qhi in zip([0.16,0.025],[0.84,0.975]):
					siglo = samps.get_KDE_values(value=sap_chain.quantile(qlo), return_only_index=True)
					sigup = samps.get_KDE_values(value=sap_chain.quantile(qhi), return_only_index=True)+1
					ax[iax].fill_between(samps.xgrid[siglo:sigup],np.zeros(sigup-siglo),samps.KDE[siglo:sigup],color=color,alpha=0.2)
			elif sap in [0,1]:
				for qlo,qhi in zip([sap,sap],[x68,x95]):
					if sap==1: qlo,qhi = qhi,qlo
					siglo = samps.get_KDE_values(value=sap_chain.quantile(qlo), return_only_index=True)
					sigup = samps.get_KDE_values(value=sap_chain.quantile(qhi), return_only_index=True)+1
					ax[iax].fill_between(samps.xgrid[siglo:sigup],np.zeros(sigup-siglo),samps.KDE[siglo:sigup],color=color,alpha=0.2)



		#Plot N68, N95
		N68   = (SAMPS.quantile(0.16) <= true_par) & (true_par < SAMPS.quantile(0.84)) ; N95   = (SAMPS.quantile(0.025) <= true_par) & (true_par < SAMPS.quantile(0.975))
		N68   = SAMPS.transpose()[N68.values].shape[0] 										 ; N95   = SAMPS.transpose()[N95.values].shape[0]
		print(round(100*(N68/SAMPS.shape[1]),1),'%',round(100*(N95/SAMPS.shape[1]),1),'%')
		if Ncred:
			line68 = r'$N_{68} = %s$'%round(100*(N68/SAMPS.shape[1]),1)+'%' 						 ;
			line95 = r'$N_{95} = %s$'%round(100*(N95/SAMPS.shape[1]),1)+'%'
		elif Parcred:
			ll68= ';\,%s}'%str(int(100*x68)) if parlabel[-1]=='}' else '_{%s}'%str(int(100*x68)) ; ll95= ';\,%s}'%str(int(100*x95)) if parlabel[-1]=='}' else '_{%s}'%str(int(100*x95))
			line68 = r'$%s%s=%s ^{+%s}_{-%s}$'%(parlabel[:-1] if parlabel[-1]=='}'  else parlabel,ll68
														,SAMPS.quantile(x68).median().round(roundint), round(SAMPS.quantile(x68).quantile(0.84)-SAMPS.quantile(x68).quantile(0.5),roundint),round(SAMPS.quantile(x68).quantile(0.5)-SAMPS.quantile(x68).quantile(0.16),roundint))
			line95 = r'$%s%s=%s ^{+%s}_{-%s}$'%(parlabel[:-1] if parlabel[-1]=='}'  else parlabel,ll95
														,SAMPS.quantile(x95).median().round(roundint), round(SAMPS.quantile(x95).quantile(0.84)-SAMPS.quantile(x95).quantile(0.5),roundint),round(SAMPS.quantile(x95).quantile(0.5)-SAMPS.quantile(x95).quantile(0.16),roundint))
		else:
			line68,line95='',''
		print (line68+line95)
		if sap is None:
			ax[iax].annotate(line68,xy=(0.95-(0.95-0.05)*Lside,0.425),xycoords='axes fraction',fontsize=FS,ha=HA)
			ax[iax].annotate(line95,xy=(0.95-(0.95-0.05)*Lside,0.375-Parcred*(0.025)),xycoords='axes fraction',fontsize=FS,ha=HA)

		#Real-Data Posterior Fit
		if real_data_samps is not False:
			if line_rec_title is None: line_rec_title   = 'Real-Data Posterior'
			samps = PARAMETER(real_data_samps,dfpar,parlabel,lims[loop_par],bounds[loop_par],-1,iax,{},XGRID=XGRID)
			samps.get_xgrid_KDE()
			ax[iax].plot(samps.xgrid,samps.KDE,alpha=1,linewidth=3,color='C3',label=f"{line_rec_title} \n"+r'$%s = %s ^{+%s}_{-%s}$'%(parlabel,real_data_samps.quantile(0.5).round(roundint),round(real_data_samps.quantile(0.84)-real_data_samps.quantile(0.5),2),round(real_data_samps.quantile(0.5)-real_data_samps.quantile(0.16),2)))
			simavheight = samps.KDE[np.argmin(np.abs(real_data_samps.quantile(0.5)-samps.xgrid))]
			ax[iax].plot(real_data_samps.quantile(0.5)*np.ones(2),[0,simavheight],c='C3',linewidth=2)

		#Plot per-Sim faint posteriors
		if plot_ind:
			KDEmax = 0
			for ISIM in SAMPS:
				samps = PARAMETER(SAMPS[ISIM],dfpar,parlabel,lims[loop_par],self.bounds[loop_par],FITS[ISIM]['summary'].loc[dfpar]['r_hat'],iax,{},XGRID=XGRID)
				samps.get_xgrid_KDE()
				KDEmax = max(KDEmax,np.amax(samps.KDE))
				ax[iax].plot(samps.xgrid,samps.KDE,alpha=0.08,color=color)

		#Plot True Parameter Value, and Annotate
		if plot_true is True:
			ax[iax].plot(true_par*np.ones(2),[0,KDEmax],c='black',linewidth=5,linestyle='--')
		if annotate_true:
			ax[iax].annotate(r'True $%s=%s$'%(parlabel,true_par),xy=(0.95-(0.95-0.05)*Lside,0.5+0.02),xycoords='axes fraction',fontsize=FS,ha='right' if not Lside else 'left',bbox=dict(facecolor='white',alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

		#Plot and Annotate Medians
		if plot_medians:
			if sap is None:
				if self.quantilemode:	line_median  = r'Median-$%s=%s^{+%s}_{-%s}$'%(parlabel,QUANTILES[0.5].quantile(0.5).round(roundint),round(QUANTILES[0.5].quantile(0.84)-QUANTILES[0.5].quantile(0.5),roundint),round(QUANTILES[0.5].quantile(0.5)-QUANTILES[0.5].quantile(0.16),roundint))
				else:					line_median  = r'Median-$%s=%s\pm%s$'%(parlabel,QUANTILES[0.5].quantile(0.5).round(roundint),QUANTILES[0.5].std().round(roundint))

				if parlabel[-1]=='}':
					line_pmedian = r"$p($"+'Median-'+r"$%s<%s ; \,\rm{True}})=%s$"% \
						(parlabel, parlabel[:-1],round(100*QUANTILES[QUANTILES[0.5]<true_par].shape[0]/QUANTILES[0.5].shape[0],1)) + '%'
				else:
					line_pmedian = r"$p($"+'Median-'+r"$%s<%s_{\rm{True}})=%s$"% \
						(parlabel, parlabel,round(100*QUANTILES[QUANTILES[0.5]<true_par].shape[0]/QUANTILES[0.5].shape[0],1)) + '%'
				if not include_pmedian:	median_labels = line_median
				else:					median_labels = '\n'.join([line_median,line_pmedian])
				ax[iax].plot(QUANTILES[0.5].quantile(0.5)*np.ones(2),[0,KDEmax],c='C1'	,linewidth=2,label=median_labels,linestyle=':')
				ax[iax].fill_between([QUANTILES[0.5].quantile(0.16),QUANTILES[0.5].quantile(0.84)],[0,0],[KDEmax,KDEmax],color='C1',alpha=0.2)
				print (line_median+line_pmedian)

			elif sap in [0,1]:
				ax[iax].plot(SAMPS.quantile(x68).quantile(0.5)*np.ones(2),[0,KDEmax],c='C2'	,linewidth=2,linestyle=':',label=line68)#,label=median_labels)
				ax[iax].fill_between([SAMPS.quantile(x68).quantile(0.16),SAMPS.quantile(x68).quantile(0.84)],[0,0],[KDEmax,KDEmax],color='C2',alpha=0.2)

				ax[iax].plot(SAMPS.quantile(x95).quantile(0.5)*np.ones(2),[0,KDEmax],c='C6'	,linewidth=2,linestyle=':',label=line95)#,label=median_labels)
				ax[iax].fill_between([SAMPS.quantile(x95).quantile(0.16),SAMPS.quantile(x95).quantile(0.84)],[0,0],[KDEmax,KDEmax],color='C6',alpha=0.2)

				XR = ax[iax].get_xlim()
				XR = XR[1]-XR[0]
				X68 = str(int(x68*100))+'%'; X95 = str(int(x95*100))+'%'; HA = {0:'left',1:'right'}[sap]
				ax[iax].annotate(X68,xy=(SAMPS.quantile(x68).quantile(0.5)+((-1)**sap)*XR*0.01,KDEmax*0.4),fontsize=FS,ha=HA,weight='bold',color='C2')
				ax[iax].annotate(X95,xy=(SAMPS.quantile(x95).quantile(0.5)+((-1)**sap)*XR*0.01,KDEmax*0.4),fontsize=FS,ha=HA,weight='bold',color='C6')

		#Set ticks and legend
		if dress_figure:
			ax[iax].set_ylim([0,max(KDEmax,simavheight)])
			ax[iax].set_yticks([])
			ax[iax].legend(fontsize=FS-1*('com_rat' in loop_par and sap is None),framealpha=1,loc='upper right')
			ax[iax].tick_params(labelsize=FS)

		#Set appropriate limits
		if XGRID is None:
			FAC  = 3
			XLIM =  FAC*np.array([sap_chain.quantile(0.16),sap_chain.quantile(0.84)])-(FAC-1)*sap_chain.quantile(0.5)#XLIMS
			with suppress(TypeError): XLIM[0] = max([XLIM[0],self.bounds[loop_par][0]])
			with suppress(TypeError): XLIM[1] = min([XLIM[1],self.bounds[loop_par][1]])
			ax[iax].set_xlim(XLIM)
		else:
			ax[iax].set_xlim(XGRID[:2])



def get_KEEPERS(GLOB_FITS,Nsim_keep,Rhat_threshold,loop_par,dfpar):
	"""
	Get Keepers

	Function to get indices of simulations to keep,
	loops through all sims for all parameters values
	identifies individual simulations where Rhat>Rhat_threshold

	Parameters
	----------
	GLOB_FITS : dict
		{parvalue:FITS} where FITS={ISIM:FIT}
	Nsim_keep : int
		No. of simulations to plot
	Rhat_threshold : float
		maximum allowed Rhat
	loop_par : str
		parname of parameter being plotted
	dfpar : str
		string name of parameter in posterior samples file

	Returns
	----------
	KEEPERS : list
		list of ISIM indices to keep
	"""
	for iim,parvalue in enumerate(GLOB_FITS):
		FITS = GLOB_FITS[parvalue]
		if iim==0:	KEEPERS = [True for _ in range(len(FITS))]
		for ISIM in range(len(KEEPERS)):
			try:
				fitsummary = FITS[ISIM]['summary']
				Rhat = fitsummary.loc[dfpar]['r_hat']
				if Rhat>Rhat_threshold:
					print (f'{loop_par}={parvalue}, ISIM={ISIM}, Rhat={Rhat}')
					KEEPERS[ISIM] = False
			except KeyError:
				KEEPERS[ISIM] = False

	KEEPERS = [ISIM for ISIM in range(len(KEEPERS)) if KEEPERS[ISIM]]
	KEEPERS = KEEPERS[:Nsim_keep]
	print ('Max index kept is:',max(KEEPERS),f'; Nsims kept=={len(KEEPERS)}')
	return KEEPERS

def trim_to_KEEPERS(GLOB_FITS,KEEPERS):
	"""
	Trim to Keepers

	Function to trim GLOB_FITS to retain only those with ISIMs in KEEPERS

	Parameters
	----------
	GLOB_FITS : dict
		{parvalue:FITS} where FITS={ISIM:FIT}
	KEEPERS : list
		list of ISIM indices to keep

	Returns
	----------
	GLOB_FITS where values==FITS are trimmed to have only ISIM in KEEPERS as keys
	"""
	for key in GLOB_FITS:
		GLOB_FITS[key] = {ISIM:GLOB_FITS[key][ISIM] for ISIM in KEEPERS}
	return GLOB_FITS

def get_sap_Ncred(loop_par,true_rho,ALT_PRIOR):
	"""
	Get Simulation-Averaged-Posterior-Summary-Style (sap) and Ncred (whether to include no. of sims in 68/95% intervals)

	Parameters
	----------
	loop_par : string
		parameter name that is being plotted

	true_rho : float
		true value of rho in simulation

	ALT_PRIOR : string or bool
		defines hyperprior used in fit

	Returns
	----------
	sap : None or 0 or 1
		if None, used Gaussian peak posterior; if 0 peaks at lower bound, if 1 peaks at upper bound

	Ncred : bool
		if False don't include N68,N95 summaries, if True, do include them.
	"""
	sap = None
	if loop_par=='rho':
		sap      = true_rho if true_rho in [0,1] else None
	if (ALT_PRIOR is False) and ((true_rho==1 and loop_par in ['sigmaRel','rel_rat','rel_rat2']) or (true_rho==0 and loop_par in ['sigmaCommon','com_rat','com_rat2'])):
			sap=0
	if (true_rho==0 and loop_par in ['rel_rat','rel_rat2']) or (true_rho==1 and loop_par in ['com_rat','com_rat2']):
			sap=1
	###GET NCRED
	Ncred = False if (sap is None and true_rho!=0.5 and (  (loop_par not in ['sigma0','sigmaRel','sigmaCommon'] and ALT_PRIOR is False) \
														or (loop_par not in ['sigma0'] and ALT_PRIOR is not False)))  else True
	if ALT_PRIOR is not False:
		if (true_rho==0 and loop_par in ['sigmaRel']) or (true_rho==1 and loop_par in ['sigmaCommon']):
			Ncred = True
	return sap,Ncred
