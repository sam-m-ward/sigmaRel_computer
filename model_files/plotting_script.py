"""
Posterior Plotter Module

Module containing POSTERIOR_PLOTTER class
Used to take in posterior samples and plot up corner plot


Contains:
--------------------
Functions:
	kde(x_data, x_target, y_data=None, y_target=None, x_bounds=None, y_bounds=None, smoothing=1.0)
	finish_corner_plot(fig,ax,Lines,save,show,plotpath,savekey,colour='C0',y0=None,lines=True,oneD=False)
	get_Lines(stan_data, c_light, alt_prior, zcosmo, alpha_zhel)

POSTERIOR_PLOTTER class:
	inputs: samples, parnames, parlabels, bounds, Rhats, choices, smoothing=2

	Methods are:
		update_lims(Nsig=3)
		corner(fig_ax, Ngrid=30, colour="C0", warn_tolerance=0.05, FS=15, blind=False)
		corner_plot(verbose=True, blind=False, colour=None, multiplot=False)
		plot_1Drow(verbose=True,blind=False, colour=None, multiplot=False)

PARAMETER class:
	inputs: chain,parname,parlabel,lim,bound,Rhat,row,choices,smoothing=2,XGRID=None,oneD=False

	Methods are:
		get_xgrid(fac=0.1)
		slice_reflection_KDE()
		get_xgrid_KDE()
		get_KDE_values(location=None,value=None, return_only_index=False)
		plot_marginal_posterior(ax,colour='C0',alph=0.2,FS=14,verbose=True,blind=False, multiplot=False)
--------------------
Mostly Copied from Bird-Snack (https://github.com/sam-m-ward/birdsnack)
Written by Sam M. Ward: smw92@cam.ac.uk
"""
import copy,scipy
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

def kde(x_data, x_target, y_data=None, y_target=None, x_bounds=None, y_bounds=None, smoothing=1.0):
	"""
	Kernel Density Estimate

	Returns 1D or 2D KDE

	Parameters
	----------
	x_data: array
		x-samples

	x_target: array
		grid of x to evaluate KDE at

	y_data: array (optional; default=None)
		y-samples

	y_target: array (optional; default=None)
		grid of y to evaluate KDE at

	x_bounds: None or [lo,up] (optional; default = None)
		define prior lower and upper bounds on the x-samples

	x_bounds: None or [lo,up] (optional; default = None)
		define prior lower and upper bounds on the y-samples

	smoothing: float (optional; default=1.0)
		level of KDE smoothing

	Returns
	----------
	KDE
	"""
	if y_data is None:
		n = len(x_data)
		d = 1
	else:
		if len(x_data) == len(y_data):
			n = len(x_data)
			d = 2
		else:
			raise ValueError("Data vectors should be same length.")
	b = smoothing*n**(-1./(d+4)) #Scott Rule x Smoothing Factor
	if d==1:
		h = np.std(x_data)*b
	else:
		h = np.cov([x_data, y_data])*b**2

	x = x_target[:,None] - x_data[None,:]
	if d==2:
		y = y_target[:,None] - y_data[None,:]
		KH = scipy.stats.multivariate_normal.pdf(np.stack([x,y], axis=-1), cov=h)
		if x_bounds is not None:
			if x_bounds[0] is not None:
				x_minus = 2*x_bounds[0] - x_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_minus[None,:], y], axis=-1), cov=h)
				if y_bounds is not None:
					if y_bounds[0] is not None:
						y_minus = 2*y_bounds[0] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_minus[None,:], y_target[:,None] - y_minus[None,:]], axis=-1), cov=h)
					if y_bounds[1] is not None:
						y_plus = 2*y_bounds[1] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_minus[None,:], y_target[:,None] - y_plus[None,:]], axis=-1), cov=h)
			if x_bounds[1] is not None:
				x_plus = 2*x_bounds[1] - x_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_plus[None,:], y], axis=-1), cov=h)
				if y_bounds is not None:
					if y_bounds[0] is not None:
						y_minus = 2*y_bounds[0] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_plus[None,:], y_target[:,None] - y_minus[None,:]], axis=-1), cov=h)
					if y_bounds[1] is not None:
						y_plus = 2*y_bounds[1] - y_data
						KH += scipy.stats.multivariate_normal.pdf(np.stack([x_target[:,None] - x_plus[None,:], y_target[:,None] - y_plus[None,:]], axis=-1), cov=h)
		if y_bounds is not None:
			if y_bounds[0] is not None:
				y_minus = 2*y_bounds[0] - y_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x, y_target[:,None] - y_minus[None,:]], axis=-1), cov=h)
			if y_bounds[1] is not None:
				y_plus = 2*y_bounds[1] - y_data
				KH += scipy.stats.multivariate_normal.pdf(np.stack([x, y_target[:,None] - y_plus[None,:]], axis=-1), cov=h)
		f = np.sum(KH, axis=1)/n
	else:
		Kh = scipy.stats.norm.pdf(x, scale=h)
		if x_bounds is not None:
			if x_bounds[0] is not None:
				x_minus = 2*x_bounds[0] - x_data
				Kh += scipy.stats.norm.pdf(x_target[:,None] - x_minus[None,:], scale=h)
			if x_bounds[1] is not None:
				x_plus = 2*x_bounds[1] - x_data
				Kh += scipy.stats.norm.pdf(x_target[:,None] - x_plus[None,:], scale=h)
		f = np.sum(Kh, axis=1)/n

	return f


class POSTERIOR_PLOTTER:

	def update_lims(self, Nsig = 3):
		"""
		Update Limits

		This method updates limits to extend Nsig sigma either way of sample average and within bounds
		Limits used for KDE and visualisation

		Parameters
		----------
		Nsig : float (optional; default = 5)
			limits set to Nsig sigma either way of sample average, but ensures it doesn't cross parameter bounds

		End Product(s)
		----------
		self.lims updated
		"""
		chains,lims,bounds = self.chains,self.lims,self.bounds
		for p in range(self.Npar):
			if lims[p][0] is None:
				if bounds[p][0] is None:
					lims[p][0] = np.average(chains[p])-Nsig*np.std(chains[p])
				else:
					lims[p][0] = max([np.average(chains[p])-Nsig*np.std(chains[p]),bounds[p][0]])
			if lims[p][1] is None:
				if bounds[p][1] is None:
					lims[p][1] = np.average(chains[p])+Nsig*np.std(chains[p])
				else:
					lims[p][1] = min([np.average(chains[p])+Nsig*np.std(chains[p]),bounds[p][1]])

		self.lims = lims

	def __init__(self, samples, parnames, parlabels, bounds, Rhats, choices, smoothing=2):
		"""
		Initialisation

		Parameters
		----------
		samples  : dict
			{parname : array of samples}

		parnames : list
			[parname1, parname2...]

		parlabels : list
			[script_parname1, script_parname2...]

		bounds : list of list
			[[low_bound_1, up_bound_1],[low_bound_2, up_bound_2],...]

		Rhats : dict
			{parname : Rhat value}

		choices : dict
			plotting_parameters choices

		smoothing : float (optional; default=2)
			smoothing of KDEs for 1D and 2D marginals
		"""
		self.samples   = samples
		self.parnames  = parnames
		self.parlabels = parlabels
		self.bounds    = bounds
		self.Rhats     = Rhats
		self.choices   = choices
		self.smoothing = smoothing

		self.chains    = [self.samples[par] for par in self.parnames]
		self.Npar      = len(self.parnames)
		self.pardict   = {key:value for key,value in zip(self.parnames,self.parlabels)}
		self.lims      = [[None,None] for _ in range(self.Npar)]
		#Set limits to extend Nsig sigma either side of sample average, but ensures doesn't cross parameter boundaries
		self.update_lims()


	def corner(self, fig_ax, Ngrid=10, colour="C0", warn_tolerance=0.05, FS=15, blind=False):
		"""
		Corner Method

		Returns 2D contours

		Parameters
		----------
		fig_ax: list (optional; default=None)
			[fig,ax]

		Ngrid : int (optional; default=30)
			if using KDE for 2D contour, sets no. of grid points, 100 is intensive, 10 is fairly fast but still noticeably jagged

		colour: str (optional; default="C0")
			colour for contours

		warn_tolerance: float (optional; default=0.05)
			width in percentage (i.e. 0.05==5%) which contours need to be correct for before displaying warning message

		FS: float (optional; default=15)
			fontsize

		blind : bool (optional; default=False)
			option to mask posterior results

		End Products(s)
		----------
		fig,ax
		"""
		if colour is None: colour="C0"
		chains    = self.chains
		names     = self.parlabels
		lims      = self.lims
		bounds    = self.bounds
		quick     = self.choices['quick']
		smoothing = self.smoothing

		if len(chains) != len(names):
			raise ValueError("First dimension of input list/array should equal number of parameter names.")
		d = len(names) ; n = len(chains[0])

		fig,ax = fig_ax[:]
		for a in ax[np.triu_indices(d, k=1)]:
			a.axis("off")

		for row in range(d):
			pyrange = np.linspace(lims[row][0] - (bounds[row][0] is not None)*(lims[row][1]-lims[row][0]), lims[row][1] + (bounds[row][1] is not None)*(lims[row][1]-lims[row][0]), Ngrid*int(1 + (bounds[row][0] is not None) + (bounds[row][1] is not None)))
			ax[row,row].set_yticklabels("")
			ax[row,row].set_yticks([])
			for col in range(row):
				ax[row,col].get_shared_y_axes().remove(ax[row,row])
				if not quick:
					#PLOT THE 2D CONTOURS
					print ('Corner 2D KDE on row,col indices',row,col)
					pxrange = np.linspace(lims[col][0] - (bounds[col][0] is not None)*(lims[col][1]-lims[col][0]), lims[col][1] + (bounds[col][1] is not None)*(lims[col][1]-lims[col][0]), Ngrid*int(1 + (bounds[col][0] is not None) + (bounds[col][1] is not None)))
					pxgrid, pygrid = np.meshgrid(pxrange, pyrange)
					try:					cons = ax[row,col].contour(pxgrid, pygrid, np.reshape(kde(chains[col], pxgrid.flatten(), chains[row], pygrid.flatten(), x_bounds=bounds[col], y_bounds=bounds[row], smoothing=smoothing), pxgrid.shape), levels=25, colors=colour, alpha=0)#alpha=0.1)
					except Exception as e:	print (e)
					fracs = []
					for c, con in enumerate(cons.collections):
						paths = con.get_paths()
						if len(paths) == 1:
							fracs.append(sum(paths[0].contains_points(np.vstack([chains[col], chains[row]]).T))/n)
						elif len(paths) == 0:
							fracs.append(np.inf)
						else:
							fracs.append(sum([sum(path.contains_points(np.vstack([chains[col], chains[row]]).T)) for path in paths])/n)
					c68 = np.fabs(np.array(fracs) - 0.68).argmin()
					c95 = np.fabs(np.array(fracs) - 0.95).argmin()
					if not 0.68 - warn_tolerance < fracs[c68] < 0.68 + warn_tolerance:
						print("WARNING: Fraction of samples contained in estimated ({}, {}) 68 percent credible interval is {:.3f}, plotted contour may be suspect!".format(names[col], names[row],fracs[c68]))
					if not 0.95 - warn_tolerance < fracs[c95] < 0.95 + warn_tolerance:
						print("WARNING: Fraction of samples contained in estimated ({}, {}) 95 percent credible interval is {:.3f}, plotted contour may be suspect!".format(names[col], names[row], fracs[c95]))
					try:
						ax[row,col].contour(pxgrid, pygrid, np.reshape(kde(chains[col], pxgrid.flatten(), chains[row], pygrid.flatten(), x_bounds=bounds[col], y_bounds=bounds[row], smoothing=smoothing), pxgrid.shape), levels=[cons.levels[c95], cons.levels[c68]], colors=colour)
					except Exception as e:
						print (e)
				elif quick:
					#PLOT THE SAMPLES INSTEAD
					ax[row,col].scatter(chains[col],chains[row],alpha=0.05,color=colour,s=50)
				if col == 0:
					ax[row,col].set_ylabel(names[row],fontsize=FS)
					if lims is not None:
						ax[row,col].set_ylim(*lims[row])
				else:
					ax[row,col].set_yticklabels("")
					ax[row,col].set_yticks([])
					ax[row,col].set_ylim(ax[row,0].get_ylim())
				ax[row,col].tick_params(labelsize=FS)
				if row == d-1:
					ax[row,col].set_xlabel(names[col],fontsize=FS)
					if lims is not None:
						ax[row,col].set_xlim(*lims[col])
				if blind:
					ax[row,col].set_xticks([])
					ax[row,col].set_yticks([])
		if blind:
			ax[-1,-1].set_xticks([])
		ax[d-1,d-1].set_xlabel(names[d-1],fontsize=FS)
		ax[d-1,d-1].tick_params(labelsize=FS)
		fig.subplots_adjust(top=0.9)
		fig.subplots_adjust(wspace=0.08, hspace=0.08)#wspace=0.075, hspace=0.075)
		for col in range(d):#Helps remove overlapping axis tick numbers
			try:
				new_xlabels = [x.get_text() if float(x.get_text()) not in [0,0.1,1] else {0:'0',0.1:'0.1',1:'1'}[float(x.get_text())] for x in ax[d-1,col].get_xticklabels()]
				ax[d-1,col].set_xticklabels(new_xlabels)
			except:
				pass
		self.fig = fig
		self.ax  = ax

	def corner_plot(self,verbose=True,blind=False, colour=None, multiplot=False):
		"""
		Plot posterior samples

		Function to plot the posterior samples

		Parameters
		----------
		verbose: bool (optional; default=True)
			print out summaries

		blind : bool (optional; default=False)
			option to mask posterior results

		colour : None or str (optional; default=None)
			if None, use default value, otherwise, use input str

		multiplot : bool or list (optional; default=False)
			if not False, input should be list e.g. [0,3] meaning the first of 3 panels

		End Product(s)
		----------
		Summary_Strs: dict
			each value is the parameters' Summary_Str: the posterior summary statistic that is displayed in the plot
		"""

		bounds  = self.bounds
		chains  = self.chains
		pardict = self.pardict

		#Create figure
		if 'fig' not in self.__dict__.keys() and 'ax' not in self.__dict__.keys():
			sfigx,sfigy = 3.1*len(pardict),2.7*len(pardict)
			fig,ax = pl.subplots(len(pardict),len(pardict),figsize=(sfigx,sfigy),sharex='col',sharey=False)
		else:
			fig,ax = self.fig,self.ax
		if len(pardict)==1 and str(type(ax))=="<class 'matplotlib.axes._axes.Axes'>":#Latter condition for when multiplot so np.array conversion already applied
			ax = np.array([[ax]])

		if verbose: print ("###"*5)
		Summary_Strs = {}
		for row,parname in enumerate(self.pardict):#Get Conf Intervals Here
			parameter = PARAMETER(self.chains[row],parname,self.pardict[parname],self.lims[row],self.bounds[row],self.Rhats[parname],row,self.choices,self.smoothing)
			parameter.get_xgrid_KDE()
			#Plot 1D Marginal KDEs
			Summary_Strs[parname] = parameter.plot_marginal_posterior(ax,verbose=verbose,blind=blind,colour=colour,multiplot=multiplot)
			if verbose: print ("###"*5)
		#Plot 2D marginals
		self.corner([fig,ax],blind=blind,colour=colour)
		if verbose: print ("Corner samples/contours plotted successfully")
		return Summary_Strs

	def plot_1Drow(self,verbose=True,blind=False, colour=None, multiplot=False):
		"""
		Plot 1D Row

		Method to plot the posterior samples; plot row of 1D marginal panels
		Method very similar to corner_plot

		Parameters
		----------
		verbose: bool (optional; default=True)
			print out summaries

		blind : bool (optional; default=False)
			option to mask posterior results

		colour : None or str (optional; default=None)
			if None, use default value, otherwise, use input str

		multiplot : bool or list (optional; default=False)
			if not False, input should be list e.g. [0,3] meaning the first of 3 panels

		End Product(s)
		----------
		Summary_Strs: dict
			each value is the parameters' Summary_Str: the posterior summary statistic that is displayed in the plot
		"""

		bounds  = self.bounds
		chains  = self.chains
		pardict = self.pardict
		FS = 15

		#Create figure
		if 'fig' not in self.__dict__.keys() and 'ax' not in self.__dict__.keys():
			sfigx,sfigy = 3.1*len(pardict),2.7
			fig,ax = pl.subplots(1,len(pardict),figsize=(sfigx,sfigy),sharex='col',sharey=False)
			ax = np.array([ax]).reshape(1,-1)
		else:
			fig,ax = self.fig,self.ax
		if len(pardict)==1 and str(type(ax))=="<class 'matplotlib.axes._axes.Axes'>":#Latter condition for when multiplot so np.array conversion already applied
			ax = np.array([[ax]])

		if verbose: print ("###"*5)
		Summary_Strs = {}
		for col,parname in enumerate(self.pardict):#Get Conf Intervals Here
			parameter = PARAMETER(self.chains[col],parname,self.pardict[parname],self.lims[col],self.bounds[col],self.Rhats[parname],col,self.choices,self.smoothing,oneD=True)
			parameter.get_xgrid_KDE()
			#Plot 1D Marginal KDEs
			Summary_Strs[parname] = parameter.plot_marginal_posterior(ax,verbose=verbose,blind=blind,colour=colour,multiplot=multiplot)
			if verbose: print ("###"*5)
			ax[0,col].tick_params(labelsize=FS)
			ax[0,col].set_xlabel(self.pardict[parname],fontsize=FS)
			ax[0,col].set_xlim(*self.lims[col])
			ax[0,col].set_yticks([])
			if blind:
				ax[0,col].set_xticks([])
			else:
				new_xlabels = [x if float(x.get_text()) not in [0,0.1,1] else {0:'0',0.1:'0.1',1:'1'}[float(x.get_text())] for x in ax[0,col].get_xticklabels()]
				ax[0,col].set_xticklabels(new_xlabels)
				#ax[0,col].set_ylim([0,ax[0,col].get_ylim()[1]])
		ax[0,0].set_ylabel('Posterior Density',fontsize=FS)
		self.fig = fig
		self.ax  = ax
		if verbose: print ("1D marginals plotted plotted successfully")
		return Summary_Strs

class PARAMETER:

	def get_xgrid(self,fac=0.1):
		"""
		Get xgrid

		Get grid of points for KDE

		Parameters
		----------
		fac : float (optional; default=2)
			factor determines number of grid points: Ngrid = fac*Nsamps

		Returns
		----------
		xgrid : array
			grid points for KDE
		"""
		if self.XGRID is None:
			lim   = self.lim
			bound = self.bound
			Ngrid = int(self.Nsamps*fac)
			#If Bounds is not None, make xgrid larger, to allow for reflection KDE (i.e. KDE extends beyond prior boundary, then is reflected and doubled over within boundary)
			xgrid = np.linspace(lim[0] - (bound[0] is not None) * (lim[1]-lim[0]),
								lim[1] + (bound[1] is not None) * (lim[1]-lim[0]),
								Ngrid*int(1 + (bound[0] is not None) + (bound[1] is not None))
								)
			self.xgrid = xgrid
		else:
			self.xgrid = np.linspace(self.XGRID[0],self.XGRID[1],self.XGRID[2])

	def slice_reflection_KDE(self):
		"""
		Slice Reflection KDE

		Reflection KDEs go outside parameter bounds; this method trims to reside within bounds

		End Product(s)
		----------
		self.xgrid,KDE trimmed to reside within parameters bounds
		"""
		xgrid, KDE, bound = self.xgrid, self.KDE, self.bound
		#Having got a full mirror reflected kde, slice down to half kde bounded by lower support
		if bound[0] != None:
			lobound = 0
			for i,xi in enumerate(xgrid):
				if xi>=bound[0]:
					lobound = i
					break
			xgrid, KDE = xgrid[lobound:], KDE[lobound:]
		#Having got a full mirror reflected kde, slice down to half kde bounded by upper support
		if bound[1] != None:
			upbound = len(xgrid)
			for i,xi in enumerate(xgrid):
				if xi>=bound[1]:
					upbound = i
					break
			xgrid, KDE = xgrid[:upbound], KDE[:upbound]

		self.xgrid = xgrid
		self.KDE   = KDE

	def __init__(self,chain,parname,parlabel,lim,bound,Rhat,row,choices,smoothing=2,XGRID=None,oneD=False):
		"""
		See POSTERIOR_PLOTTER class docstring for input descriptions
		"""
		self.chain     = chain
		self.parname   = parname
		self.parlabel  = parlabel
		self.lim       = lim
		self.bound     = bound
		self.Rhat      = Rhat
		self.row       = row
		self.choices   = choices
		self.smoothing = smoothing #Smoothing for 1D marginal KDE
		self.XGRID     = XGRID
		self.oneD      = oneD

		self.Nsamps      = len(self.chain)
		self.samp_median = np.median(self.chain)
		self.samp_std    = np.std(self.chain)

		self.dfchain = pd.DataFrame(data={'par':chain}).sort_values(ascending=True,by='par')


	def get_xgrid_KDE(self):
		"""
		Get xgrid KDE

		Gets trimmed grid of points for KDE that resides within parameter boundaries
		Gets reflection KDE taking into account prior boundaries

		End Product(s)
		----------
		self.xgrid, KDE
		"""
		#Get grid for KDE
		self.get_xgrid()
		#Get full reflection KDE
		self.KDE = kde(self.chain if str(type(self.chain))!="<class 'pandas.core.series.Series'>" else self.chain.values, self.xgrid, x_bounds=self.bound, smoothing=self.smoothing)
		#Trim xgrid,KDE to reside within boundaries
		self.slice_reflection_KDE()

	def get_KDE_values(self,location=None,value=None, return_only_index=False):
		"""
		Get KDE values

		Method to get array location/values given some parameter value

		Parameters
		----------
		location : str (optional; default=None)
			if 'mode', use mode of KDE

		value : float (optionl; default=None)
			if float, get closest grid location to this parameter value
			checks are performed to ensure this parameter value resides within parameter boundaries

		return_only_index : bool (optional; default=False)
			bool dicatates if only grid index is returned

		Returns
		----------
		grid_index, xgrid_loc, KDE_height
		if return_only_index: return grid_index only
		"""
		#Error check
		if location is not None:
			if location not in ['mode']:
				raise Exception("To get KDE feature at a location, set location to valid value such as location='mode'")
		if value is not None:
			if self.bound[0] is not None:
				assert(self.bound[0]<=value)
			if self.bound[1] is not None:
				assert(value<=self.bound[1])
		if location is not None and value is not None:
			raise Exception('Choose either to look at KDE location OR numerical value, not both')

		#Get grid location
		if location=='mode':
			grid_index = np.argmax(self.KDE)
		elif value is not None:
			grid_index = np.argmin(np.abs(self.xgrid-value))
		#Return x,y values
		KDE_height = self.KDE[grid_index]
		xgrid_loc  = self.xgrid[grid_index]
		if return_only_index:
			return grid_index
		else:
			return grid_index, xgrid_loc, KDE_height

	def plot_marginal_posterior(self,ax,colour='C0',alph=0.2,FS=14,verbose=True, blind=False, multiplot=False):
		"""
		Plot Marginal Posterior

		Function to plot the 1D marginal panels in the posterior

		Parameters
		----------
		ax : ax of figure
			ax[row,row] = marginal panels

		colour : str (optional; default='C0')
			colour of KDE

		alph : float (optional; default=0.2)
			alpha of plot items (e.g KDE fill_between)

		FS : float
			fontsize

		verbose: bool (optional; default=True)
			if True print summaries

		blind : bool (optional; default=False)
			option to mask posterior results

		multiplot : bool or list (optional; default=False)
			if not False, input should be list e.g. [0,3] meaning the first of 3 panels

		Returns
		----------
		Summary_Str : str
		   Posterior summary for table
		"""
		if colour is None: colour="C0"
		paperstyle = self.choices['paperstyle']
		chain      = self.chain
		row        = self.row
		parname    = self.parname
		parlabel   = self.parlabel
		KDE        = self.KDE
		xgrid      = self.xgrid
		#Initialisations
		y0 = len(ax)-0.85 ;
		delta_y = 0.15 ; Summary_Str = ''
		#For multiplot
		xrow = row*(not self.oneD)+0*self.oneD
		x0 = 0.5; dy = 0.15-0.03*(len(ax)<4) ; y0 = 0.05 ; ha = 'center'
		if not paperstyle: ax[xrow,row].set_title(r'$\hat{R}$('+self.parlabel.split(' (')[0]+f') = {self.Rhat:.3}')

		#Plot KDE
		ax[xrow,row].plot(xgrid, KDE, color=colour)
		ax[-1,row].set_xlim(*self.lim)

		#KDE doesnt peak at prior boundary
		condition1 = np.argmax(KDE)!=0 and np.argmax(KDE)!=len(KDE)-1
		#KDE is not near flat topped at prior boundary
		hh = np.exp(-1/8)#Gaussian height at sigma/2
		imode, xmode, KDEmode = self.get_KDE_values(location='mode')
		condition2 = not (KDE[0]>=hh*KDEmode or KDE[-1]>=hh*KDEmode)

		condition3=False ; Ncrosses=0
		if parname in ['rho','rel_rat','com_rat','rel_rat2','com_rat2']:#Check for strong bimodality in 1D KDE ('strong' assessed by step size in gradient)
			step = int(0.1*len(xgrid))
			dKDE = KDE[::step][1:] - KDE[::step][:-1]
			KDE_extent = np.amax(KDE)-np.amin(KDE)
			crosses = [i for i,k1,k2 in zip(np.arange(len(dKDE)-1),dKDE[:-1],dKDE[1:]) if k1/k2<0 and abs(k1-k2)/KDE_extent>0.01]#i.e. change is greater than 1%
			#if len(crosses)>1:
			Ncrosses = len(crosses)
			condition3 = True
			'''#Plot gradient change detector
			pl.figure()
			for i in crosses:pl.scatter(np.array([xgrid[::step][i],xgrid[::step][i+1]])++(xgrid[step]-xgrid[0]),[dKDE[i],dKDE[i+1]])
			pl.plot(xgrid,KDE,label='KDE')
			pl.plot(xgrid[::step][:-1]+(xgrid[step]-xgrid[0]),dKDE,label='smoothed dKDE')
			pl.plot(xgrid[:-1],KDE[1:]-KDE[:-1],label='KDE[1:]-KDE[:-1]')
			pl.legend()
			pl.show()
			#'''

		#If typical Gaussian-like posterior, plot median, 16 and 84th percentiles
		if verbose:
			if parname=='sigma0':
				print ('p_sigma0_0.094:',round(100*sum([1 for s in self.dfchain.par if s<0.094])/len(self.dfchain.par),3),'%')
			print (f"5%, 50%, 68%, 95% quantiles: {round(self.dfchain.par.quantile(0.05),2)}, {round(self.dfchain.par.quantile(0.5),2)}, {round(self.dfchain.par.quantile(0.68),2)},{round(self.dfchain.par.quantile(0.95),2)}")
		if condition1 and condition2 and Ncrosses<2:
			if verbose: print (f"{parname}: {round(self.samp_median,2)} +/- {round(self.samp_std,2)}; 16th and 84th intervals: -{round(self.dfchain.par.quantile(0.5)-self.dfchain.par.quantile(0.16),2)}+{round(self.dfchain.par.quantile(0.84)-self.dfchain.par.quantile(0.5),2)}")
			#Plot median and 16th 84th quantiles
			i_med, x_med, KDE_med = self.get_KDE_values(value=self.samp_median)
			ax[xrow,row].plot(np.ones(2)*x_med,[0,KDE_med],c=colour)
			i_16 = self.get_KDE_values(value=self.dfchain.par.quantile(0.16),return_only_index=True)
			i_84 = self.get_KDE_values(value=self.dfchain.par.quantile(0.84),return_only_index=True)+1
			ax[xrow,row].fill_between(xgrid[i_16:i_84],np.zeros(i_84-i_16),KDE[i_16:i_84],color=colour,alpha=alph)
			if not paperstyle:
				pl.annotate("%s ="%parlabel.split(' (')[0],                     xy=(0.3  ,y0-delta_y*(row+1)),xycoords='axes fraction',fontsize=FS,color=colour,ha='right')#String broken up for alignment
				if parname=='sigmapec':
					summary    = str(int(round(self.samp_median,0)))
				else:
					summary    = "{:.2g}".format(self.samp_median)
				dec_places = len(summary.split('.')[1]) if '.' in summary else 0
				pl.annotate(summary,   xy=(0.65 ,y0-delta_y*(row+1)),xycoords='axes fraction',fontsize=FS,color=colour,ha='right')
				pl.annotate("$\pm$ %s"%round(self.samp_std,dec_places),xy=(0.665,y0-delta_y*(row+1)),xycoords='axes fraction',fontsize=FS,color=colour,ha='left')
				#Summary_Str = "$%s\pm%s$"%(round(self.samp_median,2),round(self.samp_std,2))
				Summary_Str = f'${self.samp_median:.2f}\pm{self.samp_std:.2f}$'
				if verbose: print (Summary_Str)
			elif paperstyle:
				if parname=='sigmapec':
					summary = [int(round(x,0)) for x in [self.samp_median, self.dfchain.par.quantile(0.84)-self.samp_median,self.samp_median-self.dfchain.par.quantile(0.16)]]
				else:
					summary = ["{:.3f}".format(x) for x in [self.samp_median, self.dfchain.par.quantile(0.84)-self.samp_median,self.samp_median-self.dfchain.par.quantile(0.16)]]
				if blind: summary = ['0.X','0.X','0.X']
				Summary_Str = f"${summary[0]}^{str('{')}+{summary[1]}{str('}')}_{str('{')}-{summary[2]}{str('}')}$"

				if not multiplot:	ax[xrow,row].set_title(parlabel.split(' (')[0] + " = $%s ^{+%s}_{-%s}$"%(summary[0],summary[1],summary[2]), fontsize=FS)
				else:
					counter,Npanels = multiplot[:]
					if counter!=-1:#If it is equal -1 this is code for don't annotate
						ax[xrow,row].annotate(parlabel.split(' (')[0] + " = $%s ^{+%s}_{-%s}$"%(summary[0],summary[1],summary[2]),xy=(x0,1+y0+dy*(Npanels-counter-1)),xycoords='axes fraction', ha=ha,fontsize=FS,color=colour)
				#ax[row,row].set_title(parlabel + " = {:.2f} $\pm$ {:.2f}".format(self.samp_median, self.samp_std), fontsize=FS)
				#summary = ["{:.2f}".format(x) for x in [self.samp_median, self.samp_std,self.dfchain.par.quantile(0.95),self.dfchain.par.quantile(0.05)]]
				#Summary_Str = f"${summary[0]}\\pm {summary[1]}^{str('{')}\\,\\,{summary[2]}{str('}')}_{str('{')}\\,\\,{summary[3]}{str('}')}$"
		#Otherwise, posterior peaks at/near prior boundary, choose to summarise posterior using quantiles
		else:
			if condition3 and Ncrosses>=2:#Bimodal rho posterior
				storeinfo={0.5:self.dfchain.par.quantile(0.5)}
				#Plot median
				null, x_med, KDE_med = self.get_KDE_values(value=self.dfchain.par.quantile(0.5))
				ax[xrow,row].plot(np.ones(2)*x_med,[0,KDE_med],c=colour)
				#Plot rho=0.5
				i_conf, x_conf, KDE_conf = self.get_KDE_values(value=0.5)
				ax[xrow,row].plot(np.ones(2)*x_conf,[0,KDE_conf],c=colour,linestyle='--')
				ax[xrow,row].fill_between(xgrid[0:i_conf],np.zeros(i_conf),KDE[:i_conf],color=colour,alpha=alph)
				text_height = self.get_KDE_values(value=0.25)[2]/2
				if paperstyle:
					if blind:
						p_rhohalf  = 0.5
						summarystr = parlabel.split(' (')[0] + f"0.X"
						ax[xrow,row].annotate(str("X%"),xy=(x_conf/2,text_height),color=colour,fontsize=FS+1,weight='bold',ha='center')
					else:
						p_rhohalf   = int(round(100*sum([1 for s in self.dfchain.par if s<0.5])/len(self.dfchain.par),0))
						summarystr  = 'Median-'+parlabel.split(' (')[0] + f"={storeinfo[0.5]:.2f}"
						if not multiplot:
							summarystr += '\n'
							summarystr += 'p('+parlabel.split(' (')[0]+'<0.5)='+f'{p_rhohalf}%'
						ax[xrow,row].annotate(str(int(p_rhohalf))+str("%"),xy=(x_conf/2,text_height),color=colour,fontsize=FS+1,weight='bold',ha='center')

					if not multiplot:
						ax[xrow,row].set_title(summarystr, fontsize=FS)
					else:
						counter,Npanels = multiplot[:]
						if counter!=-1:#If it is equal -1 this is code for don't annotate
							ax[xrow,row].annotate(summarystr,xy=(x0,1+y0+dy*(Npanels-counter-1)),xycoords='axes fraction', ha=ha,fontsize=FS,color=colour)
				print (paperstyle, blind, )
				Summary_Str = f"{storeinfo[0.5]:.2f} & {p_rhohalf}\%"
				if verbose:
					print (f"{parname} {storeinfo[0.5]:.2f} {p_rhohalf}")
			else:
				storeinfo = {}
				for ic,conf in enumerate([0.68,0.95]):
					if imode>0.5*(len(xgrid)-1):#If peaks at RHS
						CONF = copy.deepcopy(1-conf)
						lg = '>'
						irange = [None,len(xgrid)]
					else:
						CONF = copy.deepcopy(conf)
						irange = [0,None]
						lg = '<'

					storeinfo[conf] = self.dfchain.par.quantile(CONF)

					i_conf, x_conf, KDE_conf = self.get_KDE_values(value=self.dfchain.par.quantile(CONF))
					irange = [i if i is not None else i_conf for i in irange]

					ax[xrow,row].plot(np.ones(2)*x_conf,[0,KDE_conf],c=colour)
					ax[xrow,row].fill_between(xgrid[irange[0]:irange[1]],np.zeros(irange[1]-irange[0]),KDE[irange[0]:irange[1]],color=colour,alpha=alph*(1-0.5*ic)*0.5)#half factor because gets doubled over
					if not multiplot:#Dont plot 68% annotations for multiplot because it gets messy
						#For RHS Boundary
						if irange[-1]==len(xgrid):
							ax[xrow,row].annotate(str(int(round(CONF*100,0)))+str("%"),xy=(x_conf,KDE_conf+0.08*KDEmode),color=colour,fontsize=FS+1,weight='bold',ha='right' if ic==0 or ic==1 and i_conf/len(xgrid)>0.1 else 'left')
						#For LHS Boundary
						elif irange[0]==0:         ax[xrow,row].annotate(str(int(round(CONF*100,0)))+str("%"),xy=(x_conf,KDE_conf),color=colour,fontsize=FS+1,weight='bold')

					if not paperstyle:
						pl.annotate("%s %s"%(parlabel,lg),xy=(0.3  ,y0-delta_y*(row+1)),xycoords='axes fraction',fontsize=FS,color=colour,ha='right')#{'<':'right','>':'left'}[lg])
						if ic==0: pl.annotate("{:.3f}".format(x_conf),  xy=(0.65 ,y0-delta_y*(row+1)),xycoords='axes fraction',fontsize=FS,color=colour,ha='right')
						if ic==1: pl.annotate("({:.3f})".format(x_conf),xy=(0.735,y0-delta_y*(row+1)),xycoords='axes fraction',fontsize=FS,color=colour,ha='left')

				if paperstyle:
					if blind:
						summarystr = parlabel.split(' (')[0] + f" {lg} 0.X (0.X)"
					else:
						if 'sigma' in parlabel and 'sigma' in parname:
							if parname=='sigmapec':
								summarystr = parlabel.split(' (')[0] + f" {lg} {storeinfo[0.68]:.0f} ({storeinfo[0.95]:.0f})"
							else:
								summarystr = parlabel.split(' (')[0] + f" {lg} {storeinfo[0.68]:.3f} ({storeinfo[0.95]:.3f})"
						else:					summarystr = parlabel.split(' (')[0] + f" {lg} {storeinfo[0.68]:.2f} ({storeinfo[0.95]:.2f})"
					if not multiplot:	ax[xrow,row].set_title(summarystr, fontsize=FS)
					else:
						counter,Npanels = multiplot[:]
						if counter!=-1:#If it is equal -1 this is code for don't annotate
							ax[xrow,row].annotate(summarystr,xy=(x0,1+y0+dy*(Npanels-counter-1)),xycoords='axes fraction', ha=ha,fontsize=FS,color=colour)

				if parname=='rho':
					Summary_Str = f"{self.dfchain.par.quantile(0.5):.2f} & {int(round(100*sum([1 for s in self.dfchain.par if s<0.5])/len(self.dfchain.par),0))}\%"+f" & ${lg} {storeinfo[0.68]:.2f} ({storeinfo[0.95]:.2f})$"
				else:
					Summary_Str = f"${lg} {storeinfo[0.68]:.2f} ({storeinfo[0.95]:.2f})$"
				if verbose:
					if parname=='rho':
						print ('Median rho:',self.dfchain.par.quantile(0.5))
						print ('p_rhohalf:',int(round(100*sum([1 for s in self.dfchain.par if s<0.5])/len(self.dfchain.par),0)))
					print (f"{parname} {lg} {storeinfo[0.68]:.2f} ({storeinfo[0.95]:.2f})")

		if not multiplot:
			ax[xrow,row].set_ylim([0,None])
		else:
			if multiplot[1]-1==multiplot[0]:#if last panel
				y = ax[xrow,row].get_ylim()
				ax[xrow,row].set_ylim([0,y[1]])

		return Summary_Str


def finish_corner_plot(fig,ax,Lines,save,show,plotpath,savekey,colour='C0',y0=None,lines=True,oneD=False):
	"""
	Finish Corner Plot

	Simple function to complete corner plot

	Parameters
	----------
	fig,ax : of pl.subplots()

	Lines : list of str
		each line is analysis choice

	save,show : bools
		whether, to save plot, show plot

	plotpath : str
		path/to/plot/save/location

	savekey : str
		plot filname

	colour : str (default='C0')
		colour for lines

	y0 : None or float (default=None)
		defines height of lines, if None, use len(ax)

	lines : bool (optional; default=True)
		if True, plot lines

	oneD : bool (optional; default=False)
		if True, activates 1D row plot functions

	End Product(s)
	----------
	plot that is saved and/or shown
	"""
	if colour is None: colour='C0'
	delta_y = 0.15
	DX      = 0 + (len(ax)==1)*1.1
	y0      = len(ax) if y0 is None else y0
	if not oneD:
		pl.annotate(r"sigmaRel_computer",     xy=(0.975+0.075*(len(ax)<3)+DX,y0-0.025),xycoords='axes fraction',fontsize=20-2*(len(ax)<3),color='black',weight='bold',ha='right',fontname='Courier New')
	if lines:
		for counter,line in enumerate(Lines):#og fontsize=15
			pl.annotate(line, xy=(1+DX,y0-0.35-delta_y*(counter-1)),xycoords='axes fraction',fontsize=15,color=colour,ha='right')#weight='bold'
	fig.subplots_adjust(top=0.9)#These lines can change axes labels, so ensure these lines are done before axis tick adjustments that are done in corner method (or are include here but are the same as just described)
	fig.subplots_adjust(wspace=0.08, hspace=0.08)
	if save:
		pl.savefig(f"{plotpath}{savekey}.pdf",bbox_inches='tight')
	if show:
		pl.show()

def get_Lines(stan_data, c_light, alt_prior, zcosmo, alpha_zhel):
	"""
	Get Lines

	Function to get each str analysis choice line for plots

	Parameters
	----------
	stan_data : dict
		dictionary of data used in model fit

	c_light : float
		speed of light to convert pec_unity to real value

	alt_prior : bool
		choice of sigmaCommon prior

	zcosmo : str
		choice of zHD or zcmb

	alpha_zhel : bool (optional; default=False)
		if zmarg is True, then alpha_zhel can be activated. This takes the pre-computed slopes of dmu_phot = alpha*dzhelio and marginalises over this in the z-pipeline

	Returns
	----------
	Lines : list
		list of str for plotting
	"""
	############################################################

	Ng  = stan_data['Ng']
	S_g = stan_data['S_g']

	fixed_sigmaRel     = bool(stan_data['sigmaRel_input'])
	eta_sigmaRel_input = stan_data['eta_sigmaRel_input']
	if 'sigma0' in stan_data:
		fixed_sigma0 = True		;	sigma0 = stan_data['sigma0']
	else:
		fixed_sigma0 = False	;	sigma0 = None

	if 'pec_unity' in stan_data:
		fixed_sigmapec = True	;	sigmapec = str(int(float(stan_data['pec_unity'])*c_light/1e3))
	else:
		fixed_sigmapec = False	;	sigmapec = None

	if 'mu_ext_gal' in stan_data:
		use_external_distances = True	;	muextstr = r'Used $%s$-Distances'%({'zHD':'\\hat{z}_{\\rm{HD}}','zcmb':'\\hat{z}_{\\rm{CMB}}'}[zcosmo])
	elif 'zhelio_hats' in stan_data:
		use_external_distances = True   ;   muextstr = r'Modelled $%s$ Parameters'%({'zHD':'z_{\\rm{HD}}','zcmb':'z_{\\rm{CMB}}'}[zcosmo])
	else:
		use_external_distances = False	;	muextstr = 'No Ext. Distances'

	if alpha_zhel:
		alpha_zhel_str = r'Modelled $\epsilon_{\mu} = \hat{\alpha} \epsilon_{z_{\rm{Helio}}}$ for %s Galaxies'%(stan_data['Nzhelgal'])

	def add_siblings_galaxies(Ng, S_g):
		if type(S_g) in [float, int]:
			Line = f"$N_{{Gal}}$({S_g} Siblings) = {Ng}"
			Lines = [Line]
		else:
			if list(S_g).count(S_g[0])==len(S_g):
				Line = f"$N_{{Gal}}\,$({S_g[0]} Sibs./Gal) = {Ng}"
				Lines = [Line]
			else:
				Sgs = list(set(S_g))
				Sgs.sort()
				counts = {sg:list(S_g).count(sg) for sg in Sgs}
				upper = "("
				for sg in Sgs:
					upper += str(sg) + ', '
				upper = upper[:-2]+')'
				Line  = r"$N^{%s}_{Gal} = ("%upper
				for sg in counts:
					Line += str(counts[sg])+', '
				Lines = [Line[:-2] + ")$"]
		return Lines

	def add_sigma_Line(Lines, fixed_sigma, substr, value, upper_bound, units):
		if fixed_sigma:
			Line = r"$\sigma^{*}_{%s} = {%s}\,$%s"%(substr, value, units)
		else:
			Line = r"$\sigma_{%s} \sim U(0,%s)$"%(substr, upper_bound)
		Lines.append(Line)
		return Lines

	def get_sigmaRelstr(fixed_sigmaRel, eta_sigmaRel_input):
		if not fixed_sigmaRel:
			return None
		else:
			if eta_sigmaRel_input == 1:
				return '\sigma_0'
			elif eta_sigmaRel_input == 0.75:
				return '3\sigma_0 / 4'
			elif eta_sigmaRel_input == 0.5:
				return '\sigma_0 / 2'
			elif eta_sigmaRel_input == 0.25:
				return '\sigma_0 / 4'
			elif eta_sigmaRel_input == 0:
				return '0'
			else:
				return f"{eta_sigmaRel_input}\sigma_0"

	sigmaRelstr = get_sigmaRelstr(fixed_sigmaRel, eta_sigmaRel_input)

	Lines = add_siblings_galaxies(Ng, S_g)
	if alt_prior==False:
		Lines = add_sigma_Line(Lines, fixed_sigma0, '0', sigma0, 1.0, 'mag')
		Lines.append(r"$\rho \sim \rm{Arcsine}(0,1)$")
	elif alt_prior==True:
		Lines = add_sigma_Line(Lines, fixed_sigmaRel, '\\rm{Rel}',sigmaRelstr, 1.0, 'mag')
		Lines = add_sigma_Line(Lines, fixed_sigmaRel, '\\rm{Common}', sigmaRelstr, 1.0, 'mag')
	elif alt_prior=='A':
		Lines = add_sigma_Line(Lines, fixed_sigma0, '0', sigma0, 1.0, 'mag')
		Lines.append(r"$\sigma_{\rm{Rel}} \sim U(0,\sigma_0)$")
	elif alt_prior=='B':
		Lines = add_sigma_Line(Lines, fixed_sigma0, '0', sigma0, 1.0, 'mag')
		Lines.append(r"$\sigma_{\rm{Common}} \sim U(0,\sigma_0)$")
	elif alt_prior=='C':
		Lines = add_sigma_Line(Lines, fixed_sigma0, '0', sigma0, 1.0, 'mag')
		Lines.append(r"$\rho \sim U(0,1)$")
	else:#Alt priors from list
		Lines = add_sigma_Line(Lines, fixed_sigma0, '0', sigma0, 1.0, 'mag')
		Lines = add_sigma_Line(Lines, fixed_sigmaRel, '\\rm{Rel}',sigmaRelstr, '\sigma_0', 'mag')
	if use_external_distances:
		Lines = add_sigma_Line(Lines, fixed_sigmapec, '\\rm{pec}',sigmapec, 'c', 'km s$^{-1}$')
	Lines.append(muextstr)
	if alpha_zhel:
		Lines.append(alpha_zhel_str)

	return Lines
