"""
Analytic

Analytic solutions to homogeneous problems
"""

import numpy as np
from scipy.integrate import quad


class AnalyticProblem1D:
	"""Base class for analytic solutions to homogeneous 1D problems
	
	Parameters:
	-----------
	:type fill: :class:`diffusion.material.Material`
	:param fill:
		Material the medium is filled with
	
	:type width: float, cm
	:param width:
		Width or diameter of the medium
		[Default: None]
	
	Attributes:
	-----------
	:vartype bg2: float
	:ivar bg2:
		Geometric buckling (exactly 0 if infinite)
	
	:vartype ngroups: int
	:ivar ngroups:
		Number of energy groups

	:vartype mag1: float
	:ivar mag1:
		Relative magnitude of fast flux.
		Only present if `ngroups' is 2.
	
	:vartype mag2: float
	:ivar mag2:
		Relative magnitude of thermal flux.
		Only present if `ngroups' is 2.
	
	:vartype ratio: float
	:ivar ratio:
		Ratio of mag1/mag2.
	"""
	def __init__(self, fill, width=None):
		self.fill = fill
		self.ngroups = fill.ngroups
		self.width = width
		self.bg2 = self.geometric_buckling()
		if self.ngroups == 2:
			self.ratio = self.fill.flux_ratio(self.bg2)
			self.mag1 = 1.0/(1 + self.ratio)
			self.mag2 = 1.0 - self.mag1
		else:
			self.ratio = None
	
	def _assert_g(self, g):
		if g is None:
			assert self.ngroups == 1, \
				"You must specify which group."
			
	def _invalid_ngroups(self, ngroups):
		errstr = "{} groups".format(ngroups)
		raise NotImplementedError(errstr)
	
	def _invalid_group(self, g):
		errstr = "group {} of {}".format(g, self.ngroups)
		raise NotImplementedError(errstr)
	
	def geometric_buckling(self):
		pass
	
	def kinf(self):
		return self.fill.get_kinf()
	
	def keff(self):
		bg2 = self.geometric_buckling()
		return self.fill.get_keff(bg2)
	
	def flux(self, coord, g=None):
		"""Get the flux at a points

		Parameters:
		-----------
		:type coord: float, cm
		:param coord:
			Point to calculate the flux at
		
		:type g: int
		:param g:
			Energy group index.
			Must be provided if ngroups > 1.

		Returns:
		--------
		:rtype: np.ndarray
		:returns:
			Array of flux values at every point in `coordvec'.
		"""
		self._assert_g(g)
		if self.width is None:
			if g is None or self.ngroups == 1:
				return 1
			elif self.ngroups == 2:
				if g == 0:
					return self.mag1
				elif g == 1:
					return self.mag2
				else:
					self._invalid_group(g)
			else:
				self._invalid_ngroups(self.ngroups)
	
	def fission_source(self, coord):
		"""Calculate the fission source at a point
		
		Parameter:
		----------
		:type coord: float, cm
		:param coord:
			Coordinate to calculate the fission source at.

		Returns:
		--------
		:rtype: float
		:returns:
			Fission source at position `coord'
		"""
		fs = 0.0
		for g in range(self.ngroups):
			fs += self.flux(coord, g)*self.fill.nu_sigma_f[g]
		return fs
	
	def get_flux_vector(self, coordvec, g=None):
		"""Get the flux vector at several points
		
		Parameters:
		-----------
		:type coordvec: list of float, cm
		:param coordvec:
			List of points to calculate the flux at
		
		:type g: int; optional
		:param g:
			Energy group index.
			Must be provided if ngroups > 1.
		
		Returns:
		--------
		:rtype: np.ndarray
		:returns:
			Array of flux values at every point in `coordvec'.
		"""
		return np.array([self.flux(x, g) for x in coordvec])
	
	def get_fission_source_vector(self, coordvec):
		"""Calculate the fission source vector at several points
	
		Parameter:
		----------
		:type coordvec: list of float, cm
		:param coordvec:
			List of points to calculate the flux at

		Returns:
		--------
		:rtype: np.ndarray
		:returns:
			Array of fission source values at every point in `coordvec'.
		"""
		return np.array([self.fission_source(x) for x in coordvec])
	
	def get_peaking_factor(self):
		"""Calculate the analytic peaking factor
		
		Returns:
		--------
		:rtype: float
		:returns:
			Peaking factor over the slab
		"""
		if self.width is None:
			return 1
		peak = self.fission_source(0)
		integral = quad(self.fission_source, -self.width/2.0, +self.width/2.0)
		return peak*self.width/(integral[0])


class AnalyticSlab1D(AnalyticProblem1D):
	"""A one-dimensional Cartesian slab with a homogeneous medium"""
	def geometric_buckling(self):
		"""Analytic geometric buckling in a Cartesian slab
		
		Returns:
		--------
		:rtype: float, cm^-2
		:returns:
			Geometric buckling
		"""
		if self.width is None:
			# infinite
			return 0.0
		return (np.pi/self.width)**2
	
	def flux(self, x, g=None):
		"""Find the groupwise scalar flux somewhere on the slab
		
		Parameters:
		-----------
		:type x: float, cm
		:param x:
			X-coordinate to find the flux at
		
		:type g: int, optional for 1-group
		:param g:
			Which energy group's flux
		
		Returns:
		--------
		:rtype: float
		:returns:
			normalized scalar flux at x
		"""
		res = super().flux(x, g)
		if res is not None:
			return res
		# otherwise, it's finite
		extrap1 = 4*self.fill.d[0]
		flux1 = np.cos(np.pi*x/(self.width + extrap1))
		if self.ngroups == 1:
			return flux1
		elif self.ngroups == 2:
			if g == 0:
				return self.mag1*flux1
			elif g == 1:
				extrap2 = 4*self.fill.d[1]
				return self.mag2*np.cos(np.pi*x/(self.width + extrap2))
			else:
				self._invalid_group(g)
		else:
			self._invalid_ngroups(self.ngroups)

