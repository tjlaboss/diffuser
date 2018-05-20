# Node
#
# Classes for nodes on a diffusion mesh

class Node1D(object):
	"""A one-dimensional Cartesian node
	
	Parameters:
	-----------
	ngroups:        int; number of energy groups
	fill:           Material
	dx:             float, cm; node width
	"""
	def __init__(self, ngroups, fill, dx):
		self._ngroups = ngroups
		self._check_fill(fill)
		self.fill = fill
		self.dx = dx
	
	def _check_fill(self, fill):
		assert fill.ngroups == self._ngroups, \
			"node fill has an inconsistent number of energy groups."
		assert fill.d.any(), "Diffusion coefficient must be set."
		assert fill.sigma_a.any(), "Absorption XS must be set."
		assert fill.scatter_matrix.any(), "Scatter matrix must be set."
	
	@property
	def G(self):
		return self._ngroups
	
	@property
	def D(self):
		return self.fill.d
	
	@property
	def sigmaA(self):
		return self.fill.sigma_a
	
	@property
	def scatterMatrix(self):
		return self.fill.scatter_matrix
	
	@property
	def sigmaS12(self):
		return self.fill.sigma_s12
	
	@property
	def nuSigmaF(self):
		return self.fill.nu_sigma_f
	
