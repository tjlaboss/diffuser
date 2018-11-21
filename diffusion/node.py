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
		self.fission_source = self.nuSigmaF*self.dx
		self.scatter_source = self.scatterMatrix*self.dx
	
	def _check_fill(self, fill):
		assert fill.ngroups == self._ngroups, \
			"node fill has an inconsistent number of energy groups."
		fill.check_cross_sections()
	
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
	def sigmaR(self):
		return self.fill.sigma_r
	
	@property
	def scatterMatrix(self):
		return self.fill.scatter_matrix
	
	@property
	def sigmaS12(self):
		return self.fill.sigma_s12
	
	@property
	def nuSigmaF(self):
		return self.fill.nu_sigma_f
	
	def get_Dhat(self, adjacent):
		return 2*self.D*adjacent.D/(self.D*adjacent.dx + adjacent.D*self.dx)
	
	def get_interior_equation(self, left, right):
		"""Get the matrix entries for a generic interior node
		
		Parameters:
		-----------
		left:       Node1D; the node at [i-1]
		right:      Node1D; the node at [i+1]
		
		Returns:
		--------
		a:          list of scipy.array; A-matrix entries at
		            [i-1], [i], [i+1]
		b:          scipy.array; B-matrix entry at [i]
		"""
		a = [None]*3
		dhat_left = self.get_Dhat(left)
		a[0] = -dhat_left
		dhat_right = self.get_Dhat(right)
		a[2] = -dhat_right
		a[1] = dhat_left + self.sigmaR*self.dx + dhat_right
		b = self.fission_source
		t = self.scatter_source
		return a, b, t
	
	def get_vacuum_boundary_equation(self, adjacent):
		"""Get the matrix entries for a vacuum (zero-incoming) boundary
		
		Parameter:
		----------
		adjacent:   Node1D; the node at [i+1] (if western boundary)
		            or at [i-1] (if eastern boundary)
		
		Returns:
		--------
		a:          list of scipy.array; A-matrix entries at
		            ([i-1] or [i+1]), [i]
		b:          scipy.array; B-matrix entry at [i]
		"""
		a = [None]*2
		dhat = self.get_Dhat(adjacent)
		a[0] = dhat
		leakage = (2*self.D/self.dx)/(1 + 4*self.D/self.dx)
		a[1] =  dhat + self.sigmaR*self.dx + leakage
		b = self.fission_source
		t = self.scatter_source
		return a, b, t
	
	def get_reflective_boundary_condition(self, adjacent):
		"""Get the matrix entries for a reflective (zero-current) boundary
		
		TODO: Ascertain whether `adjacent' is needed.

		Parameter:
		----------
		adjacent:   Node1D; the node at [i+1] (if western boundary)
		            or at [i-1] (if eastern boundary)

		Returns:
		--------
		a:          list of scipy.array; A-matrix entries at
		            ([i-1] or [i+1]), [i]
		b:          scipy.array; B-matrix entry at [i]
		"""
		a = [None]*2
		d2 = 2*self.D/self.dx
		a[0] = -d2
		a[1] = d2 + self.sigmaR*self.dx
		b = self.fission_source
		t = self.scatter_source
		return a, b, t
