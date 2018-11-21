# Material
#
# Class for multigroup diffusion material

import scipy


def get_off_diagonal(matrix):
	off_diag = scipy.array(matrix, dtype=matrix.dtype)
	off_diag[scipy.diag_indices_from(matrix)] = 0
	return off_diag


class Material(object):
	"""A multigroup material for neutron diffusion 
	
	Be sure to set all of the cross sections. You can check this by running
	`Material.check_cross_sections()`, which will raise an error if any
	are zero that cannot be. nuSigmaF is allowed to be zero.
	
	Parameters:
	-----------
	ngroups:        int; number of energy groups to use
	name:           str, optional
	
	Attributes:
	-----------
	d:              array(ngroups,1), float, cm; diffusion coefficient
	sigma_a:        array(ngroups,1), float, cm^-1; macroscopic absorption xs
	nu_sigma_f:     array(ngroups,1), float, cm^-1; macroscopic nu-fission xs
	scatter_matrix: array(ngroups,ngroups), float, cm^-1; scattering matrix
	is_fissionable: Boolean; whether there is any nu-fission cross section
	
	Only in 2-group:
	----------------
	sigma_s12:      float; corrected downscatter cross section
	"""
	def __init__(self, ngroups, name=""):
		self.ngroups = ngroups
		self.name = name

		self._d = scipy.zeros(ngroups)
		self._sigma_a = scipy.zeros(ngroups)
		self._sigma_r = scipy.zeros(ngroups)
		self._scatter_matrix = scipy.zeros((ngroups, ngroups))
		self._nu_sigma_f = scipy.zeros(ngroups)

		self._is_fissionable = False
		self._sigma_s12 = None

	def __str__(self):
		rep = """\
Material: {}
  - {} groups
  - Fissionable: {}""".format(self.name, self.ngroups, self.is_fissionable)
		return rep

	@property
	def d(self):
		return self._d

	@property
	def sigma_a(self):
		return self._sigma_a

	@property
	def scatter_matrix(self):
		return self._scatter_matrix

	@property
	def sigma_s12(self):
		if self._sigma_s12 is not None:
			return self._sigma_s12
		else:
			if self.ngroups != 2:
				errstr = "sigmaS12 only applies for 2-group models."
				raise AttributeError(errstr)
			elif not self._scatter_matrix.sum():
				errstr = "Scatter matrix must be set first."
				raise ValueError(errstr)
			else:
				raise SystemError("Unknown error.")
	
	@property
	def sigma_r(self):
		return self._sigma_r

	@property
	def nu_sigma_f(self):
		return self._nu_sigma_f

	@property
	def is_fissionable(self):
		return self._is_fissionable

	@d.setter
	def d(self, d):
		self._d[:] = d

	@sigma_a.setter
	def sigma_a(self, sigma_a):
		self._sigma_r += sigma_a - self.sigma_a
		self._sigma_a = sigma_a

	@scatter_matrix.setter
	def scatter_matrix(self, scatter_matrix):
		self._scatter_matrix[:, :] = scatter_matrix
		if self.ngroups == 2:
			old_r = get_off_diagonal(scatter_matrix).sum(axis=0)
			new_r = get_off_diagonal(scatter_matrix).sum(axis=0)
			self._sigma_r += new_r - old_r
			self._sigma_s12 = self.scatter_matrix[1, 0] - \
			                  self.scatter_matrix[0, 1]

	@nu_sigma_f.setter
	def nu_sigma_f(self, nu_sigma_f):
		self._nu_sigma_f[:] = nu_sigma_f
		self._is_fissionable = bool(self._nu_sigma_f.any())
	
	def check_cross_sections(self):
		assert self.d.any(), "Diffusion coefficient must be set."
		assert self.sigma_a.any(), "Absorption XS must be set."
		assert self.scatter_matrix.any(), "Scatter matrix must be set."
	
	def flux_ratio(self, bg2=0.0):
		"""Find the fast-to-thermal flux ratio in a homogenous material
		
		Parameter:
		----------
		bg2:        float, optional; geometric buckling
					[Default: 0.0]
		
		Returns:
		--------
		float; fast-to-thermal flux ratio
		"""
		assert self.ngroups == 2
		return self.sigma_s12/(self.d[1]*bg2 + self.sigma_a[1])

	def get_kinf(self):
		"""Wrapper for get_keff() for an infinite medium"""
		return self.get_keff(bg2=0.0)

	def get_keff(self, bg2):
		"""Find the eigenvalue in a homogenous material

		Parameter:
		----------
		bg2:        float, optional; geometric buckling
					[Default: 0.0]

		Returns:
		--------
		keff:       float; the multiplication factor/eigenvalue
		"""
		if not self.is_fissionable:
			return 0.0
		elif self.ngroups == 1:
			return scipy.float64(self.nu_sigma_f/(self.sigma_a + self.d*bg2))
		elif self.ngroups == 2:
			ratio = self.flux_ratio(bg2)
			r1 = self.sigma_a[0] + self.sigma_s12
			return (self.nu_sigma_f[0] + self.nu_sigma_f[1]*ratio)/ \
			       (self.d[0]*bg2 + r1)
		else:
			errstr = "k_inf calculation is only available for 1 or 2 groups."
			raise NotImplementedError(errstr)
