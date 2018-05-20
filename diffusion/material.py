# Material
#
# Class for multigroup diffusion material

import scipy


class Material(object):
	def __init__(self, ngroups, name=""):
		self.ngroups = ngroups
		self.name = name

		self._d = scipy.zeros(ngroups)
		self._sigma_a = scipy.zeros(ngroups)
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
		self._sigma_a = sigma_a

	@scatter_matrix.setter
	def scatter_matrix(self, scatter_matrix):
		self._scatter_matrix[:, :] = scatter_matrix
		if self.ngroups == 2:
			self._sigma_s12 = self.scatter_matrix[1, 0] - \
			                  self.scatter_matrix[0, 1]

	@nu_sigma_f.setter
	def nu_sigma_f(self, nu_sigma_f):
		self._nu_sigma_f[:] = nu_sigma_f
		self._is_fissionable = bool(self._nu_sigma_f.any())

	def flux_ratio(self, bg2=0.0):
		assert self.ngroups == 2
		return self.sigma_s12/(self.d[1]*bg2 + self.sigma_a[1])

	def get_kinf(self):
		return self.get_keff(bg2=0.0)

	def get_keff(self, bg2):
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
