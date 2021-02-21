r"""
Solvers

Module containing the numerical solution to the eigenvalue problem.
Three different solvers are provided, each solving the following
matrix equation with a different method.

.. math::

	[A] \vec{x} = \frac{1}{k} [B] \vec{x}

.. data:: MAX_INNER

	Default limit of inner (flux) iterations

.. data:: MAX_OUTER

	Default limit of outer (fission source) iterations

.. data:: EPS_INNER

	Default numerical tolerance for inner (flux) convergence

.. data:: EPS_OUTER

	Default numerical tolerance for outer (fission source) convergence
"""

import numpy as np
import scipy.linalg as la
from .matrices import gauss_seidel, l2norm_1d

MAX_INNER = 500
MAX_OUTER = 1000
EPS_INNER = 1E-7
EPS_OUTER = 1E-5


class ConvergenceError(BaseException):
	pass


class BaseSolver:
	"""Base class for diffusion numerical solvers
	
	Parameters:
	-----------
	:type matA: np.ndarray(ndim=2)
	:param matA:
		LHS Destruction matrix
	
	:type matB: np.ndarray(ndim=2)
	:param matB:
		RHS Production matrix
	
	:type xguess: np.ndarray(ndim=1); optional
	:param xguess:
		Initial guess for the flux vector.
	
	:type eps_flux: float; optional
	:param eps_flux:
		Numerical tolerance for the inner (flux) convergence.
		[Default: :const:`diffusion.solvers.EPS_INNER`]
	
	:type eps_source: float; optional
	:param eps_source:
		Numerical tolerance for the outer (fission source) convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	
	:type eps_k: float; optional
	:param eps_k:
		Numerical tolerance for eigenvalue convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	"""
	def __init__(self, matA, matB, xguess=None,
	             eps_flux=EPS_INNER, eps_source=EPS_OUTER, eps_k=EPS_OUTER):
		if xguess is None:
			xguess = np.ones(len(matA))
		self.xguess = xguess
		self.eps_flux = eps_flux
		self.eps_source = eps_source
		self.eps_k = eps_k
		self.matB = matB
		self._solved = False
		self._l2norm = np.zeros(MAX_OUTER)
	
	def get_l2norm_results(self):
		"""Get the L2norm of the solution at every outer iteration
		
		Raises a ValueError if the solver has not run yet.
		
		Returns:
		--------
		:rtype: np.ndarray(ndim=1, dtype=float)
		:returns:
			L2norm of the numerical solution
		"""
		if self._solved:
			return self._l2norm
		else:
			raise ValueError("You mut run the solver to get L2norm results.")
	
	def guess_flux(self, fuel, nx, bc_left, bc_right):
		"""Guess the flux vector from one of the materials
		
		There are four predefined shapes the flux can take,
		depending on the boundary conditions. If one of them is vacuum,
		the flux will be some variation of a cosine. If they're both
		reflective, the flux will be flat.
		In 2 groups, the magnitude of the thermal group will be scaled.
		
		This also sets self.xguess to the newly calculated flux profile.
		
		Parameters:
		-----------
		:type fuel: :class:`diffusion.material.Material`
		:param fuel:
			Material with the cross sections to use for the guess
		
		:type nx: int
		:param nx:
			Number of spatial nodes
		
		:type bc_left: str
		:param bc_left:
			LHS boundary condition. in {'v' or 'r'}
		
		:type bc_right: str
		:param bc_right:
			RHS boundary condition. in {'v' or 'r'}
		
		Returns:
		--------
		:rtype: np.ndarray(shape=nx, dtype=float)
		:returns:
			Guess for the flux vector
		"""
		assert fuel.is_fissionable, \
			"Cannot guess flux: No nu_fission cross section!"
		for bc in (bc_left, bc_right):
			assert bc in ('r', 'v'), \
				"Unknown boundary condition: {}".format(bc)
		fshape = np.ones(nx)
		cx = np.pi/nx
		if bc_left == 'v' and bc_right == 'v':
			# Cosine
			f = lambda i: np.cos(cx*(i - nx/2))
		elif bc_left == 'r' and bc_right == 'v':
			# Half-cosine
			f = lambda i: np.cos(cx*i)
		elif bc_left == 'v' and bc_right == 'r':
			# Half-cosine, the other way
			f = lambda i: np.cos(cx*(nx - i))
		elif bc_left == 'r' and bc_right == 'r':
			# Flat
			f = lambda i: i
		else:
			raise NotImplementedError("Looks like I forgot a BC!")
		fshape = f(fshape)
		if fuel.ngroups == 1:
			flux = fshape
		else:
			flux = np.zeros(fuel.ngroups*nx)
			flux[:nx] = fshape
			flux[nx:] = fshape*fuel.flux_ratio()
		self.xguess = flux
		return flux
	
	def _solve_x(self, x, s, k):
		pass
	
	def solve_eigenvalue(self, kguess):
		"""Iterate to solve for the eigenvalue and eigenvector
		
		Parameter:
		----------
		:type kguess: float
		:param kguess:
			Guess for the eigenvalue
		
		Returns:
		--------
		:rtype: tuple:
		:returns:
			x: np.ndarray
				Flux vector
			s: np.ndarray
				Fission source vector
			k: float
				Eigenvalue
		"""
		if not kguess:
			kguess = 1.0
		sguess = self.matB.dot(self.xguess)
		oldx = np.array(self.xguess)
		kdiff = 1 + self.eps_k
		sdiff = 1 + self.eps_source
		c_outer = 0
		while (sdiff > self.eps_source) or (kdiff > self.eps_k):
			# Outer (Fission Source) iteration
			c_outer += 1
			
			xdiff = 1 + self.eps_flux
			c_inner = 0
			while xdiff > self.eps_flux:
				# Inner (Flux) iteration
				c_inner += 1
				newx = self._solve_x(oldx, sguess, kguess)
				xdiff = l2norm_1d(newx, oldx)
				oldx[:] = newx
				
				if c_inner >= MAX_INNER:
					raise ConvergenceError("Maximum inner iterations reached!")
			
			# Now the flux is converged at the fission source guess
			s = self.matB.dot(newx)
			sdiff = l2norm_1d(s, sguess)
			self._l2norm[c_outer-1] = sdiff
			k = kguess*s.sum()/sguess.sum()
			kdiff = abs(k - kguess)/kguess
			kguess = k
			sguess[:] = s
			
			if c_outer >= MAX_OUTER:
				raise ConvergenceError("Maximum outer iterations reached!")
		
		# can add checks here, like fission/absorption k calculation
		self._solved = True
		print("Solver converged after {} fission source iterations.".format(c_outer))
		self._l2norm = self._l2norm[:c_outer]

		return newx, s, k


class InversionSolver(BaseSolver):
	"""Solver using matrix inversion
	
	
	Parameters:
	-----------
	:type matA: np.ndarray(ndim=2)
	:param matA:
		LHS Destruction matrix.
		Caution: inverted upon instantiation.
	
	:type matB: np.ndarray(ndim=2)
	:param matB:
		RHS Production matrix
	
	:type xguess: np.ndarray(ndim=1); optional
	:param xguess:
		Initial guess for the flux vector.
	
	:type eps_flux: float; optional
	:param eps_flux:
		Numerical tolerance for the inner (flux) convergence.
		[Default: :const:`diffusion.solvers.EPS_INNER`]
	
	:type eps_source: float; optional
	:param eps_source:
		Numerical tolerance for the outer (fission source) convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	
	:type eps_k: float; optional
	:param eps_k:
		Numerical tolerance for eigenvalue convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	"""
	def __init__(self, matA, matB, xguess=None,
	             eps_flux=EPS_INNER, eps_source=EPS_OUTER, eps_k=EPS_OUTER):
		super().__init__(
			matA, matB, xguess, eps_flux, eps_source, eps_k)
		self.invA = la.inv(matA)
		del matA

		
	def _solve_x(self, x, s, k):
		return self.invA.dot(s/k)


class ScipySolver(BaseSolver):
	"""Solver using Scipy's linalg.solve
	
	Parameters:
	-----------
	:type matA: np.ndarray(ndim=2)
	:param matA:
		LHS Destruction matrix
	
	:type matB: np.ndarray(ndim=2)
	:param matB:
		RHS Production matrix
	
	:type xguess: np.ndarray(ndim=1); optional
	:param xguess:
		Initial guess for the flux vector.
	
	:type eps_flux: float; optional
	:param eps_flux:
		Numerical tolerance for the inner (flux) convergence.
		[Default: :const:`diffusion.solvers.EPS_INNER`]
	
	:type eps_source: float; optional
	:param eps_source:
		Numerical tolerance for the outer (fission source) convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	
	:type eps_k: float; optional
	:param eps_k:
		Numerical tolerance for eigenvalue convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	"""
	def __init__(self, matA, matB, xguess=None,
	             eps_flux=EPS_INNER, eps_source=EPS_OUTER, eps_k=EPS_OUTER):
		super().__init__(
			matA, matB, xguess, eps_flux, eps_source, eps_k)
		self.matA = matA
		
	def _solve_x(self, x, s, k):
		return la.solve(self.matA, s/k)


class GaussSeidelSolver(BaseSolver):
	"""Solver using a Gauss-Seidel algorithm
	
	Parameters:
	-----------
	:type matA: np.ndarray(ndim=2)
	:param matA:
		LHS Destruction matrix
	
	:type matB: np.ndarray(ndim=2)
	:param matB:
		RHS Production matrix
	
	:type xguess: np.ndarray(ndim=1); optional
	:param xguess:
		Initial guess for the flux vector.
	
	:type eps_flux: float; optional
	:param eps_flux:
		Numerical tolerance for the inner (flux) convergence.
		[Default: :const:`diffusion.solvers.EPS_INNER`]
	
	:type eps_source: float; optional
	:param eps_source:
		Numerical tolerance for the outer (fission source) convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	
	:type eps_k: float; optional
	:param eps_k:
		Numerical tolerance for eigenvalue convergence.
		[Default: :const:`diffusion.solvers.EPS_OUTER`]
	"""
	def __init__(self, matA, matB, xguess=None,
	             eps_flux=EPS_INNER, eps_source=EPS_OUTER, eps_k=EPS_OUTER):
		super().__init__(
			matA, matB, xguess, eps_flux, eps_source, eps_k)
		self.matL = np.tril(matA)
		self.matU = matA - self.matL
		del matA
	
	def _solve_x(self, x, s, k):
		return gauss_seidel(self.matL, self.matU, s, x, k)
