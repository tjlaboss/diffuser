# Solvers
#
# Module containing the numerical solution to the eigenvalue problem.
# Three different solvers are provided, each solving the matrix equation
# [A]x = 1/k*[B]x  with a different method.


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
		fuel:       Material with the cross sections to use for the guess
		nx:         int; number of spatial nodes
		bc_left:    str; LHS boundary condition. in {'v' or 'r'}
		bc_right:   str; RHS boundary condition. in {'v' or 'r'}
		
		Returns:
		--------
		flux:       array(shape=nx, dtype=float)
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
		kguess:         float; guess for the eigenvalue
		
		Returns:
		--------
		x:              array; eigenvector
		s:              array; fission source vector
		k:              float; eigenvalue
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
	"""Flux and eigenvalue solver using matrix inversion
	
	Required Parameters:
	--------------------
	matA:           array; the destruction matrix
	matB:           array; the fission matrix
	
	Optional Parameters:
	--------------------
	xguess:         array; guess for the flux vector.
	                Will be overwritten by InversionSolver.guess_flux().
	                [Default: None]
	eps_flux:       float; tolerance on the inner (flux) l2 norm
	                [Default: 1E-7]
	eps_source:     float; tolerance on the outer (fission source) l2 norm
	                [Default: 1E-5]
	eps_k:          float; tolerance on the eigenvalue (k) l2 norm
	                [Default: 1E-5]
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
	"""Flux and eigenvalue solver using scipy.linalg

	Required Parameters:
	--------------------
	matA:           array; the destruction matrix
	matB:           array; the fission matrix

	Optional Parameters:
	--------------------
	xguess:         array; guess for the flux vector.
	                Will be overwritten by ScipySolver.guess_flux().
	                [Default: None]
	eps_flux:       float; tolerance on the inner (flux) l2 norm
	                [Default: 1E-7]
	eps_source:     float; tolerance on the outer (fission source) l2 norm
	                [Default: 1E-5]
	eps_k:          float; tolerance on the eigenvalue (k) l2 norm
	                [Default: 1E-5]
	"""
	def __init__(self, matA, matB, xguess=None,
	             eps_flux=EPS_INNER, eps_source=EPS_OUTER, eps_k=EPS_OUTER):
		super().__init__(
			matA, matB, xguess, eps_flux, eps_source, eps_k)
		self.matA = matA
		
	def _solve_x(self, x, s, k):
		return la.solve(self.matA, s/k)


class GaussSeidelSolver(BaseSolver):
	"""Flux and eigenvalue solver using a custom Gauss-Seidel solver

	Required Parameters:
	--------------------
	matA:           array; the destruction matrix
	matB:           array; the fission matrix

	Optional Parameters:
	--------------------
	xguess:         array; guess for the flux vector.
	                Will be overwritten by GaussSeidelSolver.guess_flux().
	                [Default: None]
	eps_flux:       float; tolerance on the inner (flux) l2 norm
	                [Default: 1E-7]
	eps_source:     float; tolerance on the outer (fission source) l2 norm
	                [Default: 1E-5]
	eps_k:          float; tolerance on the eigenvalue (k) l2 norm
	                [Default: 1E-5]
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
