"""
Functions for dealing with the diffusion matrices
"""

import numpy as np


def populate_matrices(node_list, ngroups, bc_left, bc_right):
	"""
	
	Parameters:
	-----------
	:param node_list: list of :class:`diffusion.node.Node1D`
	:param node_list:
		List of nodes, from left to right.
	
	:param ngroups: int
	:param ngroups:
		Number of energy groups
	
	:type bc_left: str
	:param bc_left:
		Left (west) boundary condition.
		'r' for reflective, or 'v' for vacuum.
	
	:type bc_right: str
	:param bc_right:
		Right (east) boundary condition.
		'r' for reflective, or 'v' for vacuum.

	Returns:
	--------
	:rtype: tuple:
		:returns:
			matrixA: np.ndarray
				LHS Destruction matrix
			matrixB: float
				RHS Production matrix
	"""
	nx = len(node_list)
	size = nx*ngroups
	# Matrix containing the entries for A; will be reshaped later.
	templateA = np.zeros((nx, nx, ngroups))
	templateB = np.zeros((nx, nx, ngroups))
	templateT = np.zeros((nx, nx, ngroups, ngroups))
	for i, node_c in enumerate(node_list):
		if i == 0:
			# LHS (West) boundary
			node_r = node_list[1]
			if bc_left == 'r':
				(a1, a0), b0, t0 = \
					node_c.get_reflective_boundary_condition(node_r)
			elif bc_left == 'v':
				(a1, a0), b0, t0 = \
					node_c.get_vacuum_boundary_equation(node_r)
			else:
				errstr = "West boundary condition: {}".format(bc_left)
				raise NotImplementedError(errstr)
			templateA[0, 0, :] = a0
			templateA[0, 1, :] = a1
			templateB[0, 0, :] = b0
			templateT[0, 0, :, :] = t0
		elif i == nx-1:
			# RHS (East) boundary
			node_l = node_list[-2]
			if bc_right == 'r':
				(an_1, an), bn, tn = \
					node_c.get_reflective_boundary_condition(node_l)
			elif bc_right == 'v':
				(an_1, an), bn, tn = \
					node_c.get_vacuum_boundary_equation(node_l)
			else:
				errstr = "East boundary condition: {}".format(bc_left)
				raise NotImplementedError(errstr)
			templateA[nx-1, nx-1, :] = an
			templateA[nx-1, nx-2, :] = an_1
			templateB[nx-1, nx-1, :] = bn
			templateT[nx-1, nx-1, :, :] = tn
		else:
			# Interior
			node_l = node_list[i-1]
			node_r = node_list[i+1]
			(al, ai, ar), bi, ti = \
				node_c.get_interior_equation(node_l, node_r)
			templateA[i, i, :] = ai
			templateA[i, i-1, :] = al
			templateA[i, i+1, :] = ar
			templateB[i, i, :] = bi
			templateT[i, i, :, :] = ti
	# Now construct the real matrices
	if ngroups == 1:
		# No reconstruction necessary
		matrixA = templateA.squeeze()
		matrixB = templateB.squeeze()
	else:
		matrixA = np.zeros((size, size))
		matrixB = np.zeros((size, size))
		for g in range(ngroups):
			i0 = g*nx
			i1 = (g+1)*nx
			# Destruction matrix
			matrixA[i0:i1, i0:i1] = templateA[:, :, g]
			# inscatter
			for gp in range(ngroups):
				if g != gp:
					igp0 = gp*nx
					igp1 = (gp+1)*nx
					matrixA[i0:i1, igp0:igp1] = -templateT[:, :, g, gp]
			# source matrix
			# Chi is forced to 1 --> all fission in the 'fast' group
			matrixB[0:nx, i0:i1] = templateB[:, :, g]
	# Clean up to free memory
	del templateA, templateB, templateT
	return matrixA, matrixB


def gauss_seidel(L, U, S, x, k):
	"""A Gauss-Seidel iterative solver
	
	Theoretically, this is a more efficient solution
	because we only do math on the matrix elements that
	we know contain the right solution, and x is updated as we go.
	
	In practice, this is slower than scipy'S builtin matrix solution methods
	(including inversion) because they use C or Fortran libraries.

	Parameters:
	-----------
	:type L: np.ndarray(ndim=2)
	:param L:
		Lower-triangular square matrix, containing the diagonal
	
	:type U: np.ndarray(ndim=2)
	:param U:
		Upper-triangular square matrix, only off-diagnal terms
	
	:type S: np.ndarray(ndim=1)
	:param S:
		Source vector
	
	:type x: np.ndarray;
	:param x:
		Flux vector of last iteration
	
	:type k: float
	:param k:
		Eigenvalue from the last iteration

	Returns:
	--------
	:rtype: np.ndarray
	:returns:
		Flux solution vector
	"""
	n = len(x) - 1
	m = len(x)//2
	
	# Leftmost
	x[0] = (S[0]/k - U[0, 1]*x[1])/L[0, 0]
	# Interior
	for i in range(1, m):
		x[i] = (S[i]/k - L[i, i - 1]*x[i - 1] - U[i, i + 1]*x[i + 1])/L[i, i]
	for i in range(m, n):
		x[i] = (S[i]/k - L[i, i - 1]*x[i - 1] - U[i, i + 1]*x[i + 1] -
		        L[i,i-m]*x[i-m]) / L[i,i]
	# Rightmost
	x[n] = (S[n]/k - L[n, n - 1]*x[n - 1] - L[n, n - m]*x[n - m])/L[n, n]
	return x


def l2norm_1d(new, old):
	"""Compare the L2 engineering norms of two 1-dimensional arrays

	Parameters:
	-----------
	:type new: np.ndarray
	:param new:
		Vector of the latest values
	
	:type old: np.ndarray
	:param old:
		Vector of the reference values.
		Must be the same shape as `new'.

	Returns:
	--------
	:rtype: float
	:returns:
		The L2 norm of the arrays
	"""
	diff = 0
	nx = len(new)
	for i, n in enumerate(new):
		if n:
			diff += ((n - old[i])/n)**2
	norm = np.sqrt(diff/nx)
	return norm


def get_off_diagonal(matrix):
	"""Return only the off-diagonal terms of a square matrix.
	
	Parameter:
	----------
	:type matrix: np.ndarray(ndim=2)
	:param matrix:
		Square matrix
	
	Returns:
	--------
	:rtype: np.ndarray(ndim=2)
	:returns:
		Copy of `matrix' with the diagonal terms zeroed out.
	"""
	off_diag = np.array(matrix, dtype=matrix.dtype)
	off_diag[np.diag_indices_from(matrix)] = 0
	return off_diag

