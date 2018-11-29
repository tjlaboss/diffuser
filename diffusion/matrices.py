# Functions for dealing with the diffusion matrices

import scipy

def populate_matrices(node_list, ngroups, bc_left, bc_right):
	nx = len(node_list)
	size = nx*ngroups
	# Matrix containing the entries for A; will be reshaped later.
	templateA = scipy.zeros((nx, nx, ngroups))
	templateB = scipy.zeros((nx, nx, ngroups))
	templateT = scipy.zeros((nx, nx, ngroups, ngroups))
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
		matrixA = scipy.zeros((size, size))
		matrixB = scipy.zeros((size, size))
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
	L:	array; lower-triangular square matrix, containing the diagonal
	U:	array; upper-triangular square matrix, only off-diagnal terms
	S:	array; source vector
	x:	array; flux vector of last iteration
	k:  float; eigenvalue from the last iteration

	Returns:
	--------
	x:	array; solution vector
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
	"""Compare the L2 engineering norms of two 1-dimentional arrays

	Parameters:
	-----------
	new:        array of the latest values
	old:        array of the reference values.
	            Must be the same shape as `new'.

	Returns:
	--------
	norm:       float; the L2 norm of the arrays
	"""
	diff = 0
	nx = len(new)
	for i, n in enumerate(new):
		if n:
			diff += ((n - old[i])/n)**2
	norm = scipy.sqrt(diff/nx)
	return norm


def get_off_diagonal(matrix):
	"""Return only the off-diagnoal terms of a square matrix."""
	off_diag = scipy.array(matrix, dtype=matrix.dtype)
	off_diag[scipy.diag_indices_from(matrix)] = 0
	return off_diag

