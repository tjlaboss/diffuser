import scipy

def get_off_diagonal(matrix):
	off_diag = scipy.array(matrix, dtype=matrix.dtype)
	off_diag[scipy.diag_indices_from(matrix)] = 0
	return off_diag


