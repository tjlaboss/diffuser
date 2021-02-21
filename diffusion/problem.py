"""
Problem

Class for a diffusion problem that populates and solves itself
"""

import numpy as np
from . import node
from . import matrices
from . import plotting


def _get_boundary_condition(bc_type):
	"""Turn full boundary condition names into one boundary condition key
	
	Parameter:
	----------
	:type bc_type: str
	:param bc_type:
		Boundary condition type:
			Vacuum:     'v', 'vacuum', or 'zero-incoming'
			Reflective: 'r', 'reflective', or 'zero-current'
	
	Returns;
	--------
	:rtype: str
	:returns:
		Boundary condition 'v' or 'r'
	"""
	bc_type = bc_type.lower()
	if bc_type in ('v', 'vacuum', 'zero-incoming'):
		return 'v'
	elif bc_type in ('r', 'reflective', 'zero-current'):
		return 'r'
	else:
		errstr = "Unknown boundary condition: {}".format(bc_type)
		raise NotImplementedError(errstr)


class Problem1D:
	"""1D Neutron Diffusion Problem
	
	Parameters:
	-----------
	:type ngroups: int
	:param ngroups:
		Number of energy groups
	
	:type fuel: :class:`diffusion.material.Material`; optional
	:param fuel:
		Fuel material. Used for initial solution guess.
		[Default: None]
	
	"""
	def __init__(self, ngroups, fuel=None):
		self.fuel = fuel
		self._ngroups = ngroups
		self._xwidth = 0.0
		self._node_list = list()
		self._west_bc = None
		self._east_bc = None
		
	@property
	def ngroups(self):
		return self._ngroups
	
	@property
	def xwidth(self):
		return self._xwidth
	
	@property
	def num_nodes(self):
		return len(self._node_list)
	
	def _assert_ready(self):
		"""Determine if we're ready to solve"""
		errstr = str()
		if self._west_bc is None:
			errstr += "\nWestern boundary condition needs to be set."
		if self._east_bc is None:
			errstr += "\nEastern boundary condition needs to be set."
		if self.num_nodes < 3:
			errstr += "\nNot enough spatial nodes!"
		if self._xwidth <= 0:
			errstr += "\nTotal width can't be zero."
		if errstr:
			raise AssertionError(errstr)
		else:
			return True
	
	def add_west_bc(self, bc_type):
		"""Set the wetern (left) boundary condition
		
		Parameter:
		----------
		:type bc_type: str
		:param bc_type:
			Boundary condition type.
			'v' for vacuum or 'r' for reflective
		"""
		self._west_bc = _get_boundary_condition(bc_type)
	
	def add_east_bc(self, bc_type):
		"""Set the wetern (left) boundary condition
		
		Parameter:
		----------
		:type bc_type: str
		:param bc_type:
			Boundary condition type.
			'v' for vacuum or 'r' for reflective
		"""
		self._east_bc = _get_boundary_condition(bc_type)
	
	def add_region(self, fill, xwidth, num_nodes):
		"""Add a material region to the problem
		
		Adds a region of a single material and one or more nodes
		to the problem. Nodes are added from left to right.
		
		Parameters:
		-----------
		:type fill: :class:`diffusion.material.Material`;
		:param fill:
			Material with which to fill the new node(s).
		
		:type xwidth: float; cm
		:param xwidth:
			Width of the new region.
			
		:type num_nodes: int
		:param num_nodes:
			Number of nodes to split the new region into.
		"""
		assert xwidth > 0, "Width must be positive."
		self._xwidth += xwidth
		dx = xwidth / num_nodes
		for i in range(num_nodes):
			n = node.Node1D(self.ngroups, fill, dx)
			self._node_list.append(n)
	
	def run(self, SolverClass, kguess=None, plot_level=0):
		"""Run the numerical simulation
		
		Guide to plot levels:
			Level 0:  (default) no plots
			Level 1:  Flux and fission source plots only
			Level 2:  Level 1 + matrix spy() plots
			Level 3:  Level 2 + L2norm convergence plot
		
		Parameters:
		-----------
		:type SolverClass: derived class of :class:`diffusion.solvers.BaseSolver`
		:param SolverClass:
			Which numerical solver to use for the matrix solution
		
		:type kguess: float; optional
		:param kguess:
			Guess to use for k_eff
			[Default: k_inf of self.fuel if available; otherwise 1]
		
		:type plot_level: int; optional
		:param plot_level:
			How much plotting to do
			[Default: 0]
		"""
		self._assert_ready()
		matA, matB = matrices.populate_matrices(
			self._node_list, self._ngroups, self._east_bc, self._west_bc)
		assert matB.sum() > 0, "No fission cross section found!"
		nx = self.num_nodes
		solver = SolverClass(matA, matB)
		if self.fuel:
			solver.guess_flux(self.fuel, nx, self._west_bc, self._east_bc)
			if not kguess:
				kguess = self.fuel.get_kinf()
		# Do the computation
		flux, source, keff = solver.solve_eigenvalue(kguess)
		print("\tk_eff   = {:7.5f}".format(keff))
		if plot_level >= 1:
			# Normalize power to core-average, nonzero fission source
			power = np.array(source)
			power.shape = (self._ngroups, nx)
			power = power.sum(axis=0)
			power /= np.array([n.dx for n in self._node_list])
			power[power == 0] = np.NaN
			power /= np.nanmean(power)
			peaking = np.nanmax(power)
			print("\tpeaking = {:6.4f}".format(peaking))
			# Normalize flux to core-average fast flux
			flux /= flux[:nx].mean()
			plotting.flux_and_fission_plot(
				flux, power, self._node_list, keff, peaking)
		if plot_level >= 2:
			plotting.spy_plots(matA, matB)
		if plot_level >= 3:
			plotting.l2norm_plot(solver.get_l2norm_results())
