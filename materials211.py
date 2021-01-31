# Materials
#
# Materials definitions for the 22.211 problem set

import numpy as np
from diffusion.material import Material

data = np.array([
	[1.43, 0.37, 0.0079, 0.0605, 0.0195, 0.0034, 0.0711],
	[1.43, 0.37, 0.0084, 0.0741, 0.0185, 0.0054, 0.1000],
	[1.43, 0.37, 0.0089, 0.0862, 0.0178, 0.0054, 0.1000],
	[1.43, 0.37, 0.0088, 0.0852, 0.0188, 0.0062, 0.1249],
	[1.26, 0.27, 0.0025, 0.0200, 0.0294, 0,      0],
	[1.00, 0.34, 0.0054, 0.1500, 0.0009, 0,      0],
	[1.55, 0.27, 0.0010, 0.0300, 0.0500, 0,      0]
])

fuel_016 = Material(2, name="Fuel - 1.6% enriched")
fuel_024 = Material(2, name="Fuel - 2.4% enriched")
fuel_bp  = Material(2, name="Fuel - 2.4% with BP")
fuel_031 = Material(2, name="Fuel - 3.1% enriched")
baffle_refl = Material(2, name="Baffle/Reflector")
baffle = Material(2, name="Baffle")
refl = Material(2, name="Reflector")

all_materials = (
	fuel_016, fuel_024, fuel_bp, fuel_031, baffle_refl, baffle, refl)

for _k, _mat in enumerate(all_materials):
	# Diffusion Coefficient
	_mat.d = data[_k, 0:2]
	# Absorption
	_mat.sigma_a = data[_k, 2:4]
	# Scattering
	s12 = data[_k, 4]
	_mat.scatter_matrix = np.array([
		[0,   0],
		[s12, 0]])
	# Nu-Fission
	_mat.nu_sigma_f = data[_k, 5:7]

if __name__ == "__main__":
	for m in all_materials:
		print("{}: \tk_inf = {:.5f}".format(m.name, m.get_kinf()))

