# 22.05 Problems
#
# 22.05/PSet07: Numerical Diffusion Theory Analysis

import diffusion
import materials


problems = [None]*10
# Problem 0: Infinite homogeneous medium
problems[0] = diffusion.Problem1D(2, fuel=materials.fuel_016)
problems[0].add_east_bc("r")
problems[0].add_west_bc("r")
problems[0].add_region(materials.fuel_016, 300, 60)
# Problem 1: Finite homogeneous medium
problems[1] = diffusion.Problem1D(2, fuel=materials.fuel_016)
problems[1].add_east_bc("v")
problems[1].add_west_bc("v")
problems[1].add_region(materials.fuel_016, 300, 60)
# Problem 2: Variable mesh spacing (optional for students)
problems[2] = diffusion.Problem1D(2, fuel=materials.fuel_024)
problems[2].add_east_bc("v")
problems[2].add_west_bc("v")
problems[2].add_region(materials.fuel_024, 25, 10)
problems[2].add_region(materials.fuel_024, 250, 50)
problems[2].add_region(materials.fuel_024, 25, 10)
# Problem 3: Homogenized reflector, coarse mesh
problems[3] = diffusion.Problem1D(2, fuel=materials.fuel_024)
problems[3].add_east_bc("v")
problems[3].add_west_bc("v")
problems[3].add_region(materials.baffle_refl, 25, 5)
problems[3].add_region(materials.fuel_024, 250, 50)
problems[3].add_region(materials.baffle_refl, 25, 5)
# Problem 4: Homogenized reflector, fine mesh
problems[4] = diffusion.Problem1D(2, fuel=materials.fuel_024)
problems[4].add_east_bc("v")
problems[4].add_west_bc("v")
problems[4].add_region(materials.baffle_refl, 25, 25)
problems[4].add_region(materials.fuel_024, 250, 250)
problems[4].add_region(materials.baffle_refl, 25, 25)
# Problem 5: Higher enriched zones on the edges, burnable poisons in middle
problems[5] = diffusion.Problem1D(2, fuel=materials.fuel_bp)
problems[5].add_east_bc("v")
problems[5].add_west_bc("v")
problems[5].add_region(materials.baffle_refl, 25, 25)
problems[5].add_region(materials.fuel_031, 16, 16)
problems[5].add_region(materials.fuel_bp, 218, 218)
problems[5].add_region(materials.fuel_031, 16, 16)
problems[5].add_region(materials.baffle_refl, 25, 25)
# Problem 6: Explicit baffle
problems[6] = diffusion.Problem1D(2, fuel=materials.fuel_bp)
problems[6].add_east_bc("v")
problems[6].add_west_bc("v")
problems[6].add_region(materials.refl, 23, 23)
problems[6].add_region(materials.baffle, 2, 2)
problems[6].add_region(materials.fuel_031, 14, 14)
problems[6].add_region(materials.fuel_bp, 222, 218)
problems[6].add_region(materials.fuel_031, 14, 14)
problems[6].add_region(materials.baffle, 2, 2)
problems[6].add_region(materials.refl, 23, 23)
# Problem 7: Peripheral fuel sensitivity study
problems[7] = diffusion.Problem1D(2, fuel=materials.fuel_bp)
problems[7].add_east_bc("v")
problems[7].add_west_bc("v")
problems[7].add_region(materials.refl, 23, 23)
problems[7].add_region(materials.baffle, 2, 2)
problems[7].add_region(materials.fuel_031, 16, 16)
problems[7].add_region(materials.fuel_bp, 218, 218)
problems[7].add_region(materials.fuel_031, 16, 16)
problems[7].add_region(materials.baffle, 2, 2)
problems[7].add_region(materials.refl, 23, 23)
# Problem 8: Another peripheral fuel sensitivity study
problems[8] = diffusion.Problem1D(2, fuel=materials.fuel_bp)
problems[8].add_east_bc("v")
problems[8].add_west_bc("v")
problems[8].add_region(materials.refl, 23, 23)
problems[8].add_region(materials.baffle, 2, 2)
problems[8].add_region(materials.fuel_031, 18, 18)
problems[8].add_region(materials.fuel_bp, 214, 214)
problems[8].add_region(materials.fuel_031, 18, 18)
problems[8].add_region(materials.baffle, 2, 2)
problems[8].add_region(materials.refl, 23, 23)
# Problem 9: Infinite Reflector
problems[9] = diffusion.Problem1D(2, fuel=materials.fuel_bp)
problems[9].add_east_bc("r")
problems[9].add_west_bc("r")
problems[9].add_region(materials.refl, 23, 23)
problems[9].add_region(materials.baffle, 2, 2)
problems[9].add_region(materials.fuel_031, 18, 18)
problems[9].add_region(materials.fuel_bp, 214, 218)
problems[9].add_region(materials.fuel_031, 18, 18)
problems[9].add_region(materials.baffle, 2, 2)
problems[9].add_region(materials.refl, 23, 23)

# Run ALL the problems
for i, prob in enumerate(problems):
	if i == 1:
		# Show advanced plots for one of the problems
		prob.run(diffusion.ScipySolver, plot_level=3)
	else:
		prob.run(diffusion.ScipySolver, plot_level=1)
diffusion.plotting.show()
