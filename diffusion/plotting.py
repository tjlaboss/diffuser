# Plotting
#
# Common plots

from pylab import *


def flux_and_fission_plot(flux_vector, source_vector, node_list,
                          keff=None, peaking=None):
	fig = figure()
	ax1 = fig.add_subplot(111)
	nx = len(node_list)
	xvals = np.cumsum([node.dx for node in node_list])
	ngroups = len(flux_vector) // nx
	lines = []
	# Flux plots
	if ngroups == 1:
		l1 = ax1.plot(flux_vector, "g-", label="Flux")
		ax1.set_ylim(0, 1.1*max(flux_vector))
		lines += l1
	elif ngroups == 2:
		l1 = ax1.plot(xvals, flux_vector[:nx], "b-", label="Fast Flux")
		l2 = ax1.plot(xvals, flux_vector[nx:], "r-", label="Thermal Flux")
		ax1.set_ylim(0, max(flux_vector[:nx] + flux_vector[nx:]))
		lines += l1 + l2
	else:
		raise NotImplementedError("{} groups".format(ngroups))
	ax1.set_xticks([0] + list(xvals))
	ax1.set_xlabel("$x$ (cm)", fontsize=11)
	ax1.set_ylabel("$\phi(x)$", fontsize=11)
	# Power distribution plot
	ax2 = ax1.twinx()
	if peaking is None:
		peaking = nanmax(source_vector) / nanmean(source_vector)
	rel_power = array(source_vector)
	rel_power[rel_power == 0] = NaN
	rel_power /= nanmean(rel_power)
	peaking = rel_power.max()
	lf = ax2.plot(xvals, rel_power, "-", color="gold", label="Fission Source")
	lines += lf
	labels = [l.get_label() for l in lines]
	ax1.legend(lines, labels)
	ax1.grid()
	# TODO add a short line at the top and label to show the peaking
	print("Also include peaking:", peaking)
	# TODO add text to show keff
	if keff:
		print("Also include keff")
		ktext = "$k_{eff} = " + "{:.5}".format(keff) + "$"
		ax2.text(0, peaking, ktext)
	ax2.set_ylim(0, 1.1*peaking)
	ax2.set_ylabel("Relative fission source")
	return fig, (ax1, ax2)
