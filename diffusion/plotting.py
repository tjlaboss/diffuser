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
	#ax1.set_xticks([0] + list(xvals))
	for i, node in enumerate(node_list):
		xvals[i] -= node.dx / 2.0
	xmax = xvals[-1]
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
	ax1.set_xlabel("$x$ (cm)", fontsize=11)
	ax1.set_ylabel("$\phi(x)$", fontsize=11)
	# Power distribution plot
	ax2 = ax1.twinx()
	if peaking is None:
		peaking = nanmax(source_vector)
	lf = ax2.plot(xvals, source_vector, "-", color="gold", label="Fission Source")
	lines += lf
	labels = [l.get_label() for l in lines]
	ax1.legend(lines, labels)
	ax1.grid()
	if peaking - 1 > 1E-5:
		imax = np.nanargmax(source_vector[::-1])  # Get the last peak.
		xplot = xvals[imax]
		hwidth = xmax / 10.0
		ax2.plot([xplot - hwidth, xplot + hwidth], [peaking, peaking],
		         '-', color="gray")
		ptext = "peaking = {:5.4} @ {} cm".format(peaking, int(xplot))
		ax2.text(xplot, peaking, ptext, ha="center", va="bottom")
	if keff:
		ktext = "$k_{eff} = " + "{:7.5}".format(keff) + "$"
		ax2.text(xmax*.01, peaking, ktext, ha="left")
	ax2.set_xlim(xvals[0], xmax)
	ax2.set_ylim(0, 1.1*peaking)
	ax2.set_ylabel("Relative fission source")
	return fig, (ax1, ax2)


def spy_plots(matA, matB):
	fig = figure()
	ax1 = fig.add_subplot(121)
	ax1.spy(matA)
	ax1.set_title("[A]\n", fontsize=12)
	ax2 = fig.add_subplot(122)
	ax2.spy(matB)
	ax2.set_title("[B]\n", fontsize=12)
	fig.tight_layout()
	return fig, (ax1, ax2)


def l2norm_plot(l2norm_vector):
	fig = figure()
	ax = fig.add_subplot(111)
	ax.semilogy(l2norm_vector)
	ax.set_xlabel("Outer Iteration")
	ax.set_ylabel("$L_2$ Engineering Norm")
	ax.grid()
	ax.set_xlim(0, len(l2norm_vector))
	return fig, (ax,)
