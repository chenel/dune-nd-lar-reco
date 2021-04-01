import plotting_helpers

import matplotlib
import numpy


POINT_MULTIPLICITY_BINS = list(range(10)) + list(range(10, 20, 2)) + [20, 25, 30]

IGNORE = ["LEScatter",]
POINT_LABELS = [plotting_helpers.SHAPE_LABELS[idx] for idx in sorted(plotting_helpers.SHAPE_LABELS) if plotting_helpers.SHAPE_LABELS[idx] not in IGNORE]
POINT_TYPES = sorted(idx for idx in plotting_helpers.SHAPE_LABELS if plotting_helpers.SHAPE_LABELS[idx] not in IGNORE)


@plotting_helpers.hist_aggregate("n-ppn-reco", bins=POINT_MULTIPLICITY_BINS)
def agg_nppn_reco(vals):
	ppn_types = vals["ppn_post"][:, -1]
	return {"pointtype=%d" % ppn_type: [numpy.count_nonzero(ppn_types == ppn_type),] for ppn_type in POINT_TYPES}


@plotting_helpers.hist_aggregate("n-points-true", bins=POINT_MULTIPLICITY_BINS)
def agg_npoints_true(vals):
	point_types = vals['particles_label'][:, 4]
	return {"pointtype=%d" % part_type: [numpy.count_nonzero(point_types == part_type),] for part_type in POINT_TYPES}


#------------------------------------------------------


HIST_FUNCTIONS = [
	agg_nppn_reco,
	agg_npoints_true,
]
@plotting_helpers.req_vars_hist(["ppn_post", "particles_label"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["particles_label"])):
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in HIST_FUNCTIONS:
			agg_fn(evt_data, hists)


def PlotHists(hists, outdir, fmts):
	point_type_hists = {hname: hists[hname] for hname in hists if hname.startswith("n-ppn-reco") or hname.startswith("n-points-true")}
	if all(point_type_hists.values()):
		colors = [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]]
		fig, ax = plotting_helpers.overlay_hists(point_type_hists,
		                                         xaxis_label="POI multiplicity",
		                                         yaxis_label="Events / unit",
		                                         hist_labels={hname: POINT_LABELS[int(hname.split("=")[1])]
		                                                             + (" (reco)" if hname.startswith("n-ppn-reco") else " (true)")
		                                                      for hname in point_type_hists},
		                                         color={hname: colors[int(hname.split("=")[1])] for hname in point_type_hists},
		                                         linestyle={hname: ":" for hname in point_type_hists if hname.startswith("n-points-true")})
		plotting_helpers.savefig(fig, "npoints", outdir, fmts)

