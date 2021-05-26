import plotting_helpers

import matplotlib
import numpy
import scipy.spatial

CM_PER_VOXEL = 0.4

POINT_MULTIPLICITY_BINS = list(range(10)) + list(range(10, 20, 2)) + [20, 25, 30]
POINT_DIST_BINS = numpy.concatenate((numpy.arange(0, 4, 0.4),
									numpy.arange(4, 10, 0.5),
									numpy.arange(10, 30, 2))) # cm

IGNORE = ["LEScatter",]
POINT_LABELS = [plotting_helpers.SHAPE_LABELS[idx] for idx in sorted(plotting_helpers.SHAPE_LABELS) if plotting_helpers.SHAPE_LABELS[idx] not in IGNORE]
POINT_TYPES = sorted(idx for idx in plotting_helpers.SHAPE_LABELS if plotting_helpers.SHAPE_LABELS[idx] not in IGNORE)


def ppn_reco_true_dists(vals):
	""" get the reco-to-true PPN distance matrix.  row index corresponds to reco points; column to true. """
	if "ppn_reco_true_dists" not in vals:
		dists = scipy.spatial.distance.cdist(vals["ppn_post"][:, :3], vals["particles_label"][:, :3])
#		print("raw dists:", dists)

		# only keep true-reco point pairs of same semantic type
		reco_mesh, true_mesh = numpy.meshgrid(vals["ppn_post"][:, -1], vals['particles_label'][:, 4])
		match_type_mask = (reco_mesh == true_mesh).T
		dists[~match_type_mask] = numpy.nan

#		print("dists after masking:", dists)

		vals["ppn_reco_true_dists"] = dists

	return vals["ppn_reco_true_dists"]


@plotting_helpers.hist_aggregate("n-ppn-reco", bins=POINT_MULTIPLICITY_BINS)
def agg_nppn_reco(vals):
	ppn_types = vals["ppn_post"][:, -1]
	return {"pointtype=%d" % ppn_type: [numpy.count_nonzero(ppn_types == ppn_type),] for ppn_type in POINT_TYPES}


@plotting_helpers.hist_aggregate("n-points-true", bins=POINT_MULTIPLICITY_BINS)
def agg_npoints_true(vals):
	point_types = vals['particles_label'][:, 4]
	return {"pointtype=%d" % part_type: [numpy.count_nonzero(point_types == part_type),] for part_type in POINT_TYPES}


@plotting_helpers.hist_aggregate("point-dist-reco-to-true", bins=POINT_DIST_BINS, norm="density")
def agg_points_dist_recototrue(vals):
	dists = ppn_reco_true_dists(vals)

	closest = numpy.nanmin(dists, axis=1) if len(dists) > 0 else [numpy.nan,]  # look across all true points for each reco point
	return closest


@plotting_helpers.hist_aggregate("point-dist-true-to-reco", bins=POINT_DIST_BINS, norm="density")
def agg_points_dist_truetoreco(vals):
	dists = ppn_reco_true_dists(vals)

	closest = numpy.nanmin(dists, axis=0) if len(dists) > 0 else [numpy.nan,]  # for each true point, look across all reco points
	return closest


#------------------------------------------------------


HIST_FUNCTIONS = [
	agg_nppn_reco,
	agg_npoints_true,
	agg_points_dist_recototrue,
	agg_points_dist_truetoreco,
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
		                                         yaxis_label="Events",
		                                         hist_labels={hname: POINT_LABELS[int(hname.split("=")[1])]
		                                                             + (" (reco)" if hname.startswith("n-ppn-reco") else " (true)")
		                                                      for hname in point_type_hists},
		                                         color={hname: colors[int(hname.split("=")[1])] for hname in point_type_hists},
		                                         linestyle={hname: ":" for hname in point_type_hists if hname.startswith("n-points-true")})
		plotting_helpers.savefig(fig, "npoints", outdir, fmts)

	point_dist_hists = {hname: hists[hname] for hname in hists if hname.startswith("point-dist") and hists[hname]}
	if len(point_dist_hists):
		print(POINT_DIST_BINS[-1] / CM_PER_VOXEL, "voxel bins to consider")
		for hname in point_dist_hists:
			print("For hist '%s':" % hname)
			for bound in range(len(hists[hname].data)):
				print("  ", sum(hists[hname].data[:bound])/sum(hists[hname].data),
					  "of distribution lies within %f voxels" % (hists[hname].bins[bound] / CM_PER_VOXEL))

			hists[hname].Normalize()

		fig, ax = plotting_helpers.overlay_hists(point_dist_hists,
												 xaxis_label="Distance (cm)",
												 yaxis_label="Points / cm",
												 hist_labels={"point-dist-reco-to-true": "... to closest true point",
															  "point-dist-true-to-reco": "... to closest reco point"})
		ax.set_yscale("log")
		plotting_helpers.savefig(fig, "point-dist", outdir, fmts)
