
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy
import scipy.spatial

import plotting_helpers

TRACK_LABEL = 1

TRACK_THRESHOLD = 5.0 # cm


def convert_pixel_to_geom(val, metadata):
	# assume cubical, for now...
	return val * metadata.size_voxel_x()  # note these will be in cm


def reco_track_lengths_cm(vals):
	""" return the lengths of all reco track clusters in units of cm """
	if "reco_track_lengths" in vals:
		lengths = vals["reco_track_lengths"]
	else:
		track_indices = numpy.unique(vals["track_group_pred"])
		lengths = numpy.full((len(track_indices)), numpy.nan)
		for i, trk_index in enumerate(track_indices):
			voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(vals["track_fragments"]) if vals["track_group_pred"][idx] == trk_index])
			voxels = vals["input_data"][voxel_indices][:, :3]
			lengths[i] = numpy.max(scipy.spatial.distance.cdist(voxels, voxels))
		vals["reco_track_lengths"] = lengths   # store so if called again for this event we won't have to recalculate

	return lengths


def truth_track_lengths_cm(vals):
	""" return the lengths of all true particles with 'track' semantic type in units of cm """
	if "true_track_lengths" in vals:
		lengths = vals["true_track_lengths"]
	else:
		lengths = numpy.array([p.first_step().as_point3d().distance(p.last_step().as_point3d())
		                       for p in vals["particles"]
		                       if p.shape() == TRACK_LABEL])
		vals["true_track_lengths"] = lengths

	return lengths


def matched_track_indices(vals, proj_overlap_frac_cut=0.95):
	""" true - reco track length """
	# idea: for each (true, reco) pair,
    #       compute cos(opening angle between them),
    #       multiply it by the ratio of their lengths,
    #       and normalize it by a Gaussian function
	#       of the larger of the distances between each pair of endpoints.
	# this value, which we'll call projective overlap fraction,
	# will result in 1 if they are precisely collinear,
	# and smaller numbers as they differ in angle or length or are far apart.
	# return a list of (true, reco) index pairs for which
	# the projective overlap fraction is > than a cut value.

	truth_start = []
	truth_end = []
	for p in vals["particles"]:
		if p.shape() != TRACK_LABEL:
			continue

		truth_start.append(p.position().as_point3d())
		truth_end.append(p.end_position().as_point3d())
	truth_vec = truth_end - truth_start
	truth_norm = numpy.sqrt(truth_vec.dot(truth_vec))

	reco_start = []
	reco_end = []
	for i, trk_index in enumerate(track_indices):
		voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(vals["track_fragments"]) if vals["track_group_pred"][idx] == trk_index])
		voxels = vals["input_data"][voxel_indices][:, :3]
		start, stop = numpy.argmax(scipy.spatial.distance.cdist(voxels, voxels))
		reco_start.append(start)
		reco_end.append(stop)
	reco_vec = reco_end - reco_start
	reco_norm = numpy.sqrt(reco_vec.dot(reco_vec))

	# dot them together
	dot = (truth_vec * reco_vec)

	return []


#------------------------------------------------------


@plotting_helpers.hist_aggregate("n-tracks-reco", bins=15, range=(0,15))
def agg_ntracks_reco(vals):
	lengths = reco_track_lengths_cm(vals)
	length_mask = lengths > TRACK_THRESHOLD
	return [numpy.count_nonzero(length_mask),]


@plotting_helpers.hist_aggregate("n-tracks-true", bins=15, range=(0,15))
def agg_ntracks_true(vals):
	# here the track threshold is used in voxels because that's what the internal representation of the particles is
	lengths = truth_track_lengths_cm(vals)
	return numpy.count_nonzero(lengths > TRACK_THRESHOLD)


@plotting_helpers.hist_aggregate("trk-length-true", bins=numpy.logspace(-1, numpy.log10(3000), 50))
def agg_trklen_true(vals):
	return truth_track_lengths_cm(vals)


@plotting_helpers.hist_aggregate("trk-length-reco", bins=numpy.logspace(-1, numpy.log10(3000), 50))
def agg_trklen_reco(vals):
	return reco_track_lengths_cm(vals)


@plotting_helpers.hist_aggregate("delta-longest-trk-vs-length",
                                 hist_dim=2,
                                 bins=(numpy.logspace(0, numpy.log10(3000), 50),
                                       numpy.linspace(-1, 1, 50)))
def agg_dtrklen_vs_trklen(vals):
	truth_lengths = truth_track_lengths_cm(vals)
	longest_true = numpy.max(truth_lengths) if len(truth_lengths) > 0 else None
	reco_lengths = reco_track_lengths_cm(vals)
	longest_reco = numpy.max(reco_lengths) if len(reco_lengths) > 0 else None
	#
	# print(longest_true)

	if not longest_true or not longest_reco:
		return []

	return [[longest_true,], [(longest_true - longest_reco) / longest_true,]]



#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "track_fragments", "track_group_pred", "particles", "metadata"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["particles"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_trklen_reco, agg_trklen_true, agg_ntracks_reco, agg_ntracks_true,
		               agg_dtrklen_vs_trklen):
			agg_fn(evt_data, hists)


		# compute overlaps of pixel-width trajectories from each true track and each reconstructed track fragment?
		# (should probably do this vs. track length; perhaps vs. true particle type, too)


def PlotHists(hists, outdir, fmts):
	# n-tracks plots are simple true-reco comparisons
	ntracks_hists = {hname: hists[hname] for hname in ("n-tracks-reco", "n-tracks-true")}
	if all(ntracks_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ntracks_hists,
		                                         xaxis_label="'Track' cluster multiplicity (length > %.1f cm)" % TRACK_THRESHOLD,
		                                         hist_labels={"n-tracks-reco": "Reco", "n-tracks-true": "True"})

		plotting_helpers.savefig(fig, "n-tracks", outdir, fmts)

	trklen_hists = {hname: hists[hname] for hname in ("trk-length-reco", "trk-length-true")}
	if all(trklen_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(trklen_hists,
		                                         xaxis_label="'Track' length (cm)",
		                                         yaxis_label="Tracks",
		                                         hist_labels={"trk-length-reco": "Reco", "trk-length-true": "True"})
		ax.set_xscale("log")
		plotting_helpers.savefig(fig, "trk-len", outdir, fmts)

	if hists["delta-longest-trk-vs-length"]:
		h = hists["delta-longest-trk-vs-length"]
		fig = plt.figure()
		ax = fig.add_subplot()
		x, y = numpy.meshgrid(*h.bins)
		im = ax.pcolormesh(x, y, h.data.T, cmap="Reds", norm=matplotlib.colors.LogNorm())
		plt.colorbar(im)

		ax.set_xlabel("$L_{true}$ (cm)")
		ax.set_ylabel("($L_{true} - L_{reco}) / L_{true}$ for longest true, reco tracks")
		ax.set_xscale("log")

		# line at y=0
		ax.axhline(0, color='black', linestyle=':')

		plotting_helpers.savefig(fig, "dlongesttrklen-vs-true", outdir, fmts)
