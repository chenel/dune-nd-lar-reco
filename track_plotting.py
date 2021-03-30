
from matplotlib import pyplot as plt
import numpy
import scipy.spatial

import plotting_helpers

TRACK_LABEL = 1

TRACK_THRESHOLD = 5   # voxels.  will be converted to geometry units below


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


def truth_track_lengths_vox(vals):
	""" return the lengths of all true tracks in units of voxels """
	if "true_track_lengths" in vals:
		lengths = vals["true_track_lengths"]
	else:
		lengths = numpy.array([p.end_position().as_point3d().distance(p.position().as_point3d())
		                       for p in vals["particles"]
		                       if p.shape() == TRACK_LABEL])
		vals["true_track_lengths"] = lengths

	return lengths




@plotting_helpers.hist_aggregate("n-tracks-reco", bins=15, range=(0,15))
def agg_ntracks_reco(vals):
	lengths = reco_track_lengths_cm(vals)
	length_mask = lengths > convert_pixel_to_geom(TRACK_THRESHOLD, vals["metadata"])
	return [numpy.count_nonzero(length_mask),]


@plotting_helpers.hist_aggregate("n-tracks-true", bins=15, range=(0,15))
def agg_ntracks_true(vals):
	# here the track threshold is used in voxels because that's what the internal representation of the particles is
	lengths = truth_track_lengths_vox(vals)
	return numpy.count_nonzero(lengths > TRACK_THRESHOLD)


@plotting_helpers.hist_aggregate("trk-length-true", bins=numpy.logspace(-1, numpy.log10(3000), 50))
def agg_trklen_true(vals):
	return convert_pixel_to_geom(truth_track_lengths_vox(vals), vals["metadata"])


@plotting_helpers.hist_aggregate("trk-length-reco", bins=numpy.logspace(-1, numpy.log10(3000), 50))
def agg_trklen_reco(vals):
	return reco_track_lengths_cm(vals)


#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "track_fragments", "track_group_pred", "particles", "metadata"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["particles"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_trklen_reco, agg_trklen_true, agg_ntracks_reco, agg_ntracks_true):
			agg_fn(evt_data, hists)


		# compute overlaps of pixel-width trajectories from each true track and each reconstructed track fragment?
		# (should probably do this vs. track length; perhaps vs. true particle type, too)


def PlotHists(hists, outdir, fmts):
	# n-tracks plots are simple true-reco comparisons
	ntracks_hists = {hname: hists[hname] for hname in ("n-tracks-reco", "n-tracks-true")}
	if all(ntracks_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ntracks_hists,
		                                         xaxis_label="'Track' cluster multiplicity (length > %d voxels)" % TRACK_THRESHOLD,
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
