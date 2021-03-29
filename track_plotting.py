
from matplotlib import pyplot as plt
import numpy
import scipy.spatial

import plotting_helpers

TRACK_LABEL = 1

TRACK_THRESHOLD = 5   # voxels.  will be converted to geometry units below

def convert_pixel_to_geom(val, data):
	# assume cubical, for now...
	return val * data["metadata"].size_voxel_x()


def track_lengths(vals):
	track_indices = numpy.unique(vals["track_group_pred"])
	lengths = numpy.full((len(track_indices)), numpy.nan)
	for i, trk_index in enumerate(track_indices):
		voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(vals["track_fragments"]) if vals["track_group_pred"][idx] == trk_index])
		voxels = vals["input_data"][voxel_indices][:, :3]
		lengths[i] = numpy.max(scipy.spatial.distance.cdist(voxels, voxels))

	return lengths


@plotting_helpers.hist_aggregate("n-tracks-reco", bins=15, range=(0,15))
def agg_ntracks_reco(vals):
	lengths = track_lengths(vals)
	length_mask = lengths > convert_pixel_to_geom(TRACK_THRESHOLD, vals)
	return [numpy.count_nonzero(length_mask),]


@plotting_helpers.hist_aggregate("n-tracks-true", bins=15, range=(0,15))
def agg_ntracks_true(vals):
	# here the track threshold is used in voxels because that's what the internal representation of the particles is
	return [len([p for p in vals["particles"]
	             if p.shape() == TRACK_LABEL
	             and p.end_position().as_point3d().distance(p.position().as_point3d()) > TRACK_THRESHOLD]),]


@plotting_helpers.hist_aggregate("trk-length-true", bins=numpy.logspace(0, numpy.log10(300), 50))
def agg_trklen_true(vals):
	return [p.end_position().as_point3d().distance(p.position().as_point3d()) * 0.1  # convert to cm
	        for p in vals["particles"]
	        if p.shape() == TRACK_LABEL]


@plotting_helpers.hist_aggregate("trk-length-reco", bins=numpy.logspace(0, numpy.log10(300), 50))
def agg_trklen_reco(vals):
	return track_lengths(vals)


#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "track_fragments", "track_group_pred", "particles", "metadata"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["particles"])):
		print("Evaluating event #", evt_idx)
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
		                                         xaxis_label="'Track' cluster multiplicity",
		                                         hist_labels={"n-tracks-reco": "Reco", "n-tracks-true": "True"})

		plotting_helpers.savefig(fig, "n-tracks", outdir, fmts)

	trklen_hists = {hname: hists[hname] for hname in ("trk-length-reco", "trk-length-true")}
	if all(trklen_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(trklen_hists,
		                                         xaxis_label="'Track' length (cm)",
		                                         hist_labels={"trk-length-reco": "Reco", "trk-length-true": "True"})
		plotting_helpers.savefig(fig, "trk-len", outdir, fmts)
