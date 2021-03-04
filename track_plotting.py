
from matplotlib import pyplot as plt
import numpy
import scipy.spatial

import plotting_helpers

TRACK_LABEL = 1

TRACK_THRESHOLD = 5   # voxels.  will be converted to geometry units below

def convert_pixel_to_geom(val, data):
	# assume cubical, for now...
	return val * data["metadata"][0].metadata.size_voxel_x()


def track_lengths(vals):
	lengths = numpy.full_like(vals["clust_fragments"], numpy.nan)
	for trk_index in vals["clust_frag_seg"][vals["clust_frag_seg"] == TRACK_LABEL]:
		# find the voxels within each track cluster that are furthest apart.  treat those as the ends
		voxels = vals["input_data"][vals["clust_fragments"][trk_index]][:, :3]
		lengths[trk_index] = numpy.max(scipy.spatial.distance.cdist(voxels, voxels))

	return lengths


@plotting_helpers.hist_aggregate("n-tracks-reco", bins=15, range=range(15))
def agg_ntracks_reco(vals, hists):
	track_type_mask = vals["clust_frag_seg"] == TRACK_LABEL
	length_mask = vals["clust_fragments"][track_lengths(vals) > convert_pixel_to_geom(TRACK_THRESHOLD, vals)]
	return [len(vals["clust_fragments"][track_type_mask & length_mask]),]


@plotting_helpers.hist_aggregate("n-tracks-true", bins=15, range=range(15))
def agg_ntracks_true(vals, hists):
	# here the track threshold is used in voxels because that's what the internal representation of the particles is
	return [len([p for p in vals["particles"]
	               if p.shape() == TRACK_LABEL
	                  and p.end_position().as_point3d().distance(p.position().as_point3d()) > TRACK_THRESHOLD]),]


#------------------------------------------------------

@plotting_helpers.req_vars_hist(["clust_fragments", "clust_frag_seg", "particles", "metadata"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["particles"])):
		# first: number of tracks
		agg_ntracks_reco(data, hists)
		agg_ntracks_true(data, hists)


		# compute overlaps of pixel-width trajectories from each true track and each reconstructed track fragment?
		# (should probably do this vs. track length; perhaps vs. true particle type, too)


def PlotHists(hists, outdir, fmts):

	# n-tracks plots are simple true-reco comparisons
	ntracks_hists = ("n-tracks-reco", "n-tracks-true")
	if all(h in hists for h in ntracks_hists):
		fig, ax = plt.subplots()
		for h in ntracks_hists:
			ax.bar(h.bins[:-1], h.data, width=numpy.diff(h.bins))

		plotting_helpers.savefig(fig, h, outdir, fmts)
