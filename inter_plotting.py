import copy

from matplotlib import pyplot as plt
import matplotlib.colors
import numpy
import scipy.spatial

import plotting_helpers

VOXEL_THRESHOLD_MEV = 0.25

def true_reco_inter_match(vals):
	pass

def reco_inter_voxel_ids(vals, inter_idx):
	# each entry in "inter_group_pred" is the interaction group id
	# for the cluster group object in "inter_particles" at the same index,
	# so we have to combine the voxels in all those cluster groups
	if "reco_inter_voxel_ids" not in vals:
		vals["reco_inter_voxel_ids"] =  numpy.concatenate(vals["input_data"][vals["inter_particles"][numpy.where(vals["inter_group_pred"] == inter_idx)]])

	return vals["reco_inter_voxel_ids"]


#------------------------------------------------------

@plotting_helpers.hist_aggregate("n-inter-reco", bins=50, range=(0,50))
def agg_ninter_reco(vals):
	print(vals["inter_group_pred"])
	return [len(numpy.unique(vals["inter_group_pred"])),]


@plotting_helpers.hist_aggregate("n-inter-true", bins=50, range=(0,50))
def agg_ninter_true(vals):
	return [len(numpy.unique(vals["cluster_label"][:, 7])),]


@plotting_helpers.hist_aggregate("n-vox-inter-reco", bins=50, range=(0,50))
def agg_nvox_inter_reco(vals):

	inter_nhit = []
	for inter in numpy.unique(vals["inter_group_pred"]):
		# each entry in "inter_group_pred" is the interaction group id
		# for the cluster group object in "inter_particles" at the same index,
		# so we have to add up the voxels in all those cluster groups for each interaction
		inter_vox_ids = reco_inter_voxel_ids(vals, inter)
		inter_nhit.append(numpy.count_nonzero(vals["input_data"][inter_vox_ids] > VOXEL_THRESHOLD_MEV))
	return inter_nhit


@plotting_helpers.hist_aggregate("n-vox-inter-true", bins=50, range=(0,50))
def agg_nvox_inter_true(vals):

	return [numpy.count_nonzero(vals["cluster_label"][, :7] == inter_lbl & vals["input_data"] > VOXEL_THRESHOLD_MEV)
	        for inter_lbl in numpy.unique(vals["cluster_label"][, :7])]


#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "inter_group_pred", "cluster_label",
                                 "metadata", "event_base"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["cluster_label"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_ninter_reco, agg_ninter_true, agg_nvox_inter_reco, agg_nvox_inter_true):
			agg_fn(evt_data, hists)


def PlotHists(hists, outdir, fmts):
	ninter_hists = {hname: hists[hname] for hname in ("n-inter-reco", "n-inter-true")}
	if all(ninter_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ninter_hists,
		                                         xaxis_label=r"$\nu$ interaction multiplicity",
		                                         hist_labels={"n-inter-reco": "Reco", "n-inter-true": "True"})

		plotting_helpers.savefig(fig, "n-inter", outdir, fmts)

	nvox_inter_hists = {hname: hists[hname] for hname in ("n-vox-inter-reco", "n-vox-inter-true")}
	if all(nvox_inter_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(nvox_inter_hists,
		                                         xaxis_label=r"Number of voxels in interactions",
		                                         hist_labels={"n-vox-inter-reco": "Reco", "n-vox-inter-true": "True"})

		plotting_helpers.savefig(fig, "n-vox-inter", outdir, fmts)


