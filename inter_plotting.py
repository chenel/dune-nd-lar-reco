import copy

from matplotlib import pyplot as plt
import matplotlib.colors
import numpy
import scipy.spatial

import plotting_helpers

VOXEL_THRESHOLD_MEV = 0.25


def find_matching_rows(a1, a2, mask_only=False):
	""" Find the indices of rows of 2D array a1 that match to rows of 2D array a2
	"""
	assert(a1.shape[1] == a2.shape[1])
	assert(a1.dtype == a2.dtype)

	# adapted from https://stackoverflow.com/questions/16210738/implementation-of-numpy-in1d-for-2d-arrays
	dtype = ",".join([str(a1.dtype),] * a1.shape[1])
	# this gives us a mask for the first array
	mask = numpy.in1d(a1.view(dtype=dtype).reshape(a1.shape[0]), a2.view(dtype=dtype))
	if mask_only:
		return mask
	else:
		return numpy.nonzero(mask)

	# solution below is quite slow for large arrays due to all the broadcasting
	# (adapted from adapted from https://stackoverflow.com/questions/64930665/find-indices-of-rows-of-numpy-2d-array-in-another-2d-array)
	# # 2D array where each row corresponds to a row from a2,
	# # and each column corresponds to a row from a1.
	# # matching rows should have had a row of shape[1]*True
	# # from the equality test; the all() combines them for easy testing
	# matches_by_a2row = (a1 == a2[:, None]).all(axis=2)
	#
	# # the argwhere returns an array of pairs [(a2 row index, a1 row index), (a2 row index, a1 row index), ...]
	# return numpy.argwhere(matches_by_a2row)[:, 1]

def true_inter_lbls(vals):
	if "true_inter_ids" not in vals:
		vals["true_inter_ids"] = numpy.unique(vals["cluster_label"][:, 7])

	return vals["true_inter_ids"]


def true_inter_voxel_ids(vals, inter_lbl):
	if "true_inter_voxel_ids" not in vals:
		vals["true_inter_voxel_ids"] = {}
	if inter_lbl not in vals["true_inter_voxel_ids"]:
		# unfortunately the cluster_label and input_data vox are not guaranteed to be in the same order
		voxel_pos = vals["cluster_label"][vals["cluster_label"][:, 7] == inter_lbl, :3]
		vals["true_inter_voxel_ids"][inter_lbl] = find_matching_rows(numpy.array(vals["input_data"][:, :3]), numpy.array(voxel_pos))

	return vals["true_inter_voxel_ids"][inter_lbl]

def true_reco_inter_match(vals):
	pass

def reco_inter_voxel_ids(vals, inter_lbl):
	# each entry in "inter_group_pred" is the interaction group id
	# for the cluster group object in "inter_particles" at the same index,
	# so we have to combine the voxels in all those cluster groups
	if "reco_inter_voxel_ids" not in vals:
		vals["reco_inter_voxel_ids"] = {}
	if inter_lbl not in vals["reco_inter_voxel_ids"]:
#		print("for interaction index", inter_idx, " there are the following particle voxel collections:")
		vox_collections = tuple(vals["inter_particles"][numpy.nonzero(vals["inter_group_pred"] == inter_lbl)])
#		print(type(vox_collections), vox_collections)
		concatenated = numpy.concatenate(vox_collections)
#		print("concatenated:", concatenated)
		vals["reco_inter_voxel_ids"][inter_lbl] = concatenated

	return vals["reco_inter_voxel_ids"][inter_lbl]


#------------------------------------------------------

@plotting_helpers.hist_aggregate("n-inter-reco", bins=50, range=(0,100))
def agg_ninter_reco(vals):
#	print(vals["inter_group_pred"])
	return [len(numpy.unique(vals["inter_group_pred"])),]


@plotting_helpers.hist_aggregate("n-inter-true", bins=50, range=(0,100))
def agg_ninter_true(vals):
	return [len(true_inter_lbls(vals)), ]


@plotting_helpers.hist_aggregate("n-vox-inter-reco", bins=60, range=(0,300))
def agg_nvox_inter_reco(vals):

	inter_nhit = []
	for inter in numpy.unique(vals["inter_group_pred"]):
		# each entry in "inter_group_pred" is the interaction group id
		# for the cluster group object in "inter_particles" at the same index,
		# so we have to add up the voxels in all those cluster groups for each interaction
		inter_vox_ids = reco_inter_voxel_ids(vals, inter)
		inter_nhit.append(numpy.count_nonzero(vals["input_data"][inter_vox_ids] > VOXEL_THRESHOLD_MEV))
	return inter_nhit


@plotting_helpers.hist_aggregate("n-vox-inter-true", bins=60, range=(0,300))
def agg_nvox_inter_true(vals):
	return [numpy.count_nonzero((vals["cluster_label"][:, 7] == inter_lbl) & (vals["cluster_label"][:, 4] > VOXEL_THRESHOLD_MEV))
	        for inter_lbl in true_inter_lbls(vals)]


@plotting_helpers.hist_aggregate("ungrouped-trueint-energy-frac", bins=26, range=(0,1.04))
def agg_ungrouped_trueint_energy_frac(vals):

	inter_unmatch_frac = []
	all_reco_vox = vals["input_data"][numpy.unique(numpy.concatenate(vals["inter_particles"]))]
	for inter_lbl in true_inter_lbls(vals):
		true_vox = vals["input_data"][true_inter_voxel_ids(vals, inter_lbl)]
		true_E_sum = true_vox[:, 4].sum()
		assert(true_E_sum > 0)

		matched_vox_indices = find_matching_rows(numpy.array(all_reco_vox[:, :3]), numpy.array(true_vox[:, :3]))
		matched_E_sum = all_reco_vox[matched_vox_indices][:, 4].sum()

		inter_unmatch_frac.append(1 - matched_E_sum / true_E_sum)

	return inter_unmatch_frac


@plotting_helpers.hist_aggregate("largest-trueint-energy-matched-frac", bins=26, range=(0,1.04))
def agg_trueint_largest_matched_energy_frac(vals):

	inter_match_frac = []
#	print("There are", len(vals["input_data"]), "total voxels in this spill")
	for idx, inter_lbl in enumerate(true_inter_lbls(vals)):
		# this is the true label for *all* externally-entering stuff.
		# because they're lumped together, we can't properly assess
		# whether each individual (e.g.) rock muon is correctly matched,
		# so we just skip it here.
		if inter_lbl < 0:
			continue

		true_vox = vals["input_data"][true_inter_voxel_ids(vals, inter_lbl)]
		true_E_sum = true_vox[:, 4].sum()
		assert(true_E_sum > 0)
#		print("interaction label", inter_lbl, "(index {})".format(idx), "has", len(true_vox), "voxels with a total energy of", true_E_sum)

		max_matched_E = 0
		true_vox_copy = numpy.array(true_vox[:, :3])  # need to copy to get the information contiguous
		for part_idx, particle_vox_indices in enumerate(vals["inter_particles"]):
			reco_vox = vals["input_data"][particle_vox_indices]
			matched_vox_mask = find_matching_rows(numpy.array(reco_vox[:, :3]), true_vox_copy, mask_only=True)
			matched_E = reco_vox[matched_vox_mask][:, 4].sum()
			# if matched_E > 0:
			# 	print("    particle", part_idx, "has", len(reco_vox), "voxels, of which")
			# 	print("        ", numpy.count_nonzero(matched_vox_mask), "match to the true interaction for total matched energy of", matched_E)

			max_matched_E = max(max_matched_E, matched_E)

		inter_match_frac.append(max_matched_E / true_E_sum)

	return inter_match_frac

#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "inter_group_pred", "inter_particles", "cluster_label",
                                 "metadata", "event_base"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["cluster_label"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_ninter_reco, agg_ninter_true, agg_nvox_inter_reco, agg_nvox_inter_true,
		               agg_ungrouped_trueint_energy_frac, agg_trueint_largest_matched_energy_frac):
			agg_fn(evt_data, hists)


def PlotHists(hists, outdir, fmts):
	ninter_hists = {hname: hists[hname] for hname in ("n-inter-reco", "n-inter-true")}
	if all(ninter_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ninter_hists,
		                                         xaxis_label=r"$\nu$ interaction multiplicity",
		                                         yaxis_label="Spills",
		                                         hist_labels={"n-inter-reco": "Reco", "n-inter-true": "True"})

		plotting_helpers.savefig(fig, "n-inter", outdir, fmts)

	nvox_inter_hists = {hname: hists[hname] for hname in ("n-vox-inter-reco", "n-vox-inter-true")}
	if all(nvox_inter_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(nvox_inter_hists,
		                                         xaxis_label=r"Number of voxels in interaction ($E_{{vox}} > {}$ MeV)".format(VOXEL_THRESHOLD_MEV),
		                                         yaxis_label=r"$\nu$ interactions",
		                                         hist_labels={"n-vox-inter-reco": "Reco", "n-vox-inter-true": "True"})

		plotting_helpers.savefig(fig, "n-vox-inter", outdir, fmts)

	hist_labels = {
		"ungrouped-trueint-energy-frac":       r"Frac. true vis. $E_{dep}$ unmatched to reco int.",
		"largest-trueint-energy-matched-frac": r"Max frac. true vis. $E_{dep}$ matched to reco int.",
	}
	for plot, xlabel in hist_labels.items():
		if plot in hists:
			fig, ax = plotting_helpers.overlay_hists({plot: hists[plot]},
			                                         xaxis_label=xlabel,
			                                         yaxis_label=r"True $\nu$ interactions")
			plotting_helpers.savefig(fig, plot, outdir, fmts)
