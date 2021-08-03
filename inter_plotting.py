import copy

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy
import scipy.spatial

import plotting_helpers

# a muon is considered "split" if
# more than this fraction of its energy
# is found in multiple reco interactions
TRUE_MUON_SPLIT_FRAC = 0.2

# the threshold for considering a reco interaction
# to contain a "significant" part of a muon
# when counting interactions that merge muons
TRUE_MUON_MERGE_FRAC = 0.2

VOXEL_THRESHOLD_MEV = 0.25


def find_matching_rows(a1, a2, mask_only=False):
	""" Find the indices of rows of 2D array a1 that match to rows of 2D array a2
	"""
	assert(a1.dtype == a2.dtype)

	# the 1D case is straightforward
	if len(a1.shape) == 1:
		return np.argwhere(np.isin(a1, a2))

	assert(a1.shape[1] == a2.shape[1])

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
	""" Get the list of true interaction labels.

		:return: 1D numpy array of interaction labels (ints)
	"""
	if "true_inter_ids" not in vals:
		vals["true_inter_ids"] = numpy.unique(vals["cluster_label"][:, 7])

	return vals["true_inter_ids"]


def true_inter_voxel_ids(vals, inter_lbl):
	"""  Get a list of the indices of voxels in "input_data" that match to the given true interaction label.

		:param inter_lbl   The true interaction label of interest
		:return: array of indices for voxels in vals["input_data"] """

	if "true_inter_voxel_ids" not in vals:
		vals["true_inter_voxel_ids"] = {}
	if inter_lbl not in vals["true_inter_voxel_ids"]:
		# unfortunately the cluster_label and input_data vox are not guaranteed to be in the same order
		voxel_pos = vals["cluster_label"][vals["cluster_label"][:, 7] == inter_lbl, :3]
		vals["true_inter_voxel_ids"][inter_lbl] = find_matching_rows(numpy.array(vals["input_data"][:, :3]), numpy.array(voxel_pos))
#	print("Number of voxels in true interaction", inter_lbl, ":", len(vals["true_inter_voxel_ids"][inter_lbl]))

	return vals["true_inter_voxel_ids"][inter_lbl]


def true_muon_reco_matches(vals):
	"""
	Build a table of matches from each true muon in the spill to the reco interactions
	that contain some of its energy (and store the fraction of the true muon energy's energy matched).

	:return: dict of dicts { true muon cluster id: { reco interaction label : fraction of true mu energy } }
	"""
	key = "true_muon_recoint_matches"

	if key not in vals:
		true_mu_recoint_match = {}
		for mu_cluster_id, vox_idxs in true_muon_voxidxs_by_cluster(vals).items():
			total_mu_visE = vals["input_data"][vox_idxs, 4].sum()
			assert total_mu_visE > 0
#			print("muon label", mu_cluster_id, " voxels:", vox_idxs)
			for reco_inter_label in numpy.unique(vals["inter_group_pred"]):
				reco_inter_vox = reco_inter_voxel_ids(vals, reco_inter_label)
#				print("   reco interaction", reco_inter_label, "voxels:", type(reco_inter_vox), reco_inter_vox)
				matched_vox_idxs = find_matching_rows(vox_idxs, reco_inter_vox)
				if len(matched_vox_idxs) > 0:
					if mu_cluster_id not in true_mu_recoint_match:
						true_mu_recoint_match[mu_cluster_id] = {}
					true_mu_recoint_match[mu_cluster_id][reco_inter_label] = vals["input_data"][vox_idxs[matched_vox_idxs], 4].sum() / total_mu_visE

		vals[key] = true_mu_recoint_match

#	print(vals[key])
	return vals[key]

def true_muon_voxidxs_by_cluster(vals):
	"""
	Build a map from all true muons in the spill to the indices of the voxels in the data.

	:return:  dict of arrays of indices in vals["input_data"] corresponding to each muon, keyed by true muon cluster label
	"""

	# vals["cluster_label"][:, 8] is PDG code
	key = "true_muon_vox_by_cluster"

	if key not in vals:
		# Both mu+ and mu- SHOULD get value "2" in column 9.
		# Unfortunately this info is currently broken for reasons not yet known.
		# (It only seems to be correctly filled for the first muon encountered in the spill.)
		# So we have to bootstrap from the full particle list. :(
		part_pdg_group = numpy.array([(p.pdg_code(), p.group_id()) for p in vals["particles_raw"]])
		true_muon_groups = numpy.unique(part_pdg_group[abs(part_pdg_group[:, 0]) == 13][:, 1])


		# column 5 is "cluster ID", which is the GEANT4 ID,
		# but unfortunately, our current overlay files have GEANT4 interactions
		# that were generated separately overlaid on each other.
		# column 6 is "group ID", which is usable.
		muon_vox = {}
		for mu_id in true_muon_groups:
			vox = find_matching_rows(numpy.array(vals["input_data"][:, :3]),
			                                     numpy.array(vals["cluster_label"][vals["cluster_label"][:, 6] == mu_id, :3]))[0]
			if len(vox) > 0:
				muon_vox[mu_id] = vox

		vals[key] = muon_vox

#	print("returning from true_muon_voxidxs_by_cluster():", vals[key])
	return vals[key]


def reco_inter_voxel_ids(vals, inter_lbl):
	"""
	Obtain a list of the voxels corresponding to a reconstructed interaction.

	:param inter_lbl:  reco interaction label of interest
	:return: array of indices in vals["input_data"] corresponding to the voxels for that interaction
	"""

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

@plotting_helpers.hist_aggregate("true-muon-reco-int-match-count", bins=10, range=(0, 10))
def agg_true_muon_reco_int_match_count(vals):
	"""
	Count how many reco interactions contain at least  TRUE_MUON_SPLIT_FRAC of each true muon.
	"""
	true_muon_recoint_matches = true_muon_reco_matches(vals)

	return [sum(1 if frac >= TRUE_MUON_SPLIT_FRAC else 0 for frac in mu_match_fracs.values())
	        for mu_match_fracs in true_muon_recoint_matches.values()]


@plotting_helpers.hist_aggregate("true-muon-grouped-frac", bins=27, range=(0, 1.08))
def agg_true_muon_grouped_frac(vals):
	"""
	Compute the fraction of the true muon's energy contained by the reco interaction
	that contains the most of its energy.
	"""
	true_muon_recoint_matches = true_muon_reco_matches(vals)

	return [sum(mu_match_fracs.values()) for mu_match_fracs in true_muon_recoint_matches.values()]


@plotting_helpers.hist_aggregate("reco-int-muon-match-count", bins=10, range=(0,10))
def agg_reco_int_muon_match_count(vals):
	"""
	For each reco interaction, count the number of true muons
	for which at least TRUE_MUON_MERGE_FRAC of their energy
	is contained in that interaction.
	"""
	true_muon_recoint_matches = true_muon_reco_matches(vals)
	reco_int_muon_count = {reco_int: 0 for reco_int in numpy.unique(vals["inter_group_pred"])}
	for reco_int_fracs in true_muon_recoint_matches.values():
		for reco_int, frac in reco_int_fracs.items():
			if frac >= TRUE_MUON_MERGE_FRAC:
				reco_int_muon_count[reco_int] += 1

	return list(reco_int_muon_count.values())


@plotting_helpers.hist_aggregate("n-inter-reco", bins=50, range=(0,100))
def agg_ninter_reco(vals):
#	print(vals["inter_group_pred"])
	return [len(numpy.unique(vals["inter_group_pred"])),]


@plotting_helpers.hist_aggregate("n-inter-true", bins=50, range=(0,100))
def agg_ninter_true(vals):
	return [len(true_inter_lbls(vals)), ]


@plotting_helpers.hist_aggregate("n-vox-inter-reco", bins=numpy.logspace(0, numpy.log10(1e6), 50), norm="unit")
def agg_nvox_inter_reco(vals):

	inter_nhit = []
	for inter in numpy.unique(vals["inter_group_pred"]):
		# each entry in "inter_group_pred" is the interaction group id
		# for the cluster group object in "inter_particles" at the same index,
		# so we have to add up the voxels in all those cluster groups for each interaction
		inter_vox_ids = reco_inter_voxel_ids(vals, inter)
		inter_nhit.append(numpy.count_nonzero(vals["input_data"][inter_vox_ids] > VOXEL_THRESHOLD_MEV))
	return inter_nhit


@plotting_helpers.hist_aggregate("n-vox-inter-true", bins=numpy.logspace(0, numpy.log10(1e6), 50), norm="unit")
def agg_nvox_inter_true(vals):
	labels = true_inter_lbls(vals)
	nvox = {inter_lbl : numpy.count_nonzero((vals["cluster_label"][:, 7] == inter_lbl) & (vals["cluster_label"][:, 4] > VOXEL_THRESHOLD_MEV))
	        for inter_lbl in labels}
#	print("hit counts by true interaction label:", nvox)
	return list(nvox.values())


@plotting_helpers.hist_aggregate("ungrouped-trueint-energy-frac-vs-trueEdep",
                                 hist_dim=2, bins=(numpy.linspace(0, 5, 25),
                                                   numpy.linspace(0, 1.08, 27)))
def agg_ungrouped_trueint_energy_frac_vs_trueEdep(vals):

	inter_unmatch_frac = [[], []]
	all_reco_vox = vals["input_data"][numpy.unique(numpy.concatenate(vals["inter_particles"]))]
	for inter_lbl in true_inter_lbls(vals):
		# the "-1" label corresponds to LEScatters (which won't otherwise be grouped into anything).
		# almost all of its energy is not matched, *correctly*
		if inter_lbl < 0:
			continue

		true_vox = vals["input_data"][true_inter_voxel_ids(vals, inter_lbl)]
		true_E_sum = true_vox[:, 4].sum()
		assert(true_E_sum > 0)

		matched_vox_indices = find_matching_rows(numpy.array(all_reco_vox[:, :3]), numpy.array(true_vox[:, :3]))
		matched_E_sum = all_reco_vox[matched_vox_indices][:, 4].sum()

		inter_unmatch_frac[0].append(true_E_sum * 1e-3)  # convert to GeV
		inter_unmatch_frac[1].append(1 - matched_E_sum / true_E_sum)

		if matched_E_sum / true_E_sum < 0.2 and true_E_sum > 2000:
			print("True interaction with high true E_dep (", true_E_sum, "MeV)\n"
			      " and very little matched to reco interaction (", matched_E_sum, "):",
			      vals["event_base"], inter_lbl)
#			print("Matched voxels:", all_reco_vox[matched_vox_indices][:, :3], sep='\n')

	return inter_unmatch_frac


@plotting_helpers.hist_aggregate("largest-trueint-energy-matched-frac", bins=27, range=(0,1.08))
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


@plotting_helpers.hist_aggregate("recoint-purity-frac", bins=27, range=(0,1.08))
def agg_recoint_purity(vals):

	reco_purity = []
	for inter in numpy.unique(vals["inter_group_pred"]):
		inter_vox_ids = reco_inter_voxel_ids(vals, inter)
		reco_vox = vals["input_data"][inter_vox_ids]

		matched_energy = {}
		for idx, true_inter_lbl in enumerate(true_inter_lbls(vals)):
			true_vox = vals["input_data"][true_inter_voxel_ids(vals, true_inter_lbl)]
			true_vox_copy = numpy.array(true_vox[:, :3])

			matched_vox_mask = find_matching_rows(numpy.array(reco_vox[:, :3]), true_vox_copy, mask_only=True)
			matched_E = reco_vox[matched_vox_mask][:, 4].sum()
			if matched_E > 0:
				matched_energy[true_inter_lbl] = matched_E

		max_match_true_int = max(matched_energy, key=matched_energy.get)

		# all true "low-energy scatters" are lumped together into true interaction -1,
		# so if they are the leading energy depositor to this reco interaction,
		# we can't work out the purity (impossible to say which *actual* true nu int they came from).
		# so we just skip any like that.
		if max_match_true_int >= 0:
			reco_purity.append(matched_energy[max_match_true_int] / reco_vox[:, 4].sum())

	return reco_purity

#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "inter_group_pred", "inter_particles", "cluster_label",
                                 "metadata", "event_base", "particles_raw"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["cluster_label"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_ninter_reco, agg_ninter_true, agg_nvox_inter_reco, agg_nvox_inter_true,
		               agg_ungrouped_trueint_energy_frac_vs_trueEdep, agg_trueint_largest_matched_energy_frac,
		               agg_recoint_purity, agg_true_muon_grouped_frac, agg_true_muon_reco_int_match_count,
		               agg_reco_int_muon_match_count
		               ):
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
		for hname in nvox_inter_hists:
			hists[hname].Normalize()
		fig, ax = plotting_helpers.overlay_hists(nvox_inter_hists,
		                                         xaxis_label=r"Number of voxels in interaction ($E_{{vox}} > {}$ MeV)".format(VOXEL_THRESHOLD_MEV),
		                                         yaxis_label=r"Fraction of $\nu$ interactions",
		                                         hist_labels={"n-vox-inter-reco": "Reco", "n-vox-inter-true": "True"})
		ax.set_xscale("log")
		plotting_helpers.savefig(fig, "n-vox-inter", outdir, fmts)

	hist_labels = {
		"largest-trueint-energy-matched-frac": (r"Max frac. true vis. $E_{dep}$ matched to reco int.",
		                                        r"True $\nu$ interactions"),
		"recoint-purity-frac":                 (r"Reco. interaction purity",
		                                        r"Reco. $\nu$ interactions"),
		"true-muon-reco-int-match-count":      (r"Reco. ints. w/ $\geq{0:.0f}\%$ of true vis. $E_{{\mu}}$".format(TRUE_MUON_SPLIT_FRAC*100),
		                                        "True muons"),
		"true-muon-grouped-frac":              (r"'Leading' frac. of vis. $E_{\mu}$ matched to reco int.",
	                                            "True muons"),
		"reco-int-muon-match-count":           (r"Num. muons w/ $\geq{0:.0f}\%$ of true vis. $E_{{\mu}}$".format(TRUE_MUON_MERGE_FRAC*100),
		                                        "Reco. interactions"),
	}
	for plot, (xlabel, ylabel) in hist_labels.items():
		if plot in hists:
			fig, ax = plotting_helpers.overlay_hists({plot: hists[plot]},
			                                         xaxis_label=xlabel,
			                                         yaxis_label=ylabel)
			plotting_helpers.savefig(fig, plot, outdir, fmts)


	hist2d_labels = {
		"ungrouped-trueint-energy-frac-vs-trueEdep": (r"True vis. $E_{dep}$ in interaction (GeV)",
		                                              r"Frac. true vis. $E_{dep}$ unmatched to reco int.",
		                                              r"True $\nu$ interactions"),

	}
	for plot, (xlabel, ylabel, zlabel) in hist2d_labels.items():
		if plot not in hists: continue

		fig = plt.figure()
		ax = fig.add_subplot()
		h = hists[plot]
		x, y = numpy.meshgrid(*h.bins)
		im = ax.pcolormesh(x, y, h.data.T, cmap="Reds", norm=matplotlib.colors.LogNorm())
		cbar = plt.colorbar(im)

		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		cbar.set_label(zlabel)

		plotting_helpers.savefig(fig, plot, outdir, fmts)
