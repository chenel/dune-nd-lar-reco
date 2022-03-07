import copy

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy
import scipy.spatial

import plotting_helpers
import truth_functions
import utility_functions

# a muon is considered "split" if
# more than this fraction of its energy
# is found in multiple reco interactions
TRUE_MUON_SPLIT_FRAC = 0.2

# the threshold for considering a reco interaction
# to contain a "significant" part of a muon
# when counting interactions that merge muons
TRUE_MUON_MERGE_FRAC = 0.2

VOXEL_THRESHOLD_MEV = 0.25

# what's the volume inside of which events should begin in order to be "fiducial vertex"?
FIDUCIAL_EXTENTS = [
	# Chris says active volume is 714 x 300 x 507.4 cm^3.
	# I got the ranges themselves by doing some histogramming...
	[-350, 350],
	[-150, 150],
	[410, 920],
]


def true_inter_lbls(vals):
	""" Get the list of true interaction labels.

		:return: 1D numpy array of interaction labels (ints)
	"""
	if "true_inter_ids" not in vals:
		vals["true_inter_ids"] = numpy.unique(vals["cluster_label"][:, 7])

	return vals["true_inter_ids"]


def part_is_fiducial(particle):
	return all(low_extent <= coord <= high_extent for (coord, (low_extent, high_extent)) in
	           zip((particle.x(), particle.y(), particle.z()), FIDUCIAL_EXTENTS))

def is_true_inter_fid_vtx(vals, inter_lbl):
	""" Determine whether a given true interaction has a fiducial vertex

	    :param inter_lbl:  The true interaction label of interest
	    :return:  bool answering the question
	"""

	# it would be much better to use the particle info in the "cluster_label" product.
	# unfortunately that doesn't give us any way of knowing which is the *start* of particles,
	# only which voxels go with which particle.
	# so we use the raw particle list.
	if "is_true_inter_fid_vtx" not in vals:
		is_fid = {}
		for p in vals["particles_raw"]:
			if p.creation_process() != "primary":
				continue
			inter_id = p.interaction_id()
			if inter_id not in is_fid:
				is_fid[inter_id] = part_is_fiducial(p)

		# there won't be any true particles with interaction -1
		# (that's the one all the LEScatter voxels get grouped to).
		# I guess we call them "non-fiducial"?
		is_fid[-1] = False

		vals["is_true_inter_fid_vtx"] = is_fid

	assert inter_lbl in vals["is_true_inter_fid_vtx"], "Couldn't find interaction %d in list.  Known interactions: %s" % (inter_lbl, list(vals["is_true_inter_fid_vtx"].keys()))
	return vals["is_true_inter_fid_vtx"][inter_lbl]


def is_true_muon_fid(vals, muon_cluster_group_id):
	""" Determine whether a true muon cluster has a fiducial vertex

	    :param muon_cluster_group_id:  The true muon cluster id (use group_id()!) of interest
	    :return:  bool answering the question
	"""

	key = "true_muon_fid_by_id"

	if key not in vals:
		muons_are_fid = {}

		for p in vals["particles_raw"]:
			if abs(p.pdg_code()) == 13:
				muons_are_fid[p.group_id()] = part_is_fiducial(p)

		vals[key] = muons_are_fid

	if muon_cluster_group_id not in vals[key]:
		print("muon fid clusters:", vals[key])
		print("There is no true muon cluster in spill with group id: %d" % muon_cluster_group_id)
		part = None
		for p in vals["particles_raw"]:
			if p.group_id() == muon_cluster_group_id:
				part = p
				break
		if part:
			print("That particle is has pdg:", part.pdg_code())
		else:
			print("In fact, there's no particle with that group id at all!")
		assert False

	return vals[key][muon_cluster_group_id]


def true_inter_voxel_ids(vals, inter_lbl):
	"""  Get a list of the indices of voxels in "input_data" that match to the given true interaction label.

		:param inter_lbl:   The true interaction label of interest
		:return: array of indices for voxels in vals["input_data"] """

	if "true_inter_voxel_ids" not in vals:
		vals["true_inter_voxel_ids"] = {}
	if inter_lbl not in vals["true_inter_voxel_ids"]:
		# unfortunately the cluster_label and input_data vox are not guaranteed to be in the same order
		voxel_pos = vals["cluster_label"][vals["cluster_label"][:, 7] == inter_lbl, :3]
		vals["true_inter_voxel_ids"][inter_lbl] = utility_functions.find_matching_rows(numpy.array(vals["input_data"][:, :3]), numpy.array(voxel_pos))
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
		for mu_cluster_id, vox_idxs in truth_functions.true_muon_voxidxs_by_cluster(vals).items():
			total_mu_visE = vals["input_data"][vox_idxs, 4].sum()
			assert total_mu_visE > 0
#			print("muon label", mu_cluster_id, " voxels:", vox_idxs)
			for reco_inter_label in numpy.unique(vals["inter_group_pred"]):
				reco_inter_vox = reco_inter_voxel_ids(vals, reco_inter_label)
#				print("   reco interaction", reco_inter_label, "voxels:", type(reco_inter_vox), reco_inter_vox)
				matched_vox_idxs = utility_functions.find_matching_rows(vox_idxs, reco_inter_vox)
				if len(matched_vox_idxs) > 0:
					if mu_cluster_id not in true_mu_recoint_match:
						true_mu_recoint_match[mu_cluster_id] = {}
					true_mu_recoint_match[mu_cluster_id][reco_inter_label] = vals["input_data"][vox_idxs[matched_vox_idxs], 4].sum() / total_mu_visE

		vals[key] = true_mu_recoint_match

#	print(vals[key])
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

	match_count_by_fid = {"fid": [], "nonfid": []}
	for mu_clus_id, match_fracs in true_muon_recoint_matches.items():
		key = "fid" if is_true_muon_fid(vals, mu_clus_id) else "nonfid"
		match_count_by_fid[key].append(sum(1 if frac >= TRUE_MUON_SPLIT_FRAC else 0 for frac in match_fracs.values()))

	return match_count_by_fid


@plotting_helpers.hist_aggregate("true-muon-grouped-frac", bins=27, range=(0, 1.08))
def agg_true_muon_grouped_frac(vals):
	"""
	Compute the fraction of the true muon's energy contained by the reco interaction
	that contains the most of its energy.
	"""
	true_muon_recoint_matches = true_muon_reco_matches(vals)

	matches_by_fid = {"fid": [], "nonfid": []}
	for mu_clus_id, match_fracs in true_muon_recoint_matches.items():
		key = "fid" if is_true_muon_fid(vals, mu_clus_id) else "nonfid"
		matches_by_fid[key].append(sum(match_fracs.values()))

	return matches_by_fid


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
	return [len(numpy.unique(vals["inter_group_pred"])),]


@plotting_helpers.hist_aggregate("n-inter-true", bins=50, range=(0,100))
def agg_ninter_true(vals):
	ninter_by_fid = {"fid": [0,], "nonfid": [0,]}
	for lbl in true_inter_lbls(vals):
		ninter_by_fid["fid" if is_true_inter_fid_vtx(vals, lbl) else "nonfid"][0] += 1
	return ninter_by_fid


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
	nvox_by_fid = {"fid": [], "nonfid": []}
	for lbl, count in nvox.items():
		nvox_by_fid["fid" if is_true_inter_fid_vtx(vals, lbl) else "nonfid"].append(count)
	return nvox_by_fid


@plotting_helpers.hist_aggregate("ungrouped-trueint-energy-frac-vs-trueEdep",
                                 hist_dim=2, bins=(numpy.linspace(0, 5, 25),
                                                   numpy.linspace(0, 1.08, 27)))
def agg_ungrouped_trueint_energy_frac_vs_trueEdep(vals):

	inter_unmatch_frac = {"fid": [[], []], "nonfid": [[], []]}
	all_reco_vox = vals["input_data"][numpy.unique(numpy.concatenate(vals["inter_particles"]))]
	for inter_lbl in true_inter_lbls(vals):
		# the "-1" label corresponds to LEScatters (which won't otherwise be grouped into anything).
		# almost all of its energy is not matched, *correctly*
		if inter_lbl < 0:
			continue

		key = "fid" if is_true_inter_fid_vtx(vals, inter_lbl) else "nonfid"

		true_vox = vals["input_data"][true_inter_voxel_ids(vals, inter_lbl)]
		true_E_sum = true_vox[:, 4].sum()
		assert(true_E_sum > 0)

		matched_vox_indices = utility_functions.find_matching_rows(numpy.array(all_reco_vox[:, :3]), numpy.array(true_vox[:, :3]))
		matched_E_sum = all_reco_vox[matched_vox_indices][:, 4].sum()

		inter_unmatch_frac[key][0].append(true_E_sum * 1e-3)  # convert to GeV
		inter_unmatch_frac[key][1].append(1 - matched_E_sum / true_E_sum)

		if matched_E_sum / true_E_sum < 0.2 and true_E_sum > 2000:
			print("True interaction with high true E_dep (", true_E_sum, "MeV)\n"
			      " and very little matched to reco interaction (", matched_E_sum, "):",
			      vals["event_base"], inter_lbl)
#			print("Matched voxels:", all_reco_vox[matched_vox_indices][:, :3], sep='\n')

	return inter_unmatch_frac


@plotting_helpers.hist_aggregate("largest-trueint-energy-matched-frac", bins=27, range=(0,1.08))
def agg_trueint_largest_matched_energy_frac(vals):

	inter_match_frac = {"fid": [], "nonfid": []}
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
			matched_vox_mask = utility_functions.find_matching_rows(numpy.array(reco_vox[:, :3]), true_vox_copy, mask_only=True)
			matched_E = reco_vox[matched_vox_mask][:, 4].sum()
			# if matched_E > 0:
			# 	print("    particle", part_idx, "has", len(reco_vox), "voxels, of which")
			# 	print("        ", numpy.count_nonzero(matched_vox_mask), "match to the true interaction for total matched energy of", matched_E)

			max_matched_E = max(max_matched_E, matched_E)

		inter_match_frac["fid" if is_true_inter_fid_vtx(vals, inter_lbl) else "nonfid"].append(max_matched_E / true_E_sum)

	return inter_match_frac


@plotting_helpers.hist_aggregate("recoint-purity-frac", bins=27, range=(0,1.08))
def agg_recoint_purity(vals):

	reco_purity = {"fid": [], "nonfid": []}
	for inter in numpy.unique(vals["inter_group_pred"]):
		inter_vox_ids = reco_inter_voxel_ids(vals, inter)
		reco_vox = vals["input_data"][inter_vox_ids]

		matched_energy = {}
		for idx, true_inter_lbl in enumerate(true_inter_lbls(vals)):
			true_vox = vals["input_data"][true_inter_voxel_ids(vals, true_inter_lbl)]
			true_vox_copy = numpy.array(true_vox[:, :3])

			matched_vox_mask = utility_functions.find_matching_rows(numpy.array(reco_vox[:, :3]), true_vox_copy, mask_only=True)
			matched_E = reco_vox[matched_vox_mask][:, 4].sum()
			if matched_E > 0:
				matched_energy[true_inter_lbl] = matched_E

		max_match_true_int = max(matched_energy, key=matched_energy.get)

		# all true "low-energy scatters" are lumped together into true interaction -1,
		# so if they are the leading energy depositor to this reco interaction,
		# we can't work out the purity (impossible to say which *actual* true nu int they came from).
		# so we just skip any like that.
		if max_match_true_int >= 0:
			hist = "fid" if is_true_inter_fid_vtx(vals, max_match_true_int) else "nonfid"
			reco_purity[hist].append(matched_energy[max_match_true_int] / reco_vox[:, 4].sum())

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
	ninter_hists = {hname: hists[hname] for hname in hists if hname.startswith("n-inter")}
	if all(ninter_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ninter_hists,
		                                         xaxis_label=r"$\nu$ interaction multiplicity",
		                                         yaxis_label="Spills",
		                                         hist_labels={hname: "Reco" if hname.endswith("reco") else
                                                              ("True non-fiducial" if hname.endswith("nonfid") else
	                                                          "True fiducial") for hname in ninter_hists})

		plotting_helpers.savefig(fig, "n-inter", outdir, fmts)

	nvox_inter_hists = {hname: hists[hname] for hname in hists if hname.startswith("n-vox-inter")}
	if all(nvox_inter_hists.values()):

		# since these will be stacked we need to normalize them in a special way
		truth_hists = [h for h in nvox_inter_hists if "true" in h]
		truth_sum = sum(hists[h].data.sum() for h in truth_hists)
		for h in truth_hists:
			hist_sum = hists[h].data.sum()
			if hist_sum > 0:
				hists[h].norm = truth_sum / hist_sum


		for hname in nvox_inter_hists:
			hists[hname].Normalize()
		fig, ax = plotting_helpers.overlay_hists(nvox_inter_hists,
		                                         xaxis_label=r"Number of voxels in interaction ($E_{{vox}} > {}$ MeV)".format(VOXEL_THRESHOLD_MEV),
		                                         yaxis_label=r"Fraction of $\nu$ interactions",
		                                         stack=[[hname for hname in nvox_inter_hists if "true" in hname],],
		                                         hist_labels={hname: "Reco" if hname.endswith("reco") else
		                                                             ("True non-fiducial" if hname.endswith("nonfid") else
			                                                         "True fiducial") for hname in nvox_inter_hists})
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
		this_plot_hists = [h for h in hists if h.startswith(plot)]
		if len(this_plot_hists) > 0:
			kwargs = {
				"hists": {h: hists[h] for h in this_plot_hists},
				"xaxis_label": xlabel,
				"yaxis_label": ylabel,
			}
			if any(h.endswith("nonfid") for h in this_plot_hists):
				kwargs["stack"] = [[h for h in this_plot_hists if h.endswith("fid")],]  # i.e., both "_fid" and "_nonfid"
				kwargs["hist_labels"] = {h:       "True non-fiducial" if h.endswith("nonfid")
				                             else "True fiducial" for h in this_plot_hists}
			fig, ax = plotting_helpers.overlay_hists(**kwargs)
			plotting_helpers.savefig(fig, plot, outdir, fmts)


	hist2d_labels = {
		"ungrouped-trueint-energy-frac-vs-trueEdep": (r"True vis. $E_{dep}$ in interaction (GeV)",
		                                              r"Frac. true vis. $E_{dep}$ unmatched to reco int.",
		                                              r"True $\nu$ interactions"),

	}
	for plot, (xlabel, ylabel, zlabel) in hist2d_labels.items():
		this_plot_hists = [h for h in hists if h.startswith(plot)]
		if len(this_plot_hists) == 0:
			continue

		for subplot in this_plot_hists:
			fig = plt.figure()
			ax = fig.add_subplot()
			h = hists[subplot]
			x, y = numpy.meshgrid(*h.bins)
			im = ax.pcolormesh(x, y, h.data.T, cmap="Reds", norm=matplotlib.colors.LogNorm())
			cbar = plt.colorbar(im)

			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			cbar.set_label(zlabel)

			plotting_helpers.savefig(fig, subplot, outdir, fmts)
