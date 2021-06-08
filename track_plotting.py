
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy
import scipy.spatial

import plotting_helpers

TRACK_LABEL = 1

TRACK_THRESHOLD = 5.0 # cm

MUON_ENERGY_THRESHOLD = 10  # MeV
MUON_PURITY_THRESHOLD = 0.5
MUON_COMPLETENESS_THRESHOLD = 0.5

LONG_TRACK = 100 # cm

PDG = {
	0:  "Other",
	11: "Electron",
	13: "Muon",
	22: "Photon",
	211: "Chg. pion",
	321: "Chg. kaon",
	2212: "Proton",
}

def convert_pixel_to_geom(val, metadata):
	# assume cubical, for now...
	return val * metadata.size_voxel_x()  # note these will be in cm


def longest_track_voxels(vals):
	if "longest_track_vox" not in vals:
		trk_lengths = reco_track_lengths_cm(vals)
		if len(trk_lengths) < 1:
			return numpy.empty_like(vals["input_data"])

		# i.e.: voxels part of a track fragment that's one of the fragments in the longest track's group...
		longest_track_group_mask = vals["track_group_pred"][numpy.argmax(trk_lengths)]
#		print('track_fragments:', vals["track_fragments"])
#		print("longest_track_group_mask:", longest_track_group_mask)
		track_frag_vox_indices = vals["track_fragments"][longest_track_group_mask]
#		print("track_frag_vox_indices:", track_frag_vox_indices)

		vals["longest_track_voxels"] = vals["input_data"][track_frag_vox_indices]

	return vals["longest_track_voxels"]


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


def truth_track_pids(vals):
	""" return the pids corresponding to all true particles with 'track' semantic type """
	key = "true_track_pids"

	if key in vals:
		pids = vals[key]
	else:
		pids = numpy.array([p.pdg_code() for p in vals["particles"] if p.shape() == TRACK_LABEL])
		vals[key] = pids

	return pids


def true_muon_vox(vals):
	if "true_muon_voxels" not in vals:
		part_pdg_group = numpy.array([(p.pdg_code(), p.group_id()) for p in vals["particles"] if p.shape() == TRACK_LABEL])
#		print("part_pdg_group:", part_pdg_group)

		true_muon_groups = numpy.unique(part_pdg_group[abs(part_pdg_group[:, 0]) == 13][:, 1])
#		print("true_muon_groups:", true_muon_groups)
		if len(true_muon_groups) != 1:
			return numpy.empty(shape=(0, 4))

#		print("cluster_label:", vals["cluster_label"][:, 5])
		cluster_label = vals["cluster_label"][vals["cluster_label"][:, 5] == true_muon_groups[0]][:, (0, 1, 2, 4)]
		vals["true_muon_voxels"] = numpy.row_stack(cluster_label)  # after unwrapping (?) 'cluster_label' is a list of single-row numpy arrays
#		print("true muon voxels:", vals["true_muon_voxels"])

	return vals["true_muon_voxels"]


def matched_voxels(arr1, arr2):
	""" Get the subset of rows of arr1 that have """
	arr1_posonly = arr1[:, :3]

	# I couldn't get the [:, None] trick to work in conjunction with a slice down to the first 3 columns,
	# which is why arr1_posonly is used
	return arr1[(arr1_posonly[:, None] == arr2[:, :3]).all(-1).any(1)]


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
	lengths = truth_track_lengths_cm(vals)
	return numpy.count_nonzero(lengths > TRACK_THRESHOLD)


@plotting_helpers.hist_aggregate("n-tracks-with-long-track-reco", bins=15, range=(0,15))
def agg_ntrackslongtrk_reco(vals):
	lengths = reco_track_lengths_cm(vals)
	is_longtrk_ev = numpy.count_nonzero(lengths > LONG_TRACK) > 0
	return [-1 if not is_longtrk_ev else len(lengths),]


@plotting_helpers.hist_aggregate("n-tracks-with-long-track-true", bins=15, range=(0,15))
def agg_ntrackslongtrk_true(vals):
	lengths = truth_track_lengths_cm(vals)
	is_longtrk_ev = numpy.count_nonzero(lengths > LONG_TRACK) > 0
	return [-1 if not is_longtrk_ev else len(lengths),]


@plotting_helpers.hist_aggregate("trk-length-true", bins=numpy.logspace(-1, numpy.log10(3000), 50))
def agg_trklen_true(vals):
	return truth_track_lengths_cm(vals)


@plotting_helpers.hist_aggregate("trk-length-reco", bins=numpy.logspace(-1, numpy.log10(3000), 50))
def agg_trklen_reco(vals):
	return reco_track_lengths_cm(vals)


@plotting_helpers.hist_aggregate("trk-length-truepid", bins=numpy.logspace(0, numpy.log10(3000), 50))
def agg_trklen_truepid(vals):
	lengths = truth_track_lengths_cm(vals)
	pids = abs(truth_track_pids(vals))

	ret = {}
	for pid in numpy.unique(pids):
		if pid not in PDG:
			pid = 0
		ret["pdg=%s" % pid] = lengths[pids == pid]

	return ret


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


@plotting_helpers.hist_aggregate("mu-trk-completeness",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 2000, 50),
                                       numpy.linspace(0, 1.1, 22)))
def agg_muontrk_completeness(vals):
	"""
	For the track cluster containing the majority of the true muon voxels' energy,
	what fraction of the true muon's energy (i.e. sum of its voxels) does it have?
	"""

	true_mu_vox = true_muon_vox(vals)
	total_muon_E = true_mu_vox[:, 3].sum()
	if total_muon_E < MUON_ENERGY_THRESHOLD:
		return []

	track_indices = numpy.unique(vals["track_group_pred"])
#	print("input voxels:", vals["input_data"])
	largest_sum = 0
	for i, trk_index in enumerate(track_indices):
		voxel_indices = numpy.concatenate(
			[frag for idx, frag in enumerate(vals["track_fragments"]) if vals["track_group_pred"][idx] == trk_index])
		voxels = vals["input_data"][voxel_indices]

#		print("voxels:", voxels)
#		print("true_mu_vox:", true_mu_vox.tolist())
		shared_vox = matched_voxels(voxels, true_mu_vox)
#		print("shared_vox:", shared_vox)
		current_sum = shared_vox[:, 4].sum()
		if current_sum > largest_sum:  # recall that index 3 is the batch ID for data
			largest_sum = current_sum

	return [[total_muon_E,], [largest_sum / total_muon_E,]]


@plotting_helpers.hist_aggregate("mu-trk-found",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 2000, 50),
                                       [0, 1, 2]))
def agg_muontrk_found_vs_truemuE(vals):
	"""
	Assume for now that "muon ID" is "longest track cluster".
	How often does this track cluster contain > 50% of the true mu energy
	  AND have no more than 50% contamination from other particles
	"""

	longest_track_vox = longest_track_voxels(vals)
	true_mu_vox = true_muon_vox(vals)
	true_mu_vox_E = true_mu_vox[:, 3].sum()

	if true_mu_vox_E < MUON_ENERGY_THRESHOLD:
		return []

	# say we've correctly identified the muon (i.e. this event is "signal")
	# if > 50% of the track's voxels overlap with the true muon
	# (i.e. the track is > 50% "pure")
	# and it contains more than 50% of the muon voxels' energy
	longest_trk_E = longest_track_vox[:, 4].sum()
	# print("longest_track_vox:", longest_track_vox)
	# print("true_mu_vox:", true_mu_vox)
	# print("matching vox:", (longest_track_vox[:, None] == true_mu_vox[:, :3]))
	overlap_vox_E = matched_voxels(longest_track_vox, true_mu_vox)[:, 4].sum()  # note that in the input data, index 3 is batch id
	return [[true_mu_vox[:, 3].sum(),],
	        [1 if overlap_vox_E / longest_trk_E >= MUON_PURITY_THRESHOLD
	             and overlap_vox_E / true_mu_vox_E >= MUON_COMPLETENESS_THRESHOLD
	          else 0 ,]]


@plotting_helpers.hist_aggregate("true-mu-energy-present",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 2000, 50),
                                       [0, 1, 2]))
def agg_truemu_vs_truemuE(vals):
	"""
	Assume for now that "muon ID" is "longest track cluster".
	How often do we have a true muon track?
	"""

	true_mu_vox = true_muon_vox(vals)
	true_mu_voxE = true_mu_vox[:, 3].sum()
	if true_mu_voxE < MUON_ENERGY_THRESHOLD:
		return []
	return [[true_mu_voxE,], [1,]]


@plotting_helpers.hist_aggregate("mu-trk-purity",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 2000, 50),
                                       numpy.linspace(0, 1.1, 22)))
def agg_muontrk_purity(vals):
	"""
	Assume for now that "muon track candidate" == "longest track cluster".
	What fraction of the true muon's voxels does it have?
	"""

	longest_track_vox = longest_track_voxels(vals)
	longest_track_voxE = longest_track_vox[:, 4].sum()
	true_mu_vox = true_muon_vox(vals)
	true_mu_voxE = true_mu_vox[:, 3].sum()
	if true_mu_voxE < MUON_ENERGY_THRESHOLD:
		return []

	return [[true_mu_voxE,], [longest_track_voxE/true_mu_voxE,]]

#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "track_fragments", "track_group_pred", "particles", "metadata", "cluster_label"])
def BuildHists(data, hists):
	for evt_idx in range(len(data["particles"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_trklen_reco, agg_trklen_true,
		               agg_ntracks_reco, agg_ntracks_true,
		               agg_ntrackslongtrk_reco, agg_ntrackslongtrk_true,
		               agg_dtrklen_vs_trklen, agg_trklen_truepid,
		               agg_muontrk_completeness, agg_muontrk_found_vs_truemuE, agg_muontrk_purity):
			agg_fn(evt_data, hists)


def PlotHists(hists, outdir, fmts):
	# n-tracks plots are simple true-reco comparisons
	ntracks_hists = {hname: hists[hname] for hname in ("n-tracks-reco", "n-tracks-true")}
	if all(ntracks_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ntracks_hists,
		                                         xaxis_label="'Track' cluster multiplicity (length > %.1f cm)" % TRACK_THRESHOLD,
		                                         hist_labels={"n-tracks-reco": "Reco", "n-tracks-true": "True"})

		plotting_helpers.savefig(fig, "n-tracks", outdir, fmts)

	ntracks_longtrk_hists = {hname: hists[hname] for hname in ("n-tracks-with-long-track-reco", "n-tracks-with-long-track-true")}
	if all(ntracks_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ntracks_longtrk_hists,
		                                         xaxis_label="'Track' multiplicity (evts. with length > %.1f cm)" % LONG_TRACK,
		                                         hist_labels={"n-tracks-with-long-track-reco": "Reco", "n-tracks-with-long-track-true": "True"})

		plotting_helpers.savefig(fig, "n-tracks-longtrkcut", outdir, fmts)

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

	trklen_by_pid_hists = {hname: hists[hname] for hname in hists if hname.startswith("trk-length-truepid_pdg=")}
	if len(trklen_by_pid_hists):
		hist_labels = {}
		for hname in trklen_by_pid_hists:
			hist_labels[hname] = PDG[int(hname.split("=")[-1])]
		fig, ax = plotting_helpers.overlay_hists(trklen_by_pid_hists,
		                                         xaxis_label="True particle trajectory length (cm)",
		                                         yaxis_label="Particles",
		                                         hist_labels=hist_labels)
		ax.set_xscale("log")
		plotting_helpers.savefig(fig, "trk-length-truepid", outdir, fmts)


	for hname in ["mu-trk-completeness", "mu-trk-purity"]:
		if not hists[hname]: continue

		h = hists[hname]
		fig = plt.figure()
		ax = fig.add_subplot()
		x, y = numpy.meshgrid(*h.bins)
		im = ax.pcolormesh(x, y, h.data.T, cmap="Reds", norm=matplotlib.colors.LogNorm())
		plt.colorbar(im)

		ax.set_xlabel("Visible true muon energy (MeV)")
		ax.set_ylabel("Reco. muon track " + hname.split("-")[-1])

		# line at y=0
#		ax.axhline(0, color='black', linestyle=':')

		plotting_helpers.savefig(fig, hname, outdir, fmts)


