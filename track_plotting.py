import copy

from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib.style
import matplotlib.transforms
import numpy
import scipy.spatial

import plotting_helpers
import track_functions
import truth_functions

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

TWOD_ANGLES_BINS = dict( bins=37, range=(2.5, 187.5) )

def completeness(vals):
	keys = ("total_muon_E", "largest_matched_muonE", "longesttrk_matched_muonE")
	if any(k not in vals for k in keys):
		true_mu_vox = true_muon_vox(vals)
		total_muon_E = true_mu_vox[:, 3].sum()
		if total_muon_E < MUON_ENERGY_THRESHOLD:
			return []

		# compute completeness for *longest* track ...
		longest_track_vox = longest_track_voxels(vals)
		shared_vox = matched_voxels(longest_track_vox, true_mu_vox)
		vals["longesttrk_matched_muonE"] = shared_vox[:, 4].sum()

		# ... as well as for the track with the largest amount of matched Emu
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

		vals["largest_matched_muonE"] = largest_sum

		vals["total_muon_E"] = total_muon_E

	return {k: vals[k] for k in keys}


def convert_pixel_to_geom(val, metadata):
	# assume cubical, for now...
	assert metadata.size_voxel_x() == metadata.size_voxel_y() == metadata.size_voxel_z()
	return val * metadata.size_voxel_x()  # note these will be in cm


def twod_angle(normal_axis, reference_axis, track_dir):
	if track_dir is None:
		return None

	# first project into the 2D plane
	normal_axis = normal_axis / numpy.linalg.norm(normal_axis)  # assume that user wasn't pathological and didn't pass zero-length vector...
	reference_axis = reference_axis / numpy.linalg.norm(reference_axis)
	projected_track = track_dir - numpy.dot(track_dir, normal_axis) * normal_axis
	if numpy.count_nonzero(projected_track) < 1:
		return None
#	print("projected_track:", projected_track)

	# then take dot product with reference axis
	cos_theta = numpy.dot(projected_track, reference_axis) / numpy.linalg.norm(projected_track)

	if cos_theta is None or cos_theta < -1 or cos_theta > 1:
		print("projected_track:", projected_track)
		print("reference_axis:", reference_axis)
		print("wacky cos_theta =", cos_theta)
		exit(1)

	ang = numpy.rad2deg(numpy.arccos(cos_theta))
#	print("angle:", ang)
	return ang



def longest_track_voxels(vals):
	if "longest_track_vox" not in vals:
		trk_lengths = reco_track_lengths_cm(vals)
		if len(trk_lengths) < 1:
			return numpy.empty_like(vals["input_data"])

		# i.e.: voxels part of a track fragment that's one of the fragments in the longest track's group...
		# print("trk_lengths:", trk_lengths)
		# print("track_group_pred:", vals["track_group_pred"])
		# print("longest track group:", numpy.nanargmax(trk_lengths))
		longest_track_group_mask = vals["track_group_pred"] == numpy.nanargmax(trk_lengths)
		# print("longest_track_group_mask:", type(longest_track_group_mask), longest_track_group_mask)
		# print('track_fragments:', vals["track_fragments"])

		# for whatever reason, after unwrapping (?)
		# the "track_fragments" are in a Python list
		# with one entry per fragment.
		# so the result of masking is also a list.
		# thus the concatenation
		track_frag_vox_indices = numpy.concatenate(vals["track_fragments"][longest_track_group_mask])
		# print("track_frag_vox_indices:", track_frag_vox_indices)

		vals["longest_track_voxels"] = vals["input_data"][track_frag_vox_indices]

	return vals["longest_track_voxels"]


def longest_track_purity_vars(vals):
	vars = ["true_mu_vox_E", "longest_trk_E", "overlap_vox_E"]
	if any(v not in vals for v in vars):
		longest_track_vox = longest_track_voxels(vals)
		true_mu_vox = true_muon_vox(vals)
		true_mu_vox_E = true_mu_vox[:, 3].sum()

		if true_mu_vox_E < MUON_ENERGY_THRESHOLD:
			return None, None, None

		# say we've correctly identified the muon (i.e. this event is "signal")
		# if > 50% of the track's voxels overlap with the true muon
		# (i.e. the track is > 50% "pure")
		# and it contains more than 50% of the muon voxels' energy
		longest_trk_E = longest_track_vox[:, 4].sum()
		numpy.set_printoptions(suppress=True)
		overlap_vox = matched_voxels(longest_track_vox, true_mu_vox)
		overlap_vox_E = overlap_vox[:, 4].sum()  # note that in the input data, index 3 is batch id
		# print("longest_track_vox:", longest_track_vox)
		# print("true_mu_vox:", true_mu_vox)
		# print("overlapping vox:", overlap_vox)
		# print("longest_track E:", longest_trk_E)
		# print("true_mu E:", true_mu_vox_E)
		# print("overlap_vox_E:", overlap_vox_E)

		vals["true_mu_vox_E"] = true_mu_vox_E
		vals["longest_trk_E"] = longest_trk_E
		vals["overlap_vox_E"] = overlap_vox_E

	return vals["true_mu_vox_E"], vals["longest_trk_E"], vals["overlap_vox_E"]


def reco_track_begin_dir(trk_index, vals):
	product_name = "track_begin_dir"
	if product_name in vals and trk_index in vals[product_name]:
		return vals[product_name][trk_index]

	if product_name not in vals:
		vals[product_name] = {}

	voxels = track_functions.track_voxel_coords(trk_index, vals, vals)

	endpoints = track_functions.track_endpoints(trk_index, vals, vals)

	# use the "track end direction" machinery to find the track *beginning* direction
	# (look from the "beginning" point)
	dists = track_functions.track_voxel_dists(trk_index, vals, vals, voxels)
	dists_to_front = dists[numpy.where((voxels == endpoints[0]).all(axis=1))[0]][0]
	dir_vec = track_functions.track_end_dir(voxels, dists_to_front, endpoints)

	vals[product_name][trk_index] = dir_vec

	return vals[product_name][trk_index]


def reco_tracks_2d_angles(reference_axis, normal_axis, vals):
	""" Return the angles of all >100cm tracks within a 2D plane (defined by its normal vector) relative to a reference axis. """

#	track_indices = numpy.unique(vals["track_group_pred"])

	angles = []
#	for trk_index in track_indices:
	trk_lengths = reco_track_lengths_cm(vals)
	if len(trk_lengths) < 1:
		return []
#	longest_track_idx = numpy.nanargmax(trk_lengths)
	track_indices = numpy.nonzero(trk_lengths >= LONG_TRACK)[0]
#	print("track_indices:", track_indices)

	for trk_index in track_indices:
		track_dir = reco_track_begin_dir(trk_index, vals)

		theta = twod_angle(normal_axis, reference_axis, track_dir)
		if theta is None:
			print("track dir", track_dir, "has no projection onto plane specified by normal", normal_axis, "and reference axis", reference_axis)
		angles += [theta,] if theta is not None else []

	return angles


def reco_track_lengths_cm(vals):
	""" return the lengths of all reco track clusters in units of cm """
	if "reco_track_lengths" in vals:
		lengths = vals["reco_track_lengths"]
	else:
		track_indices = numpy.unique(vals["track_group_pred"])
		lengths = numpy.full((max(track_indices)+1) if len(track_indices) > 0 else 0, numpy.nan)
		for trk_index in track_indices:
			voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(vals["track_fragments"]) if vals["track_group_pred"][idx] == trk_index])
			voxels = vals["input_data"][voxel_indices][:, :3]
			lengths[trk_index] = numpy.max(scipy.spatial.distance.cdist(voxels, voxels))
		vals["reco_track_lengths"] = lengths   # store so if called again for this event we won't have to recalculate

	return lengths


def truth_track_lengths_cm(vals):
	""" return the lengths of all true particles with 'track' semantic type in units of cm """
	if "true_track_lengths" in vals:
		lengths = vals["true_track_lengths"]
	else:
		lengths = numpy.array([p.first_step().as_point3d().distance(p.last_step().as_point3d())
		                       for p in vals["particles_raw"]
		                       if p.shape() == TRACK_LABEL])
		vals["true_track_lengths"] = lengths

	return lengths


def truth_track_pids(vals):
	""" return the pids corresponding to all true particles with 'track' semantic type """
	key = "true_track_pids"

	if key in vals:
		pids = vals[key]
	else:
		pids = numpy.array([p.pdg_code() for p in vals["particles_raw"] if p.shape() == TRACK_LABEL])
		vals[key] = pids

	return pids


def true_muon_vox(vals):
	if "true_muon_voxels" not in vals:
		part_pdg_group = numpy.array([(p.pdg_code(), p.group_id()) for p in vals["particles_raw"] if p.shape() == TRACK_LABEL])
		if len(part_pdg_group) < 1:
			return numpy.empty(shape=(0, 4))

		true_muon_groups = numpy.unique(part_pdg_group[abs(part_pdg_group[:, 0]) == 13][:, 1])
#		print("true_muon_groups:", true_muon_groups)
		if len(true_muon_groups) != 1:
			return numpy.empty(shape=(0, 4))

#		print("cluster_label:", vals["cluster_label"][:, 5])
		cluster_label = vals["cluster_label"][vals["cluster_label"][:, 5] == true_muon_groups[0]][:, (0, 1, 2, 4)]
		if len(cluster_label) > 1:
			cluster_label =  numpy.row_stack(cluster_label)  # after unwrapping (?) 'cluster_label' is a list of single-row numpy arrays
		vals["true_muon_voxels"] = cluster_label
#		numpy.set_printoptions(suppress=True)
#		print("true muon voxels:", vals["true_muon_voxels"])

	return vals["true_muon_voxels"]


def all_true_muon_begin_dirs(vals):
	product_name = "true_muon_begin_dir"
	if product_name in vals:
		return vals[product_name]

	if product_name not in vals:
		vals[product_name] = {}

	all_truemus_vox_by_cluster = truth_functions.true_muon_voxidxs_by_cluster(vals)
#	print("true_mu_vox_by_cluster:", all_truemus_vox_by_cluster)
#	voxels = true_muon_vox(vals)[:, :3]
	if len(all_truemus_vox_by_cluster) == 0:
		return vals[product_name]

	for clus_id, vox_idxs in all_truemus_vox_by_cluster.items():
		voxels = vals["input_data"][vox_idxs, :3]
		# forge a track ID so that it has some meaningful index in the map.
		# offset of 0.5 is so that we don't overlap the reco track 0 (since -0 = 0)
		trk_index = -(clus_id+0.5)
		endpoints = track_functions.track_endpoints(trk_index, vals, vals, voxels)

		# use the "track end direction" machinery to find the track *beginning* direction
		# (look from the "beginning" point)
	#	print("voxels:", voxels)
	#   print("endpoints:", endpoints)
		dists = track_functions.track_voxel_dists(trk_index, vals, vals, voxels)
	#	print("output of == test:", voxels == endpoints[0])
		dists_to_front = dists[numpy.where((voxels == endpoints[0]).all(axis=1))[0]]
		dir_vec = track_functions.track_end_dir(voxels, dists_to_front, endpoints)
		dir_vec *= -1  # the vector from track_end_dir() points outward from the endpoint; we want to point inwards for this

		vals[product_name][trk_index] = dir_vec
#	print("cluster labels:", list(vals[product_name].keys()))

	return vals[product_name]


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
	for p in vals["particles_raw"]:
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


@plotting_helpers.hist_aggregate("trk-thetax-reco", **TWOD_ANGLES_BINS)
def agg_trkanglex_reco(vals):
	# theta_x = angle w.r.t. x-axis within x-z plane
	return reco_tracks_2d_angles(reference_axis=numpy.array([1, 0, 0]), normal_axis=([0, 1, 0]), vals=vals)


@plotting_helpers.hist_aggregate("trk-thetay-reco", **TWOD_ANGLES_BINS)
def agg_trkangley_reco(vals):
	# theta_y = angle w.r.t. y-axis within y-z plane
	return reco_tracks_2d_angles(reference_axis=numpy.array([0, 1, 0]), normal_axis=([1, 0, 0]), vals=vals)


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
	longest_true = numpy.nanmax(truth_lengths) if len(truth_lengths) > 0 else None
	reco_lengths = reco_track_lengths_cm(vals)
	longest_reco = numpy.nanmax(reco_lengths) if len(reco_lengths) > 0 else None
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


@plotting_helpers.hist_aggregate("delta-longest-trk-angle",
                                 bins=numpy.linspace(0, 182, 92))
def agg_dcostheta(vals):
#	true_begin_dirs = all_true_muon_begin_dirs(vals)
#	print("reco track lengths:", reco_track_lengths_cm(vals))
	reco_lengths = reco_track_lengths_cm(vals)
	if len(reco_lengths) == 0:
		return []
	max_length_idx = numpy.nanargmax(reco_lengths)
	reco_begin_dir = reco_track_begin_dir(max_length_idx, vals)

	true_begin_dir = None
	for p in vals["particles_raw"]:
		if abs(p.pdg_code()) != 13:
			continue
		# only one muon shall pass
		if true_begin_dir is not None:
			return []
		true_begin_dir = numpy.array([p.px(), p.py(), p.pz()])

	if true_begin_dir is None or reco_begin_dir is None:
		return []

	if numpy.count_nonzero(true_begin_dir) - numpy.count_nonzero(numpy.isnan(true_begin_dir)) == 0 \
			or numpy.count_nonzero(reco_begin_dir) - numpy.count_nonzero(numpy.isnan(reco_begin_dir)) == 0:
		return []

	costheta =  numpy.dot(true_begin_dir, reco_begin_dir) / numpy.linalg.norm(true_begin_dir) / numpy.linalg.norm(reco_begin_dir)
	# print("reco dir:", reco_begin_dir)
	# print("true dir:", true_begin_dir)
	# print("cos theta:", costheta)
	return [numpy.degrees(numpy.arccos(costheta))]



@plotting_helpers.hist_aggregate("mu-trk-mostEmu-completeness-vs-muonVisE",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 1500, 30),
                                       numpy.linspace(0, 1.1, 22)))
def agg_muontrk_mostEmu_completeness_vs_muonVisE(vals):
	"""
	For the track cluster containing the majority of the true muon voxels' energy,
	what fraction of the true muon's energy (i.e. sum of its voxels) does it have?
	"""

	calc_vals = completeness(vals)
#	print("calc_vals:", calc_vals)

	if len(calc_vals) > 0:
		return [[calc_vals["total_muon_E"],], [calc_vals["largest_matched_muonE"] / calc_vals["total_muon_E"],]]
	else:
		return []


@plotting_helpers.hist_aggregate("mu-trk-longest-completeness-vs-muonVisE",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 1500, 30),
                                       numpy.linspace(0, 1.1, 22)))
def agg_muontrk_longest_completeness_vs_muonVisE(vals):
	"""
	For the longest track cluster,
	what fraction of the true muon's energy (i.e. sum of its voxels) does it have?
	"""

	calc_vals = completeness(vals)
#	print("calc_vals:", calc_vals)

	if len(calc_vals) > 0:
		return [[calc_vals["total_muon_E"],], [calc_vals["longesttrk_matched_muonE"] / calc_vals["total_muon_E"],]]
	else:
		return []


@plotting_helpers.hist_aggregate("mu-trk-mostEmu-completeness-vs-truemuKE",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 1500, 30),
                                       numpy.linspace(0, 1.1, 22)))
def agg_muontrk_completeness_vs_truemuKE(vals):
	calc_vals = completeness(vals)

	true_mu_KE = None
	for p in vals["particles_raw"]:
		if abs(p.pdg_code()) != 13: continue

		true_mu_KE = p.energy_init() - 105.658

	if not true_mu_KE:
		return []

	if len(calc_vals) > 0:
		return [[true_mu_KE,], [calc_vals["largest_matched_muonE"] / calc_vals["total_muon_E"],]]
	else:
		return []


@plotting_helpers.hist_aggregate("mu-trk-found", bins=30, range=(0, 1500))
def agg_muontrk_found_vs_truemuE(vals):
	"""
	Assume for now that "muon ID" is "longest track cluster".
	How often does this track cluster contain > 50% of the true mu energy
	  AND have no more than 50% contamination from other particles
	"""
	true_mu_vox_E, longest_trk_E, overlap_vox_E = longest_track_purity_vars(vals)
	if any(x is None for x in (true_mu_vox_E, longest_trk_E, overlap_vox_E)):
		return []

	if overlap_vox_E / longest_trk_E >= MUON_PURITY_THRESHOLD \
			and overlap_vox_E / true_mu_vox_E >= MUON_COMPLETENESS_THRESHOLD:
		return [true_mu_vox_E,]
	else:
		return []


@plotting_helpers.hist_aggregate("mu-trk-found-completeness", bins=30, range=(0, 1500))
def agg_muontrk_found_completeness_vs_truemuE(vals):
	true_mu_vox_E, longest_trk_E, overlap_vox_E = longest_track_purity_vars(vals)
	if any(x is None for x in (true_mu_vox_E, longest_trk_E, overlap_vox_E)):
		return []

	if overlap_vox_E / true_mu_vox_E >= MUON_COMPLETENESS_THRESHOLD:
		return [true_mu_vox_E,]
	else:
		if true_mu_vox_E > 600:
			print("Event w/ >600 MeV of true mu Evis and < 50% mu completeness on longest track:", vals["event_base"])
		return []


@plotting_helpers.hist_aggregate("mu-trk-found-purity", bins=30, range=(0, 1500))
def agg_muontrk_found_purity_vs_truemuE(vals):
	true_mu_vox_E, longest_trk_E, overlap_vox_E = longest_track_purity_vars(vals)
	if any(x is None for x in (true_mu_vox_E, longest_trk_E, overlap_vox_E)):
		return []

	if overlap_vox_E / longest_trk_E >= MUON_PURITY_THRESHOLD:
		return [true_mu_vox_E, ]
	else:
		return []


@plotting_helpers.hist_aggregate("true-mu-energy-present", bins=30, range=(0, 1500))
def agg_truemu_vs_truemuE(vals):
	"""
	Assume for now that "muon ID" is "longest track cluster".
	How often do we have a true muon track?
	"""

	true_mu_vox = true_muon_vox(vals)
	true_mu_voxE = true_mu_vox[:, 3].sum()
	if true_mu_voxE < MUON_ENERGY_THRESHOLD:
		return []
	return [true_mu_voxE,]


@plotting_helpers.hist_aggregate("mu-trk-purity-vs-muonVisE",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 1500, 30),
                                       numpy.linspace(0, 1.1, 22)))
def agg_muontrk_purity_vs_muonVisE(vals):
	"""
	Assume for now that "muon track candidate" == "longest track cluster".
	What fraction of the true muon's voxels does it have?
	"""

	longest_track_vox = longest_track_voxels(vals)
	longest_track_voxE = longest_track_vox[:, 4].sum()

	true_mu_vox = true_muon_vox(vals)
	if len(true_mu_vox) < 1:  # no visible muon in this event anyway
		return []
	true_mu_voxE = true_mu_vox[:, 3].sum()

	matched_vox = matched_voxels(longest_track_vox, true_mu_vox)
	matched_voxE = matched_vox[:, 4].sum()

	return [[true_mu_voxE,], [matched_voxE/longest_track_voxE,]]


@plotting_helpers.hist_aggregate("truemu-thetax", **TWOD_ANGLES_BINS)
def agg_truemu_thetax(vals):
	"""
	Plot the angle w.r.t. x-axis within x-z plane
	"""
	ret = []
	for end_dir in all_true_muon_begin_dirs(vals).values():
#		print("end_dir:", end_dir)
		angle = twod_angle(reference_axis=numpy.array([1, 0, 0]), normal_axis=([0, 1, 0]), track_dir=end_dir)
		if angle is not None:
			ret.append(angle)
#			print("angle:", angle)
	return ret


@plotting_helpers.hist_aggregate("truemu-thetay", **TWOD_ANGLES_BINS)
def agg_truemu_thetay(vals):
	"""
	Plot the angle w.r.t. y-axis within y-z plane
	"""
	ret = []
	for end_dir in all_true_muon_begin_dirs(vals).values():
		angle = twod_angle(reference_axis=numpy.array([0, 1, 0]), normal_axis=([1, 0, 0]), track_dir=end_dir)
		if angle is not None:
			ret.append(angle)
	return ret


@plotting_helpers.hist_aggregate("truemu-visE-vs-len",
                                 hist_dim=2,
                                 bins=(numpy.linspace(0, 1500, 30),
                                       numpy.linspace(0, 2.0, 40)))
def agg_truemu_visE_vs_len(vals):
	"""
	Plot true muon energy deposited as a function of true track length.
	"""

	true_mu_vox = true_muon_vox(vals)
	if len(true_mu_vox) < 1:  # no visible muon in this event anyway
		return []
	true_mu_voxE = true_mu_vox[:, 3].sum() * 1e-3

	truth_track_lens = truth_track_lengths_cm(vals)
	true_mu_len = truth_track_lens[numpy.abs(truth_track_pids(vals)) == 13]
	if len(true_mu_len) != 1:
		return []

	# print("true_mu_len:", true_mu_len[0])
	# print("true_mu_voxE:", true_mu_voxE)

	return [[true_mu_len[0],], [true_mu_voxE,]]

#------------------------------------------------------

@plotting_helpers.req_vars_hist(["input_data", "event_base", "track_fragments", "track_group_pred", "particles_raw", "cluster_label", "ppn_post"])
#@plotting_helpers.req_vars_hist(["input_data", "event_base", "track_fragments", "track_group_pred", "ppn_post"])
def BuildHists(data, hists):
	nevts = len(data["particles_raw"]) if "particles_raw" in data else len(data["input_data"])
	for evt_idx in range(nevts):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (
				       agg_trklen_reco, #agg_trklen_true,
		               agg_ntracks_reco, #agg_ntracks_true,
		               agg_trkanglex_reco, agg_trkangley_reco,
		               #agg_truemu_thetax, agg_truemu_thetay,
		               #agg_dcostheta,
		               agg_ntrackslongtrk_reco, #agg_ntrackslongtrk_true,
		               #agg_dtrklen_vs_trklen,
				       agg_trklen_truepid,
		               # agg_muontrk_mostEmu_completeness_vs_muonVisE, agg_muontrk_completeness_vs_truemuKE,
		               # agg_muontrk_longest_completeness_vs_muonVisE,
		               agg_muontrk_found_vs_truemuE, agg_muontrk_found_purity_vs_truemuE, agg_muontrk_found_completeness_vs_truemuE,
		               agg_truemu_vs_truemuE, agg_truemu_visE_vs_len,
		               # agg_muontrk_purity_vs_muonVisE
		):
			agg_fn(evt_data, hists)


def PlotHists(hists, outdir, fmts):
	# n-tracks plots are simple true-reco comparisons
	ntracks_hists = {hname: hists[hname] for hname in ("n-tracks-reco", "n-tracks-true") if hname in hists}
	if len(ntracks_hists) > 0 and all(ntracks_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ntracks_hists,
		                                         xaxis_label="'Track' cluster multiplicity (length > %.1f cm)" % TRACK_THRESHOLD,
		                                         hist_labels={"n-tracks-reco": "Reco", "n-tracks-true": "True"})

		plotting_helpers.savefig(fig, "n-tracks", outdir, fmts)

	ntracks_longtrk_hists = {hname: hists[hname] for hname in ("n-tracks-with-long-track-reco", "n-tracks-with-long-track-true") if hname in hists}
	if len(ntracks_longtrk_hists) > 0 and all(ntracks_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(ntracks_longtrk_hists,
		                                         xaxis_label="'Track' multiplicity (evts. with length > %.1f cm)" % LONG_TRACK,
		                                         hist_labels={"n-tracks-with-long-track-reco": "Reco", "n-tracks-with-long-track-true": "True"})

		plotting_helpers.savefig(fig, "n-tracks-longtrkcut", outdir, fmts)

	trklen_hists = {hname: hists[hname] for hname in ("trk-length-reco", "trk-length-true") if hname in hists}
	if all(trklen_hists.values()):
		fig, ax = plotting_helpers.overlay_hists(trklen_hists,
		                                         xaxis_label="'Track' length (cm)",
		                                         yaxis_label="Tracks",
		                                         hist_labels={"trk-length-reco": "Reco", "trk-length-true": "True"})
		ax.set_xscale("log")
		plotting_helpers.savefig(fig, "trk-len", outdir, fmts)

	for axis in ("x", "y"):
		angles_hists = {hname: hists[hname] for hname in ("truemu-theta" + axis, "trk-theta%s-reco" % axis) if hname in hists}
		if any(angles_hists.values()):
			fig, ax = plotting_helpers.overlay_hists(angles_hists,
			                                         xaxis_label=r"$\theta_{%s}$ (degrees)" % axis,
			                                         yaxis_label="Muon candidates", #"Tracks",
			                                         hist_labels={"trk-theta%s-reco" % axis: "Muon candidates", #r"Reco tracks $\geq %d$ cm" % LONG_TRACK,
			                                                      "truemu-theta" + axis: "True muons"})

			cfg = dict( ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color="black")
			blended_xform = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
			annotate_cfg = dict(  textcoords=blended_xform, xycoords=blended_xform,
			                      transform=blended_xform,
			                      arrowprops={"facecolor": "black", "width": 1, "headlength": 5, "shrink": 0.1, "headwidth": 5,
			                                  "connectionstyle": "angle3"} )
			plt.vlines(x=90, label=r"$90^{\circ}$", linestyles="dashed", **cfg)
			if axis == "y":
				plt.vlines(x=95.8, label=r"Beam angle = $95.8^{\circ}$", linestyles="dotted", **cfg)
				plt.annotate(r"Beam angle = $95.8^{\circ}$", xy=(95.8, 0.98), xytext=(115, 0.85),
				             horizontalalignment="left", verticalalignment="center",
				             **annotate_cfg)

#			ax.legend()
			plt.annotate(r"$90^{\circ}$", xy=(90, 0.96), xytext=(75, 0.9),
			             horizontalalignment="right", verticalalignment="center",
			             **annotate_cfg)

			ax.get_legend().remove()
			plt.xlim(left=0)
			plt.ylim(bottom=0)

			plotting_helpers.savefig(fig, "theta" + axis, outdir, fmts)


	if "truemu-visE-vs-len" in hists:
		h = hists["truemu-visE-vs-len"]
		fig = plt.figure()
		ax = fig.add_subplot()
		x, y = numpy.meshgrid(*h.bins)
		im = ax.pcolormesh(x, y, h.data.T, cmap="Reds")
		plt.colorbar(im)

		ax.set_xlabel("True muon track length in active vol. (cm)")
		ax.set_ylabel("True visible energy deposited (GeV)")

		plotting_helpers.savefig(fig, "truemu-visE-vs-len", outdir, fmts)


	if "delta-longest-trk-angle" in hists and hists["delta-longest-trk-angle"] and hists["delta-longest-trk-angle"].data.sum() > 0:
		h = hists["delta-longest-trk-angle"]
		fig, ax = plotting_helpers.overlay_hists({"dcostheta": h},
		                                         xaxis_label=r"$\theta_{true,reco} = \cos^{-1} \left(\hat{r}_{true\ \mu} \cdot \hat{r}_{longest\ reco} \right)$")

		# compute stddev
		plt.text(0.6, 0.8, r"Std. dev. = $%.2f^{\circ}$" % h.StdDev(), transform=ax.transAxes)
		plt.text(0.6, 0.7, r"Std. dev. ($\theta < 25^{\circ}$) = $%.2f^{\circ}$" % h.StdDev(bin_range=[0, numpy.nanargmax(h.bins[h.bins < 25])]), transform=ax.transAxes)

		plotting_helpers.savefig(fig, "dcostheta", outdir, fmts)



	if "delta-longest-trk-vs-length" in hists and hists["delta-longest-trk-vs-length"] and hists["delta-longest-trk-vs-length"].data.sum() > 0:
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

	trklen_by_pid_hists = {hname: hists[hname] for hname in hists if hname.startswith("trk-length-truepid_pdg=") if hname in hists}
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

	# compute efficiencies
	if "true-mu-energy-present" in hists and hists["true-mu-energy-present"]:
		eff_hists = {}
		labels = {}
		for eff_numerator in ("mu-trk-found", "mu-trk-found-completeness", "mu-trk-found-purity"):
			eff_hists[eff_numerator] = copy.copy(hists[eff_numerator])
			eff_hists[eff_numerator].data = hists[eff_numerator].data / hists["true-mu-energy-present"].data

			# generate labels for the plot legend
			suffix = eff_numerator.split("-")[-1]
			if suffix == "found":
				suffix = "full"
			else:
				suffix += " only"
			labels[eff_numerator] = suffix

		fig, ax = plotting_helpers.overlay_hists(eff_hists,
		                                         xaxis_label="True visible muon energy (MeV)",
		                                         yaxis_label="Muon identification efficiency",
		                                         hist_labels=labels)
		plotting_helpers.savefig(fig, "mu-trk-id-eff", outdir, fmts)

		fig, ax = plotting_helpers.overlay_hists({"": eff_hists["mu-trk-found"]},
		                                         xaxis_label="True visible muon energy (MeV)",
		                                         yaxis_label="Muon identification efficiency")
		plotting_helpers.savefig(fig, "mu-trk-id-eff-total", outdir, fmts)


	xlabels = {
		"muonVisE": "Visible true muon energy (MeV)",
		"truemuKE": "True muon KE (MeV)"
	}
	ylabels = {
		"mu-trk-mostEmu-completeness": r"Completeness of most $\mu$ track",
		"mu-trk-longest-completeness": r"Completeness of longest track",
		"mu-trk-purity": r"Frac. of longest track $E_{vis}$ from true $\mu$",
	}
	for hname in ["mu-trk-mostEmu-completeness-vs-muonVisE",
	              "mu-trk-mostEmu-completeness-vs-truemuKE",
	              "mu-trk-longest-completeness-vs-muonVisE",
	              "mu-trk-purity-vs-muonVisE"]:
		if hname not in hists or not hists[hname]: continue

		# column-normalize these
		h = hists[hname]
		for column in range(h.data.shape[0]):
			column_sum = h.data[column, :].sum()  # apparently the rows and columns are swapped <facepalm>
			if column_sum > 0:
				h.data[column, :] *= 1./column_sum

		if h.data.sum() == 0:
			continue

		fig = plt.figure()
		ax = fig.add_subplot()
		x, y = numpy.meshgrid(*h.bins)
		im = ax.pcolormesh(x, y, h.data.T, cmap="Reds", norm=matplotlib.colors.LogNorm())
		plt.colorbar(im)

		ax.set_xlabel(xlabels[hname.split("-")[-1]])
		ax.set_ylabel([ylabels[l] for l in ylabels if hname.startswith(l)][0])

		plotting_helpers.savefig(fig, hname, outdir, fmts)


