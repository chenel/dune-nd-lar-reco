"""
  track_functions.py : Helper functions for working with tracks.

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  Feb. 2022
"""

import math
import numpy
import scipy.spatial

import plotting_helpers

# max distance between track end and voxels points considered for the track-end vector
ENDPOINT_DISTANCE = 20 # cm
MIN_COS_OPEN_ANGLE = 1 - math.cos(math.radians(30))

TRACK_LABEL = [k for k, v in plotting_helpers.SHAPE_LABELS.items() if "Track" in v][0]


def track_endpoints(trk_index, input_data, reco_output, voxels=None):
	if voxels is None:
		voxels = track_voxel_coords(trk_index, input_data, reco_output)

	# compute distances between all pairs of voxels in this track segment
	# call the ones furthest apart the track's "ends"
	dists = track_voxel_dists(trk_index, input_data, reco_output, voxels)
	maxes = numpy.argwhere(dists == numpy.max(dists))  # symmetric matrix, so always at least two, but may be others if multiple points are exactly same distance
	# print("maxes (note: %d voxels total):" % len(voxels), maxes)
	endpoint1_idx, endpoint2_idx = maxes[0]

	# now collect the two points treated as the endpoints.
	endpoints = voxels[(endpoint1_idx, endpoint2_idx), :3]
	track_ppn_points = reco_output["ppn_post"][:, :3][reco_output["ppn_post"][:, -1] == TRACK_LABEL]
	# print("endpoints:", endpoints)
	# print("track_ppn_points:", track_ppn_points)

	# if there are more than 2 "track" PPN points,
	# call the one closer to the most "track" PPN points the "beginning".
	# (risky, but will hopefully suffice until we have a better notion of the event vertex)
	# otherwise, assume the one at larger z is the end.
	if len(track_ppn_points) > 2:
		dists_to_ppn = numpy.sum(scipy.spatial.distance.cdist(endpoints, track_ppn_points), axis=1)
		start_idx = numpy.argmin(dists_to_ppn)
	else:
		start_idx = 0 if voxels[endpoint1_idx][2] < voxels[endpoint2_idx][2] else 1

	return voxels[(endpoint1_idx, endpoint2_idx) if start_idx == 0 else (endpoint2_idx, endpoint1_idx), :3]


def track_end_dir(voxels, dists_to_end, endpoints):
	"""
	Determine outgoing track direction vector for a group of voxels corresponding to a track.
	:param voxels:       array of array of (x,y,z) points corresponding to positions of voxels within the track group
	:param dists_to_end: NxN 2d array of distances between the voxels.  usually calculated with cdist from the "voxels" array
	:param endpoints:    length-6 array corresponding to the 3D points of the (start, end) track voxels in that order
	:return:
	"""

	# first work out the set of voxels close to the track endpoint.
	end = endpoints[1]
	v = end - endpoints[0]  # compute displacement vector from start to end
	v_norm = numpy.linalg.norm(v, axis=0)
	v /= v_norm
	if v_norm > ENDPOINT_DISTANCE:
		voxidxs_close_to_end = numpy.nonzero((0 < dists_to_end) & (dists_to_end < ENDPOINT_DISTANCE))[0]  # find all voxels in the group within fixed distance of the endpoint
	else:
		voxidxs_close_to_end = numpy.nonzero((0 < dists_to_end))[0]

	# compute the displacement vectors of those "close-to-end" voxels relative to the endpoint
	vox_displ_to_end = -(voxels[voxidxs_close_to_end] - end)  # subtract the endpoint from all the candidate voxels to get displacements
	norms = numpy.linalg.norm(vox_displ_to_end, axis=1)
	vox_displ_to_end = vox_displ_to_end / norms[..., None]  # and normalize those too

	# compute the opening angles of those displacement vectors
	# relative to the one from start to end of the track.
	# retain only those whose displ vec is within a given opening angle
	# relative to the (end-start) vector.
	cos_open_angles = numpy.sum(vox_displ_to_end * v, axis=1)  # take dot product to determine cos(opening angles)
	end_voxels = voxels[voxidxs_close_to_end[
		numpy.nonzero((0 <= cos_open_angles) & (cos_open_angles >= MIN_COS_OPEN_ANGLE))[0]]]  # keep voxels that are within fixed opening angle
	if len(end_voxels) <= 1:
		return v

	# now compute the principal axes of those voxels.
	# keep only the principal axis corresponding to the largest eigenvector
	# of the covariance matrix.
	# interpret that as the track-end direction.
	centered = end_voxels - numpy.sum(end_voxels, axis=0) / len(end_voxels)
	cov = numpy.cov(centered.T)  # covariance of the (recentered) voxel positions
	lmbda, e = numpy.linalg.eig(cov)  # eigenvalues & eigenvectors of the covariance matrix
	max_ev_idx = numpy.argmax(lmbda)
	dir_vec = numpy.sqrt(lmbda[max_ev_idx]) * e[:, max_ev_idx]

	if v.dot(dir_vec) < 0:  # if the eigenvector points the wrong way, flip it
		dir_vec *= -1

	return dir_vec


def track_voxel_coords(trk_index, input_data, reco_output):
	"""
	    Return an array of the 3-space coordinates of all the voxels in a particular reconstructed track.
	    :param trk_index:    which track you want the positions for
	    :param input_data:   the dict of "input data" products for this event
	    :param reco_output:  the dict of reco products for this event
	    :return:             2D array with 3 columns and number of rows equivalent to number of voxels
	    """

	voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(reco_output["track_fragments"]) if
	                                   reco_output["track_group_pred"][idx] == trk_index])
	return input_data["input_data"][track_voxel_indices(trk_index, input_data, reco_output)][:, :3]


def track_voxel_dists(trk_index, input_data, reco_output, voxels=None):
	"""
	    Compute the pairwise distances between all voxels in a particular reconstructed track.
	    Caches in reco_output for each event so distances for each track are only ever computed once.
	    :param trk_index:    which track you want the distances for
	    :param input_data:   the dict of "input data" products for this event
	    :param reco_output:  the dict of reco products for this event
	    :param voxels:       the voxel coordinates for the track.  if not supplied they will be retrieved using track_voxel_coords()
	    :return:             symmetrical 2D array where [idx1, idx2] indexes the distances between voxel indices idx1 and idx2
	    """

	product_name = "track_voxel_dists"

	if product_name in reco_output and trk_index in reco_output[product_name]:
		return reco_output[product_name][trk_index]

	if product_name not in reco_output:
		reco_output[product_name] = {}

	if voxels is None:
		voxels = track_voxel_coords(trk_index, input_data, reco_output)
#	print("voxels:", voxels)
	reco_output[product_name][trk_index] = scipy.spatial.distance.cdist(voxels, voxels)

	return reco_output[product_name][trk_index]


def track_voxel_indices(trk_index, input_data, reco_output):
	"""
	    Return an array the indices (within "input_data" array) of the voxels for a particular reconstructed track.
	    :param trk_index:    which track you want the positions for
	    :param input_data:   the dict of "input data" products for this event
	    :param reco_output:  the dict of reco products for this event
	    :return:             1D array of voxel indices
	    """

	return numpy.concatenate([frag for idx, frag in enumerate(reco_output["track_fragments"]) if
	                          reco_output["track_group_pred"][idx] == trk_index])

