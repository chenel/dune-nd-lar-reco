"""
  summarize.py : Functions to summarize the output of the reco.
                   Intended to be use as plugins in load_helpers.

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  April 2021
"""

import math
import sys

import h5py
import numpy
import scipy.spatial

import plotting_helpers

# will be filled in by the decorator as they're declared
SUMMARIZER_COLUMNS = {}

TRACK_LABEL = [k for k, v in plotting_helpers.SHAPE_LABELS.items() if "Track" in v][0]
SHOWER_LABEL = [k for k, v in plotting_helpers.SHAPE_LABELS.items() if "Shower" in v][0]

# max distance between track end and voxels points considered for the track-end vector
ENDPOINT_DISTANCE = 20 # cm
MIN_COS_OPEN_ANGLE = 1 - math.cos(math.radians(30))

# how far can a "shower" PPN point be from a shower fragment before it is no longer considered associated?
MAX_SHW_PPN_OFFSET = 1  # in cm. this is ~the diagonal distance of 2 corner-to-corner voxels

def SummarizerRunner(summarize_fns_names, datasets):
    """
    Create a summarizer function that will write output to a specific hd5.
    """
    summarizer_fns = {n: getattr(sys.modules[__name__], "summarize_" + n) for n in summarize_fns_names}

    def _inner(data, output):
        for name, fn in summarizer_fns.items():
            assert name in datasets, "Couldn't find dataset '%s' in collection" % name
            fn(data, output, datasets[name])

        return data, output

    return _inner


def summarizer(columns):
    """
    Decorator to register a summarizer function.
    The summarizer function should accept 2 arguments (input data and the reco output-in-progress)
    and it should return a numpy array with shape (N, *summarizer_shape)

    :param columns: Names for the columns of your data that this function will return.
    """
    def decorator(fn):
        assert fn.__name__.startswith("summarize_"), "Summarizer function names must begin with 'summarize_'"
        summarizer_name = fn.__name__[10:]
        SUMMARIZER_COLUMNS[summarizer_name] = columns
        shape = (len(columns),)

        def _inner(input_data, reco_output, dataset=None):
            # each idx corresponds to one event.  not all products may be filled for each event
            for idx in range(len(input_data[next(iter(input_data))])):
                inputs = {k: input_data[k][idx] for k in input_data}
                reco = {k: reco_output[k][idx] if idx < len(reco_output[k]) else [] for k in reco_output}
                ret = fn(inputs, reco)
                if len(ret) == 0:
                    continue

                assert ret.shape[1:] == shape, "Summarizer data shape " + str(ret.shape[1:]) + " is different from declared: " + str(shape)

                event_info = numpy.empty(shape=(len(ret), 3))
                event_info[:, :3] = inputs["event_base"][0]

                out = numpy.column_stack((event_info, ret))

                # add more rows as needed
                if isinstance(dataset, h5py.Dataset):
                    old_size = len(dataset)
                    dataset.resize(old_size + len(out), axis=0)
                    try:
                        dataset[old_size:old_size + len(out), :] = out
                    except ValueError as e:
                        print("h5py reports error trying to write to file:", e)
                        print("Object I tried to write was:", out)
                        print("Skipping to next event...")
                elif hasattr(dataset, "append"):
                    dataset.append(out)

        return _inner
    return decorator


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


@summarizer(columns=["shw_start_x", "shw_start_y", "shw_start_z",
                     "shw_dir_x", "shw_dir_y", "shw_dir_z",
                     "shw_visE"])
def summarize_showers(input_data, reco_output):
    """
     Summarize shower information.
     :param input_data:   List-of-dicts in mlreco3d-unwrapped format corresponding to parsed input for a single event.
     :param reco_output:  List-of-dicts in mlreco3d-unwrapped format corresponding to output of reconstruction for the same event.
     :return: array of shower information with columns as shown in decorator.  (also stored in hdf5 annotation.)
    """
    showers_out = []

    shw_indices = numpy.unique(reco_output["shower_group_pred"])
    for i, shw_index in enumerate(shw_indices):
        voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(reco_output["shower_fragments"]) if reco_output["shower_group_pred"][idx] == shw_index])
        voxels = input_data["input_data"][voxel_indices][:, :3]

        shw_ppn_points =  reco_output["ppn_post"][:, :3][reco_output["ppn_post"][:, -1] == SHOWER_LABEL]
        dists_to_ppn = numpy.sum(scipy.spatial.distance.cdist(voxels, shw_ppn_points), axis=1)
        dists_to_ppn_sel = dists_to_ppn[dists_to_ppn < MAX_SHW_PPN_OFFSET]

        # if there are no PPN points close enough, use the lowest z
        if len(dists_to_ppn_sel) == 0:
            closest_ppn_idx = numpy.argmin(voxels[:, 2])
        else:
            closest_ppn_idx = numpy.argmin(dists_to_ppn == numpy.min(dists_to_ppn_sel))

        shw_start = voxels[closest_ppn_idx]

        # compute the principal components of the shower voxels.
        # use the largest eigenvector to determine their direction
        centered = voxels - numpy.sum(voxels, axis=0) / len(voxels)
        cov = numpy.cov(centered.T)  # covariance of the (recentered) voxel positions
        lmbda, e = numpy.linalg.eig(cov)  # eigenvalues & eigenvectors of the covariance matrix
        max_ev_idx = numpy.argmax(lmbda)
        dir_vec = numpy.sqrt(lmbda[max_ev_idx]) * e[:, max_ev_idx]

        # the shower direction should be the direction where
        # the vectors from the PPN point to the hits
        # are parallel (rather than antiparallel) to it
        vox_displ_to_start = voxels - shw_start  # subtract the endpoint from all the candidate voxels to get displacements
        proj = numpy.sum(vox_displ_to_start * dir_vec, axis=1)
        if numpy.count_nonzero(proj < 0) > len(proj /2):  # if the eigenvector points the wrong way, flip it
            dir_vec *= -1
        dir_vec = dir_vec / numpy.linalg.norm(dir_vec)

        # finally, how much Evis was in this shower?
        shw_visE = [input_data["input_data"][voxel_indices][:, -1].sum(),]

        showers_out.append(numpy.concatenate([shw_start, dir_vec, shw_visE]))

    ret = []
    if len(showers_out) > 0:
        ret = numpy.row_stack(showers_out)
    return ret


@summarizer(columns=["trk_start_x", "trk_start_y", "trk_start_z",
                     "trk_end_x", "trk_end_y", "trk_end_z",
                     "trk_end_dir_x", "trk_end_dir_y", "trk_end_dir_z",
                     "trk_visE"])
def summarize_tracks(input_data, reco_output):
    """
      Summarize track information.
      :param input_data:   List-of-dicts in mlreco3d-unwrapped format corresponding to parsed input for a single event.
      :param reco_output:  List-of-dicts in mlreco3d-unwrapped format corresponding to output of reconstruction for the same event.
      :return: array of track information with columns as shown in decorator.  (also stored in hdf5 annotation.)
    """
    tracks_out = []

    track_indices = numpy.unique(reco_output["track_group_pred"])
    for i, trk_index in enumerate(track_indices):

        voxel_indices = numpy.concatenate([frag for idx, frag in enumerate(reco_output["track_fragments"]) if reco_output["track_group_pred"][idx] == trk_index])
        voxels = input_data["input_data"][voxel_indices][:, :3]

        # compute distances between all pairs of voxels in this track segment
        # call the ones furthest apart the track's "ends"
        dists = scipy.spatial.distance.cdist(voxels, voxels)
        maxes = numpy.argwhere(dists == numpy.max(dists))  # symmetric matrix, so always at least two, but may be others if multiple points are exactly same distance
        # print("maxes (note: %d voxels total):" % len(voxels), maxes)
        endpoint1_idx, endpoint2_idx = maxes[0]

        # now collect the two points treated as the endpoints.
        endpoints = voxels[(endpoint1_idx, endpoint2_idx), :3]
        track_ppn_points =  reco_output["ppn_post"][:, :3][reco_output["ppn_post"][:, -1] == TRACK_LABEL]
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

        endpoints = voxels[(endpoint1_idx, endpoint2_idx) if start_idx == 0 else (endpoint2_idx, endpoint1_idx), :3]

        # finally, figure out its direction at the end of the track.
        dir_vec = track_end_dir(voxels, dists[endpoint2_idx if start_idx == 0 else endpoint1_idx], endpoints)

        # for tracks that don't wind up being the muon candidate,
        # we'll also want to know their energy deposited.
        trk_visE = [input_data["input_data"][voxel_indices][:, -1].sum(),]

        tracks_out.append(numpy.concatenate([endpoints[0], endpoints[1], dir_vec, trk_visE]))

    ret = []
    if len(tracks_out):
        # reorder the 'ret' array by length in descending order (longest first)
        ret = numpy.row_stack(tracks_out)
        lengths = numpy.linalg.norm(ret[:, 6:9] - ret[:, 3:6], axis=1)
        ret = ret[lengths.argsort()[::-1]]  # from https://stackoverflow.com/a/9008147

    return ret
