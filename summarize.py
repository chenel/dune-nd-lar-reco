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

# max distance between track end and voxels points considered for the track-end vector
ENDPOINT_DISTANCE = 20 # cm
MIN_COS_OPEN_ANGLE = 1 - math.cos(math.radians(30))


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
                if isinstance(dataset, h5py.File):
                    old_size = len(dataset)
                    dataset.resize(old_size + len(out), axis=0)
                    dataset[old_size:old_size + len(out), :] = out
                elif hasattr(dataset, "append"):
                    dataset.append(out)

        return _inner
    return decorator


def track_end_dir(voxels, dists_to_end, endpoints):
    end = endpoints[1]
    print("start voxel:", endpoints[0])
    print("end voxel:", end)
    v = end - endpoints[0]  # compute displacement vector from start to end
    v_norm = numpy.linalg.norm(v, axis=0)
    print("distance between them:", v_norm)
    v /= v_norm
    # with numpy.printoptions(threshold=numpy.inf):
    #     print("dists_to_end:", dists_to_end)
    if v_norm > ENDPOINT_DISTANCE:
        # print("indices of voxels close to end:", numpy.nonzero(dists_to_end < ENDPOINT_DISTANCE))
        voxidxs_close_to_end = numpy.nonzero((0 < dists_to_end) & (dists_to_end < ENDPOINT_DISTANCE))[0]  # find all voxels in the group within fixed distance of the endpoint
    else:
        voxidxs_close_to_end = numpy.nonzero((0 < dists_to_end))[0]
    print("voxels_close_to_end:", voxels[voxidxs_close_to_end])
    vox_displ_to_end = -(voxels[voxidxs_close_to_end] - end)  # subtract the endpoint from all the candidate voxels to get displacements
    print("vox displ vectors:", vox_displ_to_end.shape, vox_displ_to_end)
    norms = numpy.linalg.norm(vox_displ_to_end, axis=1)
    print("displ vec lengths:", norms.shape, norms)
    vox_displ_to_end = vox_displ_to_end / norms[..., None]  # and normalize those too
    print("vox_displ_to_end:", vox_displ_to_end)
    cos_open_angles = numpy.sum(vox_displ_to_end * v, axis=1)  # take dot product to determine cos(opening angles)
    print("cos_open_angles:", cos_open_angles)
    # print("cos_open_angles >= 0:", 0 <= cos_open_angles)
    #        print("selected indices:", type(numpy.nonzero((0 <= cos_open_angles) & (cos_open_angles <= COS_OPEN_ANGLE))[0]))
    # print("types:", type(voxels), type(voxidxs_close_to_end))
    end_voxels = voxels[voxidxs_close_to_end[
        numpy.nonzero((0 <= cos_open_angles) & (cos_open_angles >= MIN_COS_OPEN_ANGLE))[
            0]]]  # keep voxels that are within fixed opening angle
    if len(end_voxels) <= 1:
        print('warning: track has insuficient endpoint voxels.')
        print("  --> using (end - start) to determine track dir")
        return v
    print("end_voxels:", len(end_voxels), end_voxels)

    centered = end_voxels - numpy.sum(end_voxels, axis=0) / len(end_voxels)
    # print("centered:", centered)
    cov = numpy.cov(centered.T)  # covariance of the (recentered) voxel positions
    # print("cov:", cov)
    lmbda, e = numpy.linalg.eig(cov)  # eigenvalues & eigenvectors of the covariance matrix
    # print("eigenvalues:", lmbda)
    # print("eigenvctors:", e)
    max_ev_idx = numpy.argmax(lmbda)
    dir_vec = numpy.sqrt(lmbda[max_ev_idx]) * e[:, max_ev_idx]

    if v.dot(dir_vec) < 0:  # if the eigenvector points the wrong way, flip it
        dir_vec *= -1

    print("dir vec:", dir_vec)
    return dir_vec


@summarizer(columns=["trk_start_x", "trk_start_y", "trk_start_z",
                     "trk_end_x", "trk_end_y", "trk_end_z",
                     "trk_end_dir_x", "trk_end_dir_y", "trk_end_dir_z"])
def summarize_tracks(input_data, reco_output):
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
        # for the moment, just take all the points that are close to the end
        # and don't deviate from the beginning-end trajectory by too much.
        print("event info:", input_data["event_base"][0])
        print("track index:", trk_index)
        # print("start, end vox indices:", (endpoint1_idx, endpoint2_idx) if start_idx == 0 else (endpoint2_idx, endpoint1_idx))
#        print("row idx in 'dist' matrix corresponding to distance to endpoint:", numpy.argwhere(voxels == endpoints[1]))
        dir_vec = track_end_dir(voxels, dists[endpoint2_idx if start_idx == 0 else endpoint1_idx], endpoints)
        tracks_out.append(numpy.concatenate([endpoints[0], endpoints[1], dir_vec]))

    return numpy.row_stack(tracks_out) if len(tracks_out) > 0 else []
