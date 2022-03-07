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
import track_functions

# will be filled in by the decorator as they're declared
SUMMARIZER_COLUMNS = {}

SHOWER_LABEL = [k for k, v in plotting_helpers.SHAPE_LABELS.items() if "Shower" in v][0]


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
        voxels = track_functions.track_voxel_coords(trk_index, input_data, reco_output)

        endpoints = track_functions.track_endpoints(trk_index, input_data, reco_output)

        # finally, figure out its direction at the end of the track.
        dists = track_functions.track_voxel_dists(trk_index, input_data, reco_output, voxels)
        dists_to_end = dists[numpy.where((voxels == endpoints[1]).all(axis=1))[0]]
        dir_vec = track_functions.track_end_dir(voxels, dists_to_end, endpoints)

        # for tracks that don't wind up being the muon candidate,
        # we'll also want to know their energy deposited.
        trk_visE = [input_data["input_data"][track_functions.track_voxel_indices(trk_index, input_data, reco_output)][:, -1].sum(),]

        tracks_out.append(numpy.concatenate([endpoints[0], endpoints[1], dir_vec, trk_visE]))

    ret = []
    if len(tracks_out):
        # reorder the 'ret' array by length in descending order (longest first)
        ret = numpy.row_stack(tracks_out)
        lengths = numpy.linalg.norm(ret[:, 6:9] - ret[:, 3:6], axis=1)
        ret = ret[lengths.argsort()[::-1]]  # from https://stackoverflow.com/a/9008147

    return ret
