"""
  summarize.py : Functions to summarize the output of the reco.
                   Intended to be use as plugins in load_helpers.

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  April 2021
"""

import sys

import numpy
import scipy.spatial

import plotting_helpers

# will be filled in by the decorator as they're declared
SUMMARIZER_SHAPES = {}

TRACK_LABEL = [k for k, v in plotting_helpers.SHAPE_LABELS.items() if "Track" in v][0]


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


def summarizer(shape):
    """
    Decorator to register a summarizer function.
    The summarizer function should accept 2 arguments (input data and the reco output-in-progress)
    and it should return a numpy array with shape (N, *summarizer_shape)

    :param shape: The numpy shape of your data, excluding the rows (simplest case: number of columns).
                  Describe it in a comment!
    """
    def decorator(fn):
        assert fn.__name__.startswith("summarize_"), "Summarizer function names must begin with 'summarize_'"
        summarizer_name = fn.__name__[10:]
        SUMMARIZER_SHAPES[summarizer_name] = shape

        def _inner(input_data, reco_output, dataset):
            # each idx corresponds to one event.  not all products may be filled for each event
            for idx in range(len(input_data[next(iter(input_data))])):
                inputs = {k: input_data[k][idx] for k in input_data}
                reco = {k: reco_output[k][idx] if idx < len(reco_output[k]) else [] for k in reco_output}
                ret = fn(inputs, reco)
                if len(ret) == 0:
                    continue

                assert ret.shape[1:] == shape, "Summarizer data shape " + str(ret.shape[1:]) + " is different from declared: " + str(shape)

                event_info = numpy.empty(shape=(len(ret), 3))
                event_info[:, :3] = input_data["event_base"][0][0]

                out = numpy.column_stack((event_info, ret))

                # add more rows as needed
                old_size = len(dataset)
                dataset.resize(old_size + len(out), axis=0)
                dataset[old_size:old_size + len(out), :] = out

        return _inner
    return decorator


@summarizer(shape=(6,))   # 6 columns: [trk_start_x, trk_start_y, trk_start_z, trk_end_x, trk_end_y, trk_end_z]
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
        row, col = maxes[0]

        # now collect the two points treated as the endpoints.
        # call the one closer to the most "track" PPN points the "beginning".
        # (risky, but will hopefully suffice until we have a better notion of the event vertex)
        endpoints = voxels[(row,col), :3]
        track_ppn_points =  reco_output["ppn_post"][:, :3][reco_output["ppn_post"][:, -1] == TRACK_LABEL]
        # print("endpoints:", endpoints)
        # print("track_ppn_points:", track_ppn_points)
        dists_to_ppn = numpy.sum(scipy.spatial.distance.cdist(endpoints, track_ppn_points), axis=1)
        min_idx = numpy.argmin(dists_to_ppn)

        tracks_out.append(numpy.concatenate([endpoints[min_idx], endpoints[min_idx ^ 1]]))

    return numpy.row_stack(tracks_out) if len(tracks_out) > 0 else []
