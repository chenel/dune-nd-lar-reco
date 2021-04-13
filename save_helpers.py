
import contextlib

import h5py

import summarize

# stand-in for contextlib.nullcontext for Py < 3.7
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass

def GetHDF5(filename, datasets, max_events=None):
    if filename is None:
        if hasattr(contextlib, "nullcontext"):
            return contextlib.nullcontext()
        else:
            return NullContextManager()

    # add some options to customize this list ... eventually
    summary_out = h5py.File(filename, "w")
    for dataset in datasets:
        # add 3 to the 'columns' for (run, subrun, event)
        shape = (3+summarize.SUMMARIZER_SHAPES[dataset][0], *summarize.SUMMARIZER_SHAPES[dataset][1:])
        summary_out.create_dataset(dataset,
                                   shape=(0, *shape),
                                   maxshape=(None, *shape))

    return summary_out

