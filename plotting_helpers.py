import itertools
import numpy
import os.path

import larcv

LABELS_TO_IGNORE = ("Ghost", "Unknown")
SHAPE_LABELS = {getattr(larcv, s): s[6:] for s in dir(larcv.ShapeType_t) if s.startswith("kShape") and not any(s.endswith(l) for l in LABELS_TO_IGNORE) }


class Hist(object):
	def __init__(self, dim=1, norm=None, bins=None, data=None):
		self.dim = dim
		self.norm = norm

		self.bins = bins
		self.data = data


def req_vars_hist(fn, var_names):
	def _inner(data, hists):
		vars_missing = [v not in data for v in var_names]
		if len(vars_missing) > 0:
			print("Warning: var(s)", vars_missing, "missing from products. skipping plots from function:", fn)
			return

		return fn(data, hists)

	return _inner


def hist_aggregate(fn, hist_name, **hist_args):
	"""
	Decorator to manage creating/updating a NumPy histogram inside a collection.

	Pass it a function that accepts a single argument (incoming data)
	and returns an array of values to be histogrammed.

	Arguments to numpy.histogram() (e.g., 'bins', 'range') may be specified as keyword arguments.
	"""
	def _inner(vals, hist_collection):
		vals = fn(vals)
		hist, bins = numpy.histogram(vals, **hist_args)

		if hist_name in hist_collection:
			assert all(hist_collection[hist_name].bins == bins)
			hist_collection[hist_name].data += hist
		else:
			h = Hist()
			h.bins = bins
			h.data = hist
			hist_collection[hist_name] = h

	return _inner


def savefig(fig, name_stub, outdir, fmts):
	for fmt in fmts:
		fig.savefig(os.path.join(outdir, name_stub + "." + fmt))
