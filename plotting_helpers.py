import itertools
import numpy
import os.path
from matplotlib import pyplot as plt

from larcv import larcv

LABELS_TO_IGNORE = ("Ghost", "Unknown")
SHAPE_LABELS = {getattr(larcv, s): s[6:] for s in dir(larcv.ShapeType_t) if s.startswith("kShape") and not any(s.endswith(l) for l in LABELS_TO_IGNORE) }

# this will be updated by req_vars_hist() below
REQUIRED_VARS = set()


class Hist:
	def __init__(self, dim=1, norm=None, bins=None, data=None):
		self.dim = dim
		self.norm = norm

		self.bins = bins
		self.data = data


def req_vars_hist(var_names):
	global REQUIRED_VARS
	REQUIRED_VARS.update(var_names)

	def decorator(fn):
		def _inner(data, hists):
			vars_missing = [v for v in var_names if v not in data]
			if len(vars_missing) > 0:
				print("Warning: var(s)", vars_missing, "missing from products.)")
				print("Known products:", data.keys())
				print("Skipping plots from function:", fn)
				return

			return fn(data, hists)
		return _inner
	return decorator


def hist_aggregate(hist_name, **hist_args):
	"""
	Decorator to manage creating/updating a NumPy histogram inside a collection.

	Pass it a function that accepts a single argument (incoming data)
	and returns an array of values to be histogrammed.

	Arguments to numpy.histogram() (e.g., 'bins', 'range') may be specified as keyword arguments.
	"""
	def decorator(fn):
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

	return decorator


def overlay_hists(hists, xaxis_label=None, yaxis_label="Events", hist_labels={}):
	fig, ax = plt.subplots()
	for hname, h in hists.items():
		ax.step(x=h.bins[:-1], y=h.data, where="post", label=hist_labels[hname] if hname in hist_labels else None)
	#			ax.bar(h.bins[:-1], h.data, width=numpy.diff(h.bins), fill=False, label="True" if "true" in hname else "Reco")
	for axname in ("x", "y"):
		axlabel = locals()[axname + "axis_label"]
		if axlabel:
			getattr(ax, "set_%slabel" % axname)(axlabel)
	if len(hist_labels) > 0:
		ax.legend()

	return fig, ax


def savefig(fig, name_stub, outdir, fmts):
	for fmt in fmts:
		fig.savefig(os.path.join(outdir, name_stub + "." + fmt))
