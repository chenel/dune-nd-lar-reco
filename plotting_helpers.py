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

	def Normalize(self):
		if self.norm is None:
			return

		if self.norm != "density":
			raise ValueError("Unknown normalization: '%s'" % self.norm)

		self.data = self.data / numpy.array(numpy.diff(self.bins), float)
		self.norm = None  # don't allow it to be done multiple times

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


def hist_aggregate(hist_name, hist_dim=1, norm=None, **hist_args):
	"""
	Decorator to manage creating/updating a NumPy histogram inside a collection.

	Pass it a function that accepts a single argument (incoming data)
	and returns an array of values to be histogrammed (if 1D histogram)
	or two arrays of values to be histogrammed (if 2D).

	By default creates 1D histogram.  If 2D is desired pass hist_dim=2.

	Arguments to numpy.histogram() (e.g., 'bins', 'range') may be specified as keyword arguments.
	"""
	def decorator(fn):
		def _inner(vals, hist_collection):
			vals = fn(vals)

			# we want to be able to handle dicts
			# for the case where multiple instances of the "same" hist
			# separated by a selection (the dict key) are returned.
			# if that *isn't* what happened, turn it into a dict with a single key.
			if not isinstance(vals, dict):
				vals = {None: vals}

			for subsample, vs in vals.items():
				full_hist_name = "%s_%s" % (hist_name, subsample) if subsample else hist_name
				if hist_dim == 1:
					hist, bins = numpy.histogram(vs, **hist_args)
				elif hist_dim == 2:
					if len(vs) == 0:
						return
					hist, binsx, binsy = numpy.histogram2d(*vs, **hist_args)
					bins = (binsx, binsy)
				else:
					raise ValueError("Unsupported histogram dimension: " + str(hist_dim))

				if full_hist_name in hist_collection:
					h = hist_collection[full_hist_name]
					if h.dim == 1:
						assert all(h.bins == bins)
					elif h.dim == 2:
						assert all([numpy.array_equal(h.bins[i], bins[i]) for i in range(len(h.bins))])
					hist_collection[full_hist_name].data += hist
				else:
					h = Hist(dim=hist_dim, bins=bins, data=hist, norm=norm)
					hist_collection[full_hist_name] = h

		return _inner

	return decorator


def overlay_hists(hists, xaxis_label=None, yaxis_label="Events", hist_labels={}, **kwargs):
	fig, ax = plt.subplots()
	for hname, h in hists.items():
		this_kwargs = {}
		for key, val in kwargs.items():
			if hasattr(val, "__getitem__"):
				if hname in val:
					this_kwargs[key] = val[hname]
			else:
				this_kwargs[key] = val
		ax.step(x=h.bins[:-1], y=h.data, where="post", label=hist_labels[hname] if hname in hist_labels else None, **this_kwargs)
	#			ax.bar(h.bins[:-1], h.data, width=numpy.diff(h.bins), fill=False, label="True" if "true" in hname else "Reco")
		ax.margins(x=0)  # eliminate the whitespace between the x-axis and the first bin
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
