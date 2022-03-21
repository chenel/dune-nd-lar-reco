import functools
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

		if self.norm not in ("density", "unit") and not isinstance(self.norm, (int, float)):
			raise ValueError("Unknown normalization: '%s'" % self.norm)

		self.data = self.data / numpy.array(numpy.diff(self.bins), float)
		if self.norm == "unit":
			norm_factor = self.data.sum()
		elif isinstance(self.norm, (int, float)):
			assert self.norm > 0
			norm_factor = self.data.sum() * self.norm
		else:
			norm_factor = 1.0

		if norm_factor != 1.0:
			self.data = self.data / norm_factor

		self.norm = None  # don't allow it to be done multiple times

	def StdDev(self, bin_range=(0, -1)):
		# constrain to range if requested
		data = self.data[bin_range[0]:bin_range[1]]
		bins = self.bins[bin_range[0]:(bin_range[1]+1 if bin_range[1] >= 0 else -1)]

		print("bins:", bins)
		print("data:", data)

		bin_ctrs = bins[:-1] + (bins[1:] - bins[:-1]) * 0.5
		N = data.sum()
		mean = (data * bin_ctrs).sum() / N
		return numpy.sqrt(((bin_ctrs - mean) ** 2 * data).sum() / (N - 1))

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

	The function may also return a dictionary of arrays (per above)
	to indicate that the histograms are of the same value but split into subsamples.

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
					try:
						hist, bins = numpy.histogram(vs, **hist_args)
					except:
						print("Exception encountered inside aggregator function:", fn)
						raise
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


def overlay_hists(hists, xaxis_label=None, yaxis_label="Events", hist_labels={}, stack=[], **kwargs):
	"""
	Draw histograms overlaid on the same axis.

	:param hists:  dict of {name: plotting_helpers.Hist} with histograms to overlay
	:param xaxis_label: x-axis label to use.
	:param xaxis_label: y-axis label to use.
	:param hist_labels: legend labels for histograms in {name: label} format
	:param stack: [list of [list of old names]] specifying histograms from 'hists' to add together (order matters: bottom to top).
	:param kwargs: other arguments to pass to matplotlib.pyplot.axis.step()
	:return: matplotlib.pyplot.Figure and Axis objects in case you want to modify them further

	"""
	fig, ax = plt.subplots()

	# first build stacks where requested
	flat_stack = list(itertools.chain(*stack))
	if len(flat_stack) > 0:
		dups = set(l for l in flat_stack if flat_stack.count(l) > 1)
		assert len(dups) == 0, "Histogram(s) %s were included in multiple stacks, but it can only go in one" % dups
	bottoms = {}
	for hist_name_list in stack:
		assert isinstance(hist_name_list, list), "overlay_hists stack: Expecting a list of lists but found instead type %s.  Is your list too shallow?" % type(hist_name_list)
		# if there aren't at least two, no point in stacking
		if len(hist_name_list) < 2:
			continue
		unknown_hists = [h for h in hist_name_list if h not in hists]
		assert len(unknown_hists) == 0, "Unknown histogram(s) passed to overlay_hists 'stack' keyword: " + str(unknown_hists)
		assert functools.reduce(lambda a, b: numpy.array_equal(a, b), (hists[h].bins for h in hist_name_list)), "Histograms to stack must have same bin edges..."

		# swap out the previous histograms with versions that have the running sum of the stack
		running_sum = numpy.zeros_like(hists[hist_name_list[0]].data)
		for i, hname in enumerate(hist_name_list):
			bottoms[hname] = numpy.copy(running_sum)
			running_sum = running_sum + hists[hname].data

	for hname, h in hists.items():
		this_kwargs = {}
		for key, val in kwargs.items():
			if hasattr(val, "__getitem__"):
				if hname in val:
					this_kwargs[key] = val[hname]
			else:
				this_kwargs[key] = val
		if hname in flat_stack:
			ax.bar(x=h.bins[:-1] + numpy.diff(h.bins)*0.5, height=h.data, width=numpy.diff(h.bins), bottom=bottoms[hname], alpha=0.5,
			       label=hist_labels[hname] if hname in hist_labels else None, **this_kwargs)
		else:
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
