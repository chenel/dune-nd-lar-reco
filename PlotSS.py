"""
  PlotSS.py : Make some diagnostic plots of the semantic segmentation output

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  November 2020
"""

import argparse
import collections
import numpy
import matplotlib.pyplot as plt
import os.path
import seaborn

from larcv import larcv

SHAPE_LABELS = {getattr(larcv, s): s[6:] for s in dir(larcv.ShapeType_t) if s.startswith("kShape")}

class Hist:
	def __init__(self, dim=1, norm=None):
		self.dim = dim
		self.norm = norm

		self.bins = None
		self.data = None

def ParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input_file", "-i", required=True, action="append", default=[],
	                    help=".npz format file(s) containing reconstructed events.")
	parser.add_argument("--output_dir", "-o", required=True,
	                    help="Target directory to write output plots.")

	return parser.parse_args()

def HistSSPerformance(data, hists):
	# really I should reformat the data so that the event number is one of the columns...
	for evt_idx in range(len(data["raw_data"]["segment_label"])):
		# idea: histogram the network's deduced label for classes of true labels.
		# we'll make a migration matrix out of them later
		len_true = len(data["raw_data"]["segment_label"][evt_idx])
		len_reco = len(data["ss_output"]["segmentation"][evt_idx])
		assert len_true == len_reco, \
		       "true and reco labels have different sizes!  true = %d, reco = %d" % (len_true, len_reco)

		true_labels = data["raw_data"]["segment_label"][evt_idx][:, 4]                 # first 3 indices are the spatial position
		reco_labels = numpy.argmax(data["ss_output"]["segmentation"][evt_idx], axis=1) # scores for each label.  find the index of the highest one

		for label_enum, label in SHAPE_LABELS.items():
			hist_name = "segmentation_" + label
#			print(true_labels == label_enum)
			hist, bins = numpy.histogram(reco_labels[true_labels == label_enum], bins=len(SHAPE_LABELS), range=(min(SHAPE_LABELS), max(SHAPE_LABELS)+1))
#			print("bins:",bins)
			if hist_name in hists:
				assert all(hists[hist_name].bins == bins)
				hists[hist_name].data += hist
			else:
				h = Hist()
				h.bins = bins
				h.data = hist
				hists[hist_name] = h

def PlotHists(hists, outdir):

	# make a migration matrix for the segmentation
	mig_mx = [None,] * len(SHAPE_LABELS)
	label_idx_by_name = dict((v, k) for k, v in SHAPE_LABELS.items())
	for histname, hist in hists.items():
		if not histname.startswith("segmentation_"):
			continue

		# add the row-normalized row into the matrix
		mig_mx[label_idx_by_name[histname.split("_")[1]]] = hist.data

	mig_mx = numpy.array(mig_mx)
	import pprint
	pprint.pprint(mig_mx)

	shape_labels_sorted = [SHAPE_LABELS[idx] for idx in sorted(SHAPE_LABELS)]
	bins = hists[next(iter(hists))].bins[:-1]
	# for axis_name in "x", "y":
	# 	# set the axis labels to the correct strings
	# 	getattr(plt, "%sticks" % axis_name)(bins, shape_labels_sorted)
	# plt.xlabel("Predicted category")
	# plt.ylabel("True category")
	# plt.colorbar()
	# plt.savefig(os.path.join(outdir, "ss_migration.png"))

	fig, ax = plt.subplots(figsize=(10,7))
	row_sums = numpy.sum(mig_mx, axis=1)
	seaborn.heatmap(mig_mx / row_sums[:, None], annot=True,
	                xticklabels=shape_labels_sorted, yticklabels=[shape_labels_sorted[idx] + "\n(%d)" % row_sums[idx] for idx in range(len(row_sums))],
	                fmt=".2g", cmap="Blues")
	ax.set_xlabel("Predicted label", size="large")
	ax.xaxis.set_ticks_position("top")
	ax.set_ylabel("True label", size="large")
	ax.tick_params(axis="y", rotation=0)
	plt.savefig(os.path.join(outdir, "ss_migration.png"))


def Load(filenames):
	for f in filenames:
		with open(f, "rb"):
			data = numpy.load(f, allow_pickle=True)

			# these are usually dicts, so the actual type needs to be reconstructed
			data = {k: data[k].item() for k in data}
		assert all(k in data for k in ("raw_data", "ss_output"))
		print("Loaded", len(data), "keys from file:", f)
		print("   keys =", [(k, type(data[k])) for k in data])
		yield data

if __name__ == "__main__":

	args = ParseArgs()

	hists = {}
	for data in Load(args.input_file):
		HistSSPerformance(data, hists)

	PlotHists(hists, args.output_dir)
