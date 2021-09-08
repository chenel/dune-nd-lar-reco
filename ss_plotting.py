from matplotlib import pyplot as plt
import numpy
import os.path
import seaborn

import plotting_helpers


@plotting_helpers.hist_aggregate("vox-E", bins=numpy.logspace(numpy.log10(0.01), numpy.log10(10), 50))
def agg_voxE_reco(vals):
#	print(vals["segmentation"])
#	print(vals["input_data"][numpy.argmax(vals["segmentation"], axis=1) == 1][:, 4])
	reco_labels = numpy.argmax(vals["segmentation"], axis=1)
	return { "label=" + plotting_helpers.SHAPE_LABELS[label]: vals["input_data"][reco_labels == label][:, 4]
	         for label in plotting_helpers.SHAPE_LABELS }

@plotting_helpers.req_vars_hist(["input_data", "segment_label", "segmentation"])
def BuildHists(data, hists):
	# really I should reformat the data so that the event number is one of the columns...
	for evt_idx in range(len(data["segment_label"])):
		# idea: histogram the network's deduced label for classes of true labels.
		# we'll make a migration matrix out of them later
		len_true = len(data["segment_label"][evt_idx])
		len_reco = len(data["segmentation"][evt_idx])
		assert len_true == len_reco, \
		       "true and reco labels have different sizes!  true = %d, reco = %d" % (len_true, len_reco)

		true_labels = data["segment_label"][evt_idx][:, 4]                 # first 3 indices are the spatial position
		reco_labels = numpy.argmax(data["segmentation"][evt_idx], axis=1) # scores for each label.  find the index of the highest one

		for label_enum, label in plotting_helpers.SHAPE_LABELS.items():
			hist_name = "segmentation_" + label
#			print(true_labels == label_enum)
			hist, bins = numpy.histogram(reco_labels[true_labels == label_enum],
			                             bins=len(plotting_helpers.SHAPE_LABELS),
			                             range=(min(plotting_helpers.SHAPE_LABELS),
			                                    max(plotting_helpers.SHAPE_LABELS)+1))
#			print("bins:",bins)
			if hist_name in hists:
				assert all(hists[hist_name].bins == bins)
				hists[hist_name].data += hist
			else:
				h = plotting_helpers.Hist()
				h.bins = bins
				h.data = hist
				hists[hist_name] = h

	for evt_idx in range(len(data["segmentation"])):
		# first: number of tracks
		evt_data = { k: data[k][evt_idx] for k in data }
		for agg_fn in (agg_voxE_reco,):
			agg_fn(evt_data, hists)


# def HistPPNPerformance(data, hists):
# 	for evt_idx in range(len(data["raw_data"]["segment_label"])):


def PlotHists(hists, outdir, fmts):

	# make a migration matrix for the segmentation
	mig_mx = [None,] * len(plotting_helpers.SHAPE_LABELS)
	label_idx_by_name = dict((v, k) for k, v in plotting_helpers.SHAPE_LABELS.items())
	for histname, hist in hists.items():
		if not histname.startswith("segmentation_"):
			continue

		# add the row-normalized row into the matrix
		mig_mx[label_idx_by_name[histname.split("_")[1]]] = hist.data

	mig_mx = numpy.array(mig_mx)
	# import pprint
	# pprint.pprint(mig_mx)

	shape_labels_sorted = [plotting_helpers.SHAPE_LABELS[idx] for idx in sorted(plotting_helpers.SHAPE_LABELS)]
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

	plotting_helpers.savefig(plt, "ss_migration", outdir, fmts)

	vox_by_label_hists = {hname: hists[hname] for hname in hists if hname.startswith("vox-E_label=")}
	if len(vox_by_label_hists):
		hist_labels = {}
		for hname in vox_by_label_hists:
			hist_labels[hname] = hname.split("=")[-1]
		fig, ax = plotting_helpers.overlay_hists(vox_by_label_hists,
		                                         xaxis_label="Voxel energy (MeV)",
		                                         yaxis_label="Voxels",
		                                         hist_labels=hist_labels)
#		ax.set_xscale("log")
		plotting_helpers.savefig(fig, "reco-vox-E", outdir, fmts)

