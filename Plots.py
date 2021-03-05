"""
  PlotSS.py : Make some diagnostic plots of the semantic segmentation output

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  November 2020
"""

import argparse
import importlib
import itertools
import numpy
import os.path

KNOWN_PLOT_MODULES = {
	"ss": "Semantic segmentation",
	"ppn": "Point proposal",

	"shower": "Shower fragment grouping",
	"track":  "Track fragment grouping",
}

ALLOWED_OUTPUT_FORMATS = [
	"png",
	"pdf"
]

for module in KNOWN_PLOT_MODULES:
	importlib.import_module(module + "_plotting")

def convert_to_geom_coords(values, metadata, evnums=()):
	# for coord in ("x", "y", "z"):
	#     print("min", coord, "=", getattr(metadata, "min_%s" % coord)())
	#     print ("voxel size", coord, "=",  getattr(metadata, "size_voxel_%s" % coord)())
	if len(evnums) > 0:
		values = itertools.compress(values, (i in evnums for i in range(len(values)) ))
	for ev in values:
		ev[:, 0] = ev[:, 0] * metadata.size_voxel_x() + metadata.min_x()
		ev[:, 1] = ev[:, 1] * metadata.size_voxel_y() + metadata.min_y()
		ev[:, 2] = ev[:, 2] * metadata.size_voxel_z() + metadata.min_z()


def ParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input_file", "-i", required=True, action="append", default=[],
	                    help=".npz format file(s) containing reconstructed events.")
	parser.add_argument("--output_dir", "-o", required=True,
	                    help="Target directory to write output plots.")
	parser.add_argument("--img_format", "-f", action="append", choices=["eps", "png", "pdf"],
	                    help="Image format(s) to write output in.  Pass as many as desired.  Default uses 'png' and 'pdf'.")

	plots_args = parser.add_argument_group("plots", "Which plots to make")
	for module, description in KNOWN_PLOT_MODULES.items():
		plots_args.add_argument("--disable_" + module, help="Don't make plots regarding " + description, default=False)

	parser.add_argument("--pixel_coords", help="Use pixel units for spatial coordinates rather than real detector geometry coordinates", default=False)

	args = parser.parse_args()

	# needs special treatment.
	# ('append' action doesn't work well with 'default' keyword---it *adds* to the default.
	#  so we set no default above, and if it comes back empty, we add the default here)
	if args.img_format is None:
		args.img_format = ["pdf", "png"]

	return args


def Load(filenames, pixel_coords=False):
	import plotting_helpers

	for f in filenames:
		with open(f, "rb"):
			datafile = numpy.load(f, allow_pickle=True)

			# these are usually dicts, so the actual type needs to be reconstructed
			data = {}
			for k in datafile:
				if k not in plotting_helpers.REQUIRED_VARS:
					continue

				print("Loading key:", k, type(datafile[k]))
				try:
					data[k] = datafile[k].item()
				except:
					data[k] = datafile[k]

			if not pixel_coords:
				if "metadata" in data:
					convert_to_geom_coords(data, data["metadata"][0])

		print("Loaded", len(data), "keys from file:", f)
		print("   keys =", [(k, type(data[k])) for k in data])
		yield data


if __name__ == "__main__":

	args = ParseArgs()
	modules = [ m for m in KNOWN_PLOT_MODULES if not getattr(args, "disable_" + m) ]

	hists = {}
	for data in Load(args.input_file, args.pixel_coords):
		for m in modules:
			hists[m] = {}
			getattr(m, "BuildHists")(data, hists[m])

	for m in modules:
		os.path.mkdir(os.path.join(args.output_dir, m))
		getattr(m, "PlotHists")(hists[m], args.output_dir, args.img_format)
