"""
  PlotSS.py : Make some diagnostic plots of the semantic segmentation output

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  November 2020
"""

import argparse
import importlib
import itertools
import numpy
import os, os.path
import sys

KNOWN_PLOT_MODULES = {
	"ss": "Semantic segmentation",
	"ppn": "Point proposal",

	"shower": "Shower fragment grouping",
	"track":  "Track fragment grouping",

	"inter": "Neutrino interaction grouping",
}

ALLOWED_OUTPUT_FORMATS = [
	"png",
	"pdf"
]


def ParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input_file", "-i", required=True, action="append", default=[],
	                    help=".npz format file(s) containing reconstructed events.")
	parser.add_argument("--output_dir", "-o", required=True,
	                    help="Target directory to write output plots.")
	parser.add_argument("--overwrite", action="store_true", default=False,
	                    help="Overwrite output directory.")
	parser.add_argument("--start-evt", "-0", type=int, help="Event index to start processing at", default=0)
	parser.add_argument("--max-evts", "-n", type=int, help="Maximum number of events to process", default=-1)
	parser.add_argument("--img_format", "-f", action="append", choices=["eps", "png", "pdf"],
	                    help="Image format(s) to write output in.  Pass as many as desired.  Default uses 'png' and 'pdf'.")

	plots_args = parser.add_argument_group("plots", "Which plots to make")
	for module, description in KNOWN_PLOT_MODULES.items():
		module_args = parser.add_mutually_exclusive_group()
		module_args.add_argument("--disable_" + module,action="store_true", default=False,
		                         help="Don't make plots regarding " + description)

		module_args.add_argument("--only_" + module,action="store_true", default=False,
		                         help="ONLY make plots regarding " + description)

	args = parser.parse_args()

	# needs special treatment.
	# ('append' action doesn't work well with 'default' keyword---it *adds* to the default.
	#  so we set no default above, and if it comes back empty, we add the default here)
	if args.img_format is None:
		args.img_format = ["pdf", "png"]

	# manually handle any `--only_` flags here by reinterpreting them with `--disable` ones
	n_onlys = 0
	for arg in vars(args):
		argval = getattr(args, arg)
		if not arg.startswith("only_") or not argval: continue
		n_onlys += 1
		if n_onlys > 1:
			print("Specified multiple '--only' arguments.  Abort.")
			exit(1)
		for module in KNOWN_PLOT_MODULES:
			if module == arg.split("_")[-1]: continue
			setattr(args, "disable_" + module, True)

	return args


def Load(filenames, start_evt=0, max_evts=-1):
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

				data[k] = data[k][start_evt:max_evts]

		print("Loaded", len(data), "keys from file:", f)
		print("   keys =", [(k, type(data[k])) for k in data])
		yield data


if __name__ == "__main__":

	args = ParseArgs()

	if not os.path.isdir(args.output_dir):
		print("WARNING: output dir '%s' does not exist.  Attempting to create it..." % args.output_dir)
		os.mkdir(args.output_dir)
	elif not args.overwrite and len(os.listdir(args.output_dir)) > 0:
		print("ERROR: Output dir '%s' is not empty.  (Pass --overwrite if you want to overwrite it." % args.output_dir)
		sys.exit(1)

	modules = { m: importlib.import_module(m + "_plotting")
	            for m in KNOWN_PLOT_MODULES if not getattr(args, "disable_" + m) }

	hists = {}
	for data in Load(args.input_file, start_evt=args.start_evt, max_evts=args.max_evts):
		for mod_name, module in modules.items():
			if mod_name not in hists:
				hists[mod_name] = {}
			getattr(module, "BuildHists")(data, hists[mod_name])

	for mod_name, module in modules.items():
		outdir = os.path.join(args.output_dir, mod_name)
		os.makedirs(outdir, exist_ok=True)
		getattr(module, "PlotHists")(hists[mod_name], outdir, args.img_format)
