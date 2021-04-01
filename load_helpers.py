"""
  load_helpers.py : Tools for loading up the lartpc_mlreco3d

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  February 2021
"""

import argparse
import enum
import numpy
import os.path
import sys
import yaml

import mlreco.main_funcs
from mlreco.utils.ppn import uresnet_ppn_type_point_selector


class RunType(enum.Enum):
	TRAIN = enum.auto()
	INFERENCE = enum.auto()


def PPNPostProcessing(data, output):
	# there's post-processing that needs to be done with PPN before we transform coordinates
	missing_products = [p in data for p in ("points", "mask_ppn2", "segmentation")]
	if any(missing_products):
		print("Warning: missing products", missing_products, " so can't do PPN post-processing")
		return []

	ppn = [None, ] * len(data["input_data"])
	for entry, input_data in enumerate(data["input_data"]):
		ppn[entry] = uresnet_ppn_type_point_selector(input_data,
		                                             output,
		                                             entry=entry,
		                                             score_threshold=0.5,
		                                             type_threshold=2)  # latter two args are from Laura D...
	output["ppn_post"] = ppn

def ProcessData(cfg, before=None, during=None, max_events=None):
	"""
	Process the data.
	:param cfg:     Full yaml configuration for mlreco3d
	:param before:  Function to be executed before run starts.  Should accept argument 'handlers'
	:param during:  Function to be executed every data event.   Should accept arguments 'input', 'output' and return a tuple (input, output) with (possibly) modified values
	:return:  dictionary containing lists of input & output values (one per event) with parser_name (input) or module_name (output) keys
	"""
	handlers = mlreco.main_funcs.prepare(cfg)
	if before is not None:
		before(handlers=handlers)

	key = next(iter(cfg["iotool"]["dataset"]["schema"]))

	print("Processing...")
	data = {}
	output = {}
	evt_counter = 0
	n_evts = len(handlers.data_io) * cfg["iotool"]["batch_size"]
	if max_events and 0 < max_events < n_evts:
		n_evts = (max_events // cfg["iotool"]["batch_size"]) * cfg["iotool"]["batch_size"]

	# the handlers.data_io_iter is an endless cycler.  we want to stop when we've made it through the dataset once
	def cycle(data_io):
		for x in data_io:
			yield x

	it = iter(cycle(handlers.data_io))
	while True:
		d = {key: []}
		o = {}
		try:
			d, o = handlers.trainer.forward(it)
		except StopIteration:
			break
		finally:
			lengths = set(len(d[k]) for k in d)
			assert len(lengths) == 1, "key lengths not all the same: " + str({k: len(v) for k, v in d.items()})
			evt_counter += lengths.pop()
			print("\rProcessed %d/%d" % (evt_counter, n_evts), "events...", end='')
			sys.stdout.flush()

		if during is not None:
			d, o = during(data=d, output=o)

		# these are all dicts of lists (each list has one entry per event)
		for this_batch, all_batches in (d, data), (o, output):
			for k in this_batch:
				if k not in all_batches:
					all_batches[k] = []
				all_batches[k] += this_batch[k]

		if max_events is not None and evt_counter >= max_events:
			break

	PPNPostProcessing(data, output)

	data.update(output)

	return data


def LoadConfig(filename, input_files, use_gpu=True, **kwargs):
	cfg = yaml.load(open(filename))

	if "iotool" in cfg and "dataset" in cfg["iotool"]:
		cfg["iotool"]["dataset"]["data_keys"] = input_files
	if "trainval" in cfg:
		cfg["trainval"]["gpus"] = "0" if use_gpu else ""

	if "trainval" in cfg and "train" in cfg["trainval"] and cfg["trainval"]["train"] is True:
		ConfigTrain(cfg, **kwargs)
	else:
		ConfigInference(cfg, **kwargs)

	# pre-process configuration (checks + certain non-specified default settings)
	mlreco.main_funcs.process_config(cfg)

	return cfg


def ConfigInference(cfg, model_file):
	cfg["trainval"]["model_path"] = model_file

	assert os.path.isfile(os.path.expandvars(model_file)), "Invalid model file path provided: " + model_file


def ConfigTrain(cfg, output_dir, random_seed=None, num_iterations=None, checkpoint_freq=None, debug=False):
	cfg["trainval"]["log_dir"] = output_dir
	cfg["trainval"]["weight_prefix"] = os.path.join(output_dir, cfg["trainval"]["weight_prefix"] if "weight_prefix" in cfg["trainval"] else "")
	cfg["trainval"]["debug"] = debug

	if random_seed is not None:
		cfg["trainval"]["seed"] = random_seed
	if num_iterations is not None:
		cfg["trainval"]["iterations"] = num_iterations
	if checkpoint_freq is not None:
		cfg["trainval"]["checkpoint_step"] = checkpoint_freq

	cfg["trainval"]["train"] = True



def ParseArgs(run_type):
	assert isinstance(run_type, RunType)

	parser = argparse.ArgumentParser()

	parser.add_argument("--config_file", "-c", required=True,
	                    help="YAML base configuration that will be augmented with other arguments.")
	
	if run_type is RunType.INFERENCE:
		parser.add_argument("--model_file", "-m", required=True,
		                    help="Path to Torch stored model weights file.")
		parser.add_argument("--input_file", "-i", required=True, action="append", default=[],
		                    help="Processed LArCV input file(s) to reconstruct.")
		parser.add_argument("--output_file", "-o", required=True,
		                    help="Target file to write output to.")

	elif run_type is RunType.TRAIN:
		parser.add_argument("--input_file", "-i", required=True, action="append", default=[],
		                    help="Supera input .root file(s) to use for training.")
		parser.add_argument("--output_dir", required=True,
		                    help="Directory to write output files into.")

		training_group = parser.add_argument_group("Training parameters (override the configuration)")
		training_group.add_argument("--random-seed", type=int,
		                            help="Random seed to use in training")
		training_group.add_argument("--num-iterations", "-n", type=int,
		                            help="Number of training iterations")
		training_group.add_argument("--chkpt-freq",  type=int,
		                            help="Frequency (in iterations) to write the weights to the log")

		training_group.add_argument("--debug", "-d", action="store_true",
		                            help="Enable debug output")

	parser.add_argument("-n", "--max_events", type=int,
	                    help="Maximum number of events to process.")

	parser.add_argument("--use_gpu", action="store_true", default=True,
	                    help="Use GPU.  Default: %(default)s")

	return parser.parse_args()
