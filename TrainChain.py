"""
  TrainChain.py : Train the LArCV full reco chain.

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  February 2021
"""

import argparse
import os.path
import yaml

import mlreco.main_funcs

def LoadConfig(filename, input_files, output_dir, random_seed=None, num_iterations=None, checkpoint_freq=None, debug=False, use_gpu=True):
	cfg = yaml.load(open(filename))

	if any(x in input_files for x in ('*', '?')):
		import glob
		input_files = glob.glob(input_files)
	cfg["iotool"]["dataset"]["data_keys"] = input_files
	cfg["trainval"]["log_dir"] = output_dir
	cfg["trainval"]["weight_prefix"] = os.path.join(output_dir, cfg["trainval"]["weight_prefix"] if "weight_prefix" in cfg["trainval"] else "")
	cfg["trainval"]["gpus"] = "0" if use_gpu else ""
	cfg["trainval"]["debug"] = debug

	if random_seed is not None:
		cfg["trainval"]["seed"] = random_seed
	if num_iterations is not None:
		cfg["trainval"]["iterations"] = num_iterations
	if checkpoint_freq is not None:
		cfg["trainval"]["checkpoint_step"] = checkpoint_freq

	cfg["trainval"]["train"] = True

	# pre-process configuration (checks + certain non-specified default settings)
	mlreco.main_funcs.process_config(cfg)

	return cfg

def ParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument("--config_file", "-c", required=True,
	                    help="YAML base configuration that will be augmented with other arguments.")
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
	training_group.add_argument("--disable_gpu", action="store_true", default=False,
	                            help="Disable the use of the GPU")

	return parser.parse_args()

if __name__ == "__main__":
	args = ParseArgs()

	cfg = LoadConfig(args.config_file,
	                 args.input_file,
	                 args.output_dir,
	                 random_seed=args.random_seed,
	                 num_iterations=args.num_iterations,
	                 checkpoint_freq=args.chkpt_freq,
	                 debug=args.debug,
	                 use_gpu=not args.disable_gpu)

	print("\nBegin training...")

	mlreco.main_funcs.train(cfg)
