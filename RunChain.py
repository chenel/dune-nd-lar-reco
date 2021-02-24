"""
  RunChain.py : Run the LArTPC reconstruction chain on Supera input.
                Store the output in a file.

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  February 2021
"""

import numpy

import load_helpers

if __name__ == "__main__":
	args = load_helpers.ParseArgs(load_helpers.RunType.INFERENCE)

	cfg = load_helpers.LoadConfig(filename=args.config_file,
	                              input_files=args.input_file,
	                              model_file=args.model_file,
	                              use_gpu=args.use_gpu)

	data = load_helpers.ProcessData(cfg)

	with open(args.output_file, "wb") as outf:
		numpy.savez(outf, **data)

	# this falls down because we also want to save the PPN maps, which are a list of arrays for each event
# 	with h5py.File(args.output_file, "w") as outf: #open(args.output_file, "wb") as outf:
# 		for k, list_of_vals in data.items():
# 			if len(list_of_vals) < 1:
# 				continue
# 			if all(len(v.shape) == 0 if hasattr(v, "shape") else False for v in list_of_vals):
# #				dtype = list_of_vals[0].dtype
# #				print(list_of_vals)
# 				list_of_vals = [numpy.array([[v,]]) for v in list_of_vals]
# #				print(list_of_vals)
#
# 			print(k, [v.shape if hasattr(v, "shape") else "%s of length %d" % (type(v), len(v)) for v in list_of_vals])
# 			frame = pandas.DataFrame(numpy.concatenate([numpy.concatenate([numpy.full(shape=(vals.shape[0], 1), fill_value=idx, dtype=numpy.int32), vals], axis=1)
# 			                                            for idx, vals in enumerate(list_of_vals)], axis=0))
# 			outf.create_dataset(name=k, data=frame)
#
# #		numpy.savez(outf, **data)
