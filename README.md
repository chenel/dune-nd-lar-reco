# `dune-nd-lar-reco`: Tools for using [lartpc_mlreco3d](https://github.com/DeepLearnPhysics/lartpc_mlreco3d) with DUNE ND-LAr sim

----------------------------------------------------------------------

  *Original author*:               J. Wolcott  <[jwolcott@fnal.gov](mailto:jwolcott@fnal.gov)>
  
  *Last update to this document*:  August 2021

----------------------------------------------------------------------         

-------------------

# Overview

This repository contains tools for applying and evaluating the [DeepLearnPhysics/lartpc_mlreco3d](https://github.com/DeepLearnPhysics/lartpc_mlreco3d) deep learning reconstruction pipeline on DUNE ND-LAr simulation.

At present, the application is to the output of edep-sim, i.e., the GEANT4 simulation of the detector.
The output of full `larnd-sim` simulation will hopefully be connected to the pipeline in the near future.

There are essentially 3 steps the tools here facilitate:
* Training the algorithm weights using DUNE ND-LAr edep-sim ([Section 1](#1-training))
* Evaluating the algorithms on edep-sim ([Section 2](#2-evaluation))
* Making summary plots from the output of the previous step ([Section 3](#3-making-plots))

There is also an inference Jupyter notebook that serves as a rudimentary event display ([Section 4](#4-event-display)).

[Section 0](#0-prerequisites) contains notes about software setup.

# 0. Prerequisites 

## Software environment

### `lartpc_mlreco3d`

`lartpc_mlreco3d` depends on a number of Python libraries that can be a pain to install.
The DeepLearnPhysics group maintains a collection of Docker containers that have the requisite software already available inside them: https://hub.docker.com/r/deeplearnphysics/larcv2
Choose the latest one that is compatible with the version of CUDA necessary for your GPU.

**If you plan to only run the reconstruction, and not make `.larcv.root` files yourself, you can likely use one of these containers without building `larcv2` yourself.**

### This package (`dune-nd-lar-reco`)

Always run the scripts in this package using the `dune-nd-lar-reco` dir as the working directory (`cd /path/to/dune-nd-lar-reco`).  Otherwise you'll get `Module not found` errors when they try to import their dependencies.

(`todo`: add `__init__.py` and make this an actual importable package)

## Input files: `larcv2`

**You probably won't need the instructions in this section unless you need to make `larcv2` files yourself.**  (See the Docker containers above.)  The instructions are still retained here for reference, just in case.

### Build & setup
In order to be understood by mlreco3d, input files must be in `LArCV` format (ROOT `TTree`s with special objects in them, stored in a `.root` file).  The appropriate `.larcv.root` files are produced using the [DeepLearnPhysics/larcv2](https://github.com/DeepLearnPhysics/larcv2) package.

However, because `larcv2` was originally written to accept the fully simulated output from the `larsoft` framework, we use a fork:

https://github.com/chenel/larcv2/tree/edepsim-formattruth

LArCV is relatively straightforward to build.  This fork requires recent builds of ROOT 6, GEANT4, and `edep-sim`, all of which should be accessible in the `$PATH` (use their setup scripts).  It also requires Python with the `numpy` library.  (This should be the same as the `numpy` you're going to use with mlreco3d in the subsequent steps.)  Once they are:
```shell
cd /path/to/larcv2
source configure.sh
make
```
Then, subsequently, repeat the `source configure.sh` step in order to set up `larcv2` for use.

### Creating files from `edep-sim`

(This section may be skipped if input files in `larcv.root` format are already available.  The DUNE ND Production effort intends to produce `.larcv.root` files as part of its workflow, so check there first!)

The software necessary for the `edep-sim`-to-`larcv2` process is kept in the [`larcv/app/Supera`](https://github.com/chenel/larcv2/tree/edepsim-formattruth/larcv/app/Supera/) subdirectory of the `larcv2` fork mentioned above. 
There is a driver script, [`run_supera.py`](https://github.com/chenel/larcv2/tree/edepsim-formattruth/larcv/app/Supera/run_supera.py).  Usage:
```shell
# could also be `python3` depending on your installation
python supera.py <config.cfg> edep_sim.root [edep_sim2.root ...]
```

Configuration files (used in place of the `<config.cfg>` placeholder) configure the operation.  You'll probably want to use `supera-ndlar.cfg`, which is set up for the ND-LAr geometry.

The output will be written to the file specified by the `OutFileName` key in the configuration file.

### Batch processing on the grid

`Supera` has been integrated into the [ND Production](https://github.com/DUNE/ND_Production) workflow.  This means that `.larcv.root` files should be part of the official output.

However, if you need to batch-produce `.larcv.root` files from `edep-sim` output, the [`run_supera_grid.sh`](https://github.com/chenel/dune-nd-lar-reco/blob/main/grid/run_supera_grid.sh) script in the `grid/` subdirectory of *this* package can be used (along with an installation of `larcv2`) on Fermigid to produce them.

# 1. Training

### Context
**Before proceeding here you should at minimum have a basic understanding of the `mlreco3d` algorithms, how to configure them using `.yaml` files, and what to expect when you train them.**  The DeepLearnPhysics group maintains a set of [tutorials](http://deeplearnphysics.org/lartpc_mlreco3d_tutorials/) that you can use to familiarize yourself with the workflow.   (The [SLAC group's wiki](https://nu-wiki.herokuapp.com/chain) also has useful information on the reconstruction chain, though I believe it's in the process of being deprecated.)

### Hardware requirements

At present, the later stages of the chain (the GNNs) tend to be relatively memory-hungry and fairly slow to train.  (An update to mlreco3d that is slated to be released soon should improve this performance substantially, however.)  Because of this, you will likely find it difficult to train those models using a GPU with less than 20-25 GB of video RAM.

The UResNet+PPN network does not suffer from these issues.

### Running the training
Though training can be done straightforwardly in a Jupyter notebook (as in the tutorials), I find it convenient to have a driver script that can be run from the command line with switches to easily override various configuration options.  The `TrainChain.py` script in this package serves that purpose.   Pass the `--help` flag to it to see what options it supports.

Example invocation, specifying config, input `.larcv.root` file, directory for output:
```shell
python3 TrainChain.py -c config.train.inter-gnn.yaml \
                      -i $data/dune/nd/nd-lar-reco//supera/geom-20210405-pileup/FHC.1000001.overlay.larcv.root \
                      --output_dir $data/dune/nd/nd-lar-reco/train/intergnn-120evs-1000Kits-batch16
```

### Configuration files

There are a number of reasonably current configurations for training various subsets of the models in the [`configs/`](https://github.com/chenel/dune-nd-lar-reco/tree/main/configs) subdirectory of *this* repository.  At present, they are trained sequentially:
* UResNet+PPN first ([`config.train.uresnet+ppn.yaml`](https://github.com/chenel/dune-nd-lar-reco/blob/main/configs/config.train.uresnet%2Bppn.yaml))
* Track+Shower GNNs next ([`config.train.gnn.yaml`](https://github.com/chenel/dune-nd-lar-reco/blob/main/configs/config.train.gnn.yaml))
* Interaction GNN last ([`config.train.inter-gnn.yaml`](https://github.com/chenel/dune-nd-lar-reco/blob/main/configs/config.train.inter-gnn.yaml))

Note that because the latter two each depend on the previous stage for input, the names of the checkpoints from the previous step are baked into the configurations (look for the `model_path` keys under the various models).  They will have to be modified if you choose to run them yourself.

### Outputs
A successful run will populate the output directory with a `.csv` log file (containing loss metrics for each iteration) and, depending on the configuration, `.ckpt` "checkpoint" files containing all network weights at a given iteration of training.

### Metrics

A Jupyter notebook illustrating how to plot relevant training metrics from the `.csv` log file may be found in (`notebooks/LossAna.ipynb`)[https://github.com/chenel/dune-nd-lar-reco/blob/main/notebooks/LossAna.ipynb] in this repository.  (It's more or less a copy of one of the DeepLearnPhysics tutorial notebooks, and doesn't contain anything new.)

# 2. Evaluation

### Running evaluation

Evaluating the models on `.larcv.root` files is similar to [training](#1-training).  There is an analogous driver script, [`RunChain.py`](https://github.com/chenel/dune-nd-lar-reco/blob/main/RunChain.py), in this repository.  Again, consult its `--help` to see what options it supports.  

Sample invocation:

```shell
python3 RunChain.py --config_file config.inference.fullchain.yaml \
                    --model_file $data/dune/nd/nd-lar-reco/train/track+intergnn-1400evs-1000Kits-batch8/snapshot-99.ckpt \
                    --batch_size 1 
                    --input_file $data/dune/nd/nd-lar-reco/supera/geom-20210405-pileup/FHC.1000015.larcv.root \
                    --output_file $data/dune/nd/nd-lar-reco/reco-out/geom-20210405-pileup/FHC.1000015.overlay.reco.npz
```

Notes:
* **Always run inference using ``--batch_size 1``** (or ensure your inference configuration files have `batch_size: 1`).  Though it's marginally (~15%) faster to run with larger batches, if an event happens not to produce any of the output products, the empty product(s) will be collapsed together with those of the subsequent event.  This will result in misalignment of the products in the output and you won't be able to match events together correctly.

### Outputs

`RunChain.py` produces two types of output:
* A dump of the output products from inference in the file specified by `--output_file` in Numpy compressed `.npz` format.
* (Optionally) a "summary" `.h5` (HDF5 format) containing higher-level object groupings.  (Currently only records track candidates; but soon, interaction info and eventually other interesting quantities should be available as well.)  The TMS group has been the primary user of these files so far.  Use `--summary_hdf5` to enable and specify the destination file.


# 3. Making plots

### Quickstart
This package contains a rudimentary plotting framework to help evaluate the performance of the reconstruction.   Once again it has a driver macro, [`Plots.py`](https://github.com/chenel/dune-nd-lar-reco/blob/main/Plots.py).
This accepts the `.npz` output of the [evaluation step](#2-evaluation) as its input.  Example invocation:
```shell
python3 Plots.py --input_file "$data/dune/nd/nd-lar-reco/reco-out/geom-20210405-pileup/FHC.1000015.overlay.reco.npz" \
                  --output_dir $data/dune/nd/nd-lar-reco/plots/fullchain-geom20210405-pileup \
                  --overwrite \
                  --only_inter
```

This example would create *only* the plots about neutrino interactions (`--only_inter`), overwriting (`--overwrite`) whatever happened to already be present in the `--output_dir`.  For more options, consult the `--help`.

### Design notes

#### Modularization of plot-making

The various plots are modularized by step in the reconstruction chain; each has its own module (`ss_plotting.py` for semantic segmentation; `ppn_plotting.py` for PPN; etc.) 
The driver script `Plots.py` has a registry `KNOWN_PLOT_MODULES`, from which the various flags for enabling or disabling the modules are built.

`Plots.py` takes care of handling command-line arguments and loading the reco products from input file.  These are then passed along to the individual plotting modules.  Each plotting module has two required ingredients:
* `BuildHists()`, which consumes the reco products and produces summaries (usually histograms, see the discussion below) from them;
* `PlotHists()`, which takes the output from `BuildHists()`, draws it, and saves its output to the location passed in from `Plots.py`.

Each module's `BuildHists()` is wrapped with a Python decorator function `@plotting_helpers.req_vars_hist` that informs `Plots.py` which products it needs from the file; this enables the `Plots.py` to optimize which products are used across all the enabled modules.  (I briefly investigated trying to deduce these automatically from the plotting functions themselves, but it appeared to require too much Python introspection black magic to be feasible in the time I had available.  Contributions welcome!)

#### Histograms

Though not by any means mandatory, most of the plotting modules make use of some histogram-making helpers in [`plotting_helpers.py`](https://github.com/chenel/dune-nd-lar-reco/blob/main/plotting_helpers.py).  The relevant ingredients:
* `plotting_helpers.Hist`, which is a wrapper class that aggregates repeated calls to `numpy.histogram()` from independent events;
* decorator function `plotting_helpers.hist_aggregate`, which declares a function in one of the plotting modules to be a "histogram function" that, given the data products, will return a value to be summed into a histogram.  (Read its docstring to get a sense of what it's doing.)  

Each of the histogram functions is then called over each event by `BuildHists()`.  (See, e.g., `track_plotting.py` for examples.)

`todo`: Create a decorator for caching internal-calculation functions and document here. 


# 4. "Event display"

mlreco3d has some helper functions that making plotting the output of the inference relatively straightforward.  This can serve as a rudimentary event display.

The Jupyter notebook [`notebooks/FullChainInference`](https://github.com/chenel/dune-nd-lar-reco/blob/main/notebooks/FullChainInference.ipynb) in this repository illustrates how this can work.
