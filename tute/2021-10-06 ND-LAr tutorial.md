# ND-LAr reco tutorial
Oct. 6, 2021
Jeremy Wolcott <jwolcott@fnal.gov>


## Plan for this session
- Getting set up: GPU resources
  - Wilson Cluster
  - SLAC or other institutional resources
- How to run the reco chain & plot output
- "Event display"
- (if time) Training the algorithms

## 1. Workshop setup

As was mentioned yesterday, to make full use of the ML reco toolkit (*especially* if you want to do any training) you'll need access to GPU(s).  As Kazu mentioned, the software containers that the DeepLearnPhysics group supports can use [only fairly new GPUs](https://hackmd.io/@CuhPVDY3Qregu7G4lr1p7A/SyuW79O4F#Supported-GPUS).  If you don't have your own machine with a supported GPU on it, you have a few options, listed in descending order of desirability:

* Compute cluster at your institution
  - May be best depending on hardware, availability
  - Depends on local institution support ...
* SLAC "On Demand" cluster -- Kazu et al. can help arrange
  - Requires authorization (takes ~a week)
  - Lots of high-quality GPUs
  - ~Good availability
* FNAL's Wilson Cluster has a DUNE allocation
  - Every DUNE collaborator has access
  - Not many GPUs (only 8 compatible with container)
  - Heavily subscribed, so availability is highly variable
  - Maximum slot time is 8h (serious training usually takes longer than this)

For the purpose of this tutorial, I've provided instructions for using the Wilson Cluster, since it won't require any preliminary access clearance if you didn't get it already (unlike, e.g., SLAC).

If you do have SLAC access, and you want to run things there, you'll need to use some alternate settings in the instructions below:
* If you use the Jupyter web interface to SDF, you'll want to pick "Custom Singularity Image" when you start your session.  Under "Commands to initiate Jupyter", change `SINGULARITY_IMAGE_PATH` to the container below.
* "home" area: probably you want to use `APP_DIR=/gpfs/slac/staas/fs1/g/neutrino/$USER`
* "scratch" area: use `SCRATCH=/scratch/$USER`

The items I'm providing have been copied there too:
* container: `/gpfs/slac/staas/fs1/g/neutrino/jwolcott/images/ub20.04-cuda11.0-pytorch1.7.1-larndsim.sif`
* trained weights:
  - singles: `SINGLES_WGTS=/gpfs/slac/staas/fs1/g/neutrino/jwolcott/data/dune/nd/nd-lar-reco/train/track+showergnn-380Kevs-15Kits-batch32/snapshot-1499.ckpt`
  - pileup: `PILEUP_WGTS=/gpfs/slac/staas/fs1/g/neutrino/jwolcott/data/dune/nd/nd-lar-reco/train/track+intergnn-1400evs-1000Kits-batch8/snapshot-49.ckpt`
* input files:
  - singles: `SINGLES_INPUT=/gpfs/slac/staas/fs1/g/neutrino/jwolcott/data/dune/nd/nd-lar-reco/supera/geom-20210623/neutrino.0.larcv.root`
  - pileup: `PILEUP_INPUT=/gpfs/slac/staas/fs1/g/neutrino/jwolcott/data/dune/nd/nd-lar-reco/supera/geom-20210405-pileup/FHC.1000001.larcv.root`

#### Getting set up on the Wilson Cluster

##### Login
Everybody who has a DUNE Fermilab computing account with a Kerberos principal should in principle be able to log into the Wilson Cluster.  Try it:
```
ssh <kerberos principal>@wc.fnal.gov
```
This should put you on the login node.

##### Working areas
You can make working areas for yourself if you need to keep files on the Wilson Cluster.  The storage space is fairly limited, however, so anything you want to keep long-term should be transferred to your `/dune/data/users` area on a GPVM.  (See [Wilson Cluster Filesystems](https://computing.fnal.gov/wilsoncluster/) for more.)
You can make a subdirectory for yourself at:
`/work1/dune/users/<your principal>`

***Note***: some users have had permissions issues.  For the purposes of this tutorial I've made a directory `/work1/dune/users/jwolcott/shared/` which you should be able to use as a temporary work dir.  You'll have to make the appropriate substitutions below.

Be sure to read the guide on filesystems above so you know what you can put where.
(You are also allowed to make a directory `/wclustre/dune/<your principal>` if you need a place to store big files, but beware: it's designed for large files *only*.)

##### Software setup
You'll want to grab a couple of software packages to be able to run the reco.

Put these in a "home" area inside the stub you made in the previous section:
```
export APP_DIR=/work1/dune/users/<your principal>/home   # or see above for SLAC SDF
mkdir -p APP_DIR
```

First, copy my edition of `lartpc_mlreco3d`.  (This fork is a bit behind the latest version from the main `lartpc_mlreco3d` shown in the past couple of days.  The main reason is that there's a big shift in the underlying software that isn't fully finished yet.)

```
cd $APP_DIR
git clone https://github.com/chenel/lartpc_mlreco3d.git
cd lartpc_mlreco3d
git checkout jw_dune_nd_lar
```

Then, get ahold of my ND-LAr reco scripts:
```
cd $APP_DIR
git clone https://github.com/chenel/dune-nd-lar-reco.git

```

##### Getting a worker node allocation
To actually *do* anything you'll need to start a job on a worker node in the cluster.  The WC uses SLURM for scheduling; you can see the [WC SLURM documentation](https://computing.fnal.gov/wilsoncluster/slurm-job-scheduler/) for full details.  (If you're using the SLAC SDF system, see Kazu's tutorial notes for how to get a session instead of following the instructions below.) 

For our purposes today, the following (run from the login node) should be sufficient:
```bash
srun --unbuffered --pty -A dune --partition=gpu_gce \
      --time=08:00:00 \
      --nodes=1 --ntasks-per-node=1 --gres=gpu:v100:1 \
      --cpus-per-task=3 /bin/bash
```
After a few moments, this should get you an interactive shell on a worker node (you'll note the hostname in the prompt changes from `wc` to `wcgpu0X`).  If (like me) you see some errors about your home directory not being accessible, that's ok; your home area is not generally accessible from the worker nodes because your Kerberos ticket won't be forwarded.  It may take a bit if the GPUs are heavily subscribed --- there are only 8 that are compatible with our container.

*Don't close this session* (or let your internet connection drop) until you're done for the day, if you can help it---if you do, your job will die and you'll have to start over.

##### Container setup
The ND-LAr reco has a number of dependencies that are a pain to set up.  Fortunately the SLAC group distributes Docker images that contain everything you need.
I've already grabbed one and made it available on WC.  (If you're using SLAC SDF's web interface, you'll open this container automatically when you start a shell within the web interface and don't need the instructions in this section at all.)
Once you have an open interactive shell on a worker node (see above), you can start up the container by doing the following:

```bash
# set up the container software
module load singularity

# start the container
export SCRATCH=/scratch/work  # or see above for SLAC SDF
mkdir $SCRATCH
export SINGULARITY_CACHEDIR=$SCRATCH/.singularity
export SINGULARITY_LOCALCACHEDIR=$SINGULARITY_CACHEDIR
singularity shell --userns --nv \
  --workdir=$SCRATCH \
  --home=$APP_DIR \
  -B /wclustre -B $SCRATCH \
  /work1/dune/users/jwolcott/larcv2:ub20.04-cuda11.0-pytorch1.7.1-larndsim
```

At this point you should be inside the Singularity container (your prompt will probably change to `Singularity>` to let you know).

Unfortunately you will need to re-extract the tarball each time you start a new job (so don't let the session drop until you're done, if you can help it!).


##### Environment setup inside the container
To run `mlreco3d` there's some environmental setup necessary inside the container environment.  Do the following (you may want to wrap it inside a setup script):

```
. /app/root/bin/thisroot.sh
. /app/larcv2/configure.sh
export PYTHONPATH="$PYTHONPATH:$APP_DIR/lartpc_mlreco3d"
```

##### Running ND-LAr reco tasks
You should now be ready to run training, infererence, or plotting from inside the `dune-nd-lar-reco` repository:

```
cd $APP_DIR/dune-nd-lar-reco
```

Refer to the [`README`](https://github.com/chenel/dune-nd-lar-reco#readme) inside that package for more documentation.


#### 2. ND-LAr Reco

##### Running inference
"Inference" is where you take a trained model and let it run on a dataset.  This is what most people mean by "running the reco".  Sometimes it's also called "evaluation".  Refer also to the [Evaluation](https://github.com/chenel/dune-nd-lar-reco#2-evaluation) section of the `dune-nd-lar-reco`'s `README`.

Here's an example that should run in your WC environment:
```bash
# see top of file for alternate file locations on SLAC SDF
SINGLES_WGTS=/wclustre/dune/jwolcott/dune/nd/nd-lar-reco/train/track+showergnn-380Kevs-15Kits-batch32/snapshot-1499.ckpt
SINGLES_INPUT=/wclustre/dune/jwolcott/dune/nd/nd-lar-reco/supera/singles/neutrino.0.larcv.root
python3 RunChain.py --config_file $APP_DIR/dune-nd-lar-reco/configs/config.inference.fullchain-singles.yaml \
                    --model_file $SINGLES_WGTS \
                    --batch_size 1 \
                    -n 5 \
                    --input_file $SINGLES_INPUT \
                    --output_file $SCRATCH/neutrino.0.larcv.5events.npz
```

The `-n 5` in this command limits processing to the first 5 events; it can be removed to run over the whole file (which will take a while).

There is also a "pileup" file generated with neutrinos overlaid to approximate 1.2 MW beam configuration in `PILEUP_INPUT=/wclustre/dune/jwolcott/dune/nd/nd-lar-reco/supera/1.2MW/FHC.1000001.larcv.root`.  This can be processed using the `configs/config.inference.fullchain-pileup.yaml` configuration and `PILEUP_WGTS=/wclustre/dune/jwolcott/dune/nd/nd-lar-reco/train/track+intergnn-1400evs-1000Kits-batch8/snapshot-49.ckpt`.  (Again see top of this file for SLAC SDF alternate locations.)  Beware, however, that it is *much* slower than the 'singles'!

##### Exploring the output file
As you've seen from the previous days, the output of the reco is a bunch of data products which take the form of Numpy arrays.  Here, that information has been stored in the file in a numpy compressed format.  (I had designs of making it an HDF5 data store, but never got around to figuring out the conversion.)  You can look at it in the Python prompt (this would also work in a Jupyter notebook, but running the notebook server on the Wilson Cluster is difficult):

```
Singularity> python3 
Python 3.8.5 (default, May 27 2021, 13:30:53) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> import os.path
>>> f = open(os.path.expandvars("$SCRATCH/neutrino.0.larcv.5events.npz"), "rb")
>>> npz = np.load(f, allow_pickle=True)
>>> list(npz.keys())
['metadata', 'event_base', 'input_data', 'segment_label', 'cluster_label', 'particles_label', 'particles_raw', 'i
ndex', 'segmentation', 'ppn_feature_enc', 'ppn_feature_dec', 'points', 'ppn1', 'ppn2', 'mask_ppn1', 'mask_ppn2', 
'fragments', 'fragments_seg', 'shower_fragments', 'shower_group_pred', 'track_fragments', 'track_node_pred', 'tra
ck_edge_pred', 'track_edge_index', 'track_group_pred', 'particles', 'particles_seg', 'ppn_post', 'shower_node_pre
d', 'shower_edge_pred', 'shower_edge_index']
```
Each of these keys corresponds to a single data product.
Francois has shown you a lot about how to interpret them in the other tutorial days (though I have also included the SLAC group's tutorials page in the links below for future reference); to remind you, typically, each consists of a list of `numpy` arrays, for which there is one entry per event (singles) or spill (overlay).  For instance, the voxels that were input to the reconstruction are in `input_data`. So, the voxels from the first event in our sample output file are:
```
>>> npz["input_data"][:1]
array([array([[-2.18000000e+02, -1.34400000e+02,  7.18000000e+02,
                0.00000000e+00,  9.59462777e-04],
              [-2.38000000e+02, -1.11600000e+02,  7.82800000e+02,
                0.00000000e+00,  9.86517407e-04],
              [-8.48000000e+01, -1.38800000e+02,  8.11600000e+02,
                0.00000000e+00,  6.13885713e-05],
              ...,
              [-1.77200000e+02, -1.32400000e+02,  8.78400000e+02,
                0.00000000e+00,  1.09288409e-01],
              [-1.81200000e+02, -1.32000000e+02,  8.80400000e+02,
                0.00000000e+00,  1.69851370e-02],
              [-1.82000000e+02, -1.28400000e+02,  8.80400000e+02,
                0.00000000e+00,  1.21262833e-01]])               ],
      dtype=object)
```
Here each row corresponds to a single voxel, with columns:
```
[voxel x-pos, voxel y-pos, voxel z-pos, event/spill, energy]
```

The output of the semantic segmentation algorithm on the input voxels for the events we processed is in the `segmentation` product.  Hopefully looking at the first event in our sample file from above is a familiar operation by today:
```
npz["segmentation"][:1]
array([array([[-4.3260612 , -0.40785632, -4.700892  , -5.290818  ,  4.8061604 ],
              [-3.7457318 , -1.217933  , -4.4402623 , -4.7870884 ,  5.355474  ],
              [-3.1292    , -1.4162513 , -4.6319337 , -5.963332  ,  5.8305125 ],
              ...,
              [-4.1908927 , -1.1247922 , -5.8151903 , -4.9852867 ,  6.023964  ],
              [-3.9636374 , -1.4159595 , -5.0879817 , -5.007062  ,  5.8366804 ],
              [-4.470032  , -0.955091  , -5.871276  , -4.9818454 ,  6.0377374 ]],
             dtype=float32)                                                      ],
      dtype=object)

```
The columns in this particular sub-array corresond to the scores for the types of segmentation classes: "track", "shower", etc.  The class for each column index can be printed out using my `plotting_helpers.py`:
```
>>> import plotting_helpers as pl
>>> pl.SHAPE_LABELS
{3: 'Delta', 4: 'LEScatter', 2: 'Michel', 0: 'Shower', 1: 'Track'}
```
Each row in this 2D array corresponds to one input voxel from before.

A better way to explore these products, if you want to do that, is via a Jupyter notebook.  (You probably want to do this on the SLAC SDF system, your own institution's compute cluster, or a machine you have locally, if you can run Jupyter there.)

You can also learn a lot about how the products are organized and what's in them from reading the plotting code discussed in the next section.

##### Making plots from the output file

To make plots from these products, one *could* intercept the products as they are made and aggregate histograms that way, or tack a histogramming operation onto the end of the processing job (more or less what happens when you do it inside a Jupyter notebook like in the other tutorials).  
I've chosen instead to save all the products from an inference run to a file, then run plot-making on that file separately.  This makes it easier to run only the plotting over and over again, which is helpful when trying to get plot formatting etc. right.  (Again, this can be done in a Jupyter notebook, but once you quit the kernel where the reco was run, you'll lose all that.  If you want, you can combine both worlds by taking the output of `RunChain.py` and importing the machinery described below into a notebook.  But since the notebook server is tough to run on the Wilson Cluster, we'll stick to the command line for today.)

As documented in the ("Making plots")[https://github.com/chenel/dune-nd-lar-reco#3-making-plots] section of the `dune-nd-lar-reco` package, I wrote a small histogram plotting framework to work with these files.  Here's a sample invocation from our test output above:

```bash
mkdir -p $SCRATCH/plots/test
python3 Plots.py --input_file "$SCRATCH/neutrino.0.larcv.5events.npz" \
                  --output_dir $SCRATCH/plots/test \
                  --disable_inter
```

Here I've disabled the "interaction" plots because for these "singles" (one event == one neutrino event) the neutrino interaction segmentation isn't relevant, and wasn't run in the  `config.inference.fullchain-singles.yaml` configuration chosen for the evaluation run.
There will be some `RuntimeWarning`s in the output here because our input file only contains 5 events, and thus there are lots of empty bins in the histograms that get created, but it'll give you a feel for what can be done.  (If you like you're welcome to go back and reprocess the whole input file without `-n 5` to see the full plots.)

Remember that you'll have to manually copy the output plots (as well as the `.npz` file) off of the Wilson Cluster before your session ends, otherwise you'll lose them when the `/scratch` area is purged!
I recommend copying them to a subdir of your user area, e.g:
```bash
cp -r $SCRATCH/plots/test /work1/dune/users/$USER/plots/
```
from whence you can copy them to `/dune/data` via `scp` or elsewhere according to your heart's content.

##### Event display

Kazu's and Francois's tutorials have given a very comprehensive tutorial on plotting the output of the reco.
I've included  [`notebooks/FullChainInference.ipynb`](https://github.com/chenel/dune-nd-lar-reco/blob/main/notebooks/FullChainInference.ipynb) in this repository as a reference of a few things that can be done.
One thing that this notebook does which the other tutorials have not is that it converts the display into geometry coordinates, which makes it much more straightforward to compare to what came out of `edep-sim`.

If you're on the SLAC SDF service, you can open the notebook yourself to follow along.

##### Training networks

Training follows the same patterns as the other bits, and by now a lot of the stuff in the `dune-nd-lar-reco`'s [Training](https://github.com/chenel/dune-nd-lar-reco#1-training) section will look familiar.  Francois's tutorials from Monday will give you basic orientation if you need it.

The main consideration to be aware of is that at the moment, training is fairly memory hungry, particularly for the "interaction" GNN.

A sample training invocation is below.  It should be noted that in reality you'd want to train with way more input files than this, because the events in this file will be consumed quickly, and the weights will become overtrained in only a few thousand iterations.
```bash
python3 -u TrainChain.py -c configs/config.train.uresnet+ppn.yaml -i $SINGLES_INPUT --random-seed 20211008 --output_dir $SCRATCH/train
```

Some of the training metrics that Francois discussed the other day can be plotted using the notebook [`notebooks/LossAna.ipynb`](https://github.com/chenel/dune-nd-lar-reco/blob/main/notebooks/LossAna.ipynb), which is a very lightly edited copy of one of the notebooks Francois showed.

## Links
* JW slides from internal consortium review: <https://indico.fnal.gov/event/48671/#57-event-reconstruction>
* JW slides on 'interaction' reconstruction: <https://indico.fnal.gov/event/50176/#1-ml-reco-update>
* Wilson Cluster: <https://computing.fnal.gov/wilsoncluster/>
* DeepLearnPhysics:
    - code <https://github.com/DeepLearnPhysics>
    - tutorials etc. <http://deeplearnphysics.org/lartpc_mlreco3d_tutorials/>
* Analysis code: <https://github.com/chenel/dune-nd-lar-reco>
