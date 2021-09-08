# ND-LAr reco crash course
Sept. 8, 2021
Jeremy Wolcott <jwolcott@fnal.gov>


## Plan for today
1. Introductions
2. 5-minute orientation to ND-LAr reco
3. Workshop:
	- Getting set up on the Wilson Cluster
	- How to run the reco chain & plot output
	- (if time) Training the algorithms

## 1. Workshop setup

#### Getting set up on the Wilson Cluster

The Wilson Cluster has a DUNE allocation we can use as a stop-gap for today's tutorial session.  Ultimately the SLAC resources are likely to be better, however, because there are more (and, I think, better) GPUs and they are not as heavily subscribed.  More importantly, the DUNE allocation is limited to 8-hour jobs, which you can't really use for serious training.

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
Be sure to read the guide on filesystems above so you know what you can put where.
(You are also allowed to make a directory `/wclustre/dune/<your principal>` if you need a place to store big files, but beware: it's designed for large files *only*.)

##### Software setup
You'll want to grab a couple of software packages to be able to run the reco.

Put these in a "home" area inside the stub you made in the previous section:
```
mkdir /work1/dune/users/<your principal>/home
```

First, copy my edition of `lartpc_mlreco3d`.  (This fork has a couple of divergences from the main `lartpc_mlreco3d` that are specific to the DUNE ND-LAr situation.)

```
cd /work1/dune/users/<your principal>/home
git clone https://github.com/chenel/lartpc_mlreco3d.git
cd lartpc_mlreco3d
git checkout jw_dune_nd_lar
```

Then, get ahold of my ND-LAr reco scripts:
```
cd /work1/dune/users/<your principal>/home
git clone https://github.com/chenel/dune-nd-lar-reco.git

```

##### Getting a worker node allocation
To actually *do* anything you'll need to start a job on a worker node in the cluster.  The WC uses SLURM for scheduling; you can see the [WC SLURM documentation](https://computing.fnal.gov/wilsoncluster/slurm-job-scheduler/) for full details.

For our purposes today, the following (run from the login node) should be sufficient:
```bash
srun --unbuffered --pty -A dune --partition=gpu_gce \
      --time=08:00:00 \
      --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
      --cpus-per-task=3 /bin/bash
```
After a few moments, this should get you an interactive shell on a worker node (you'll note the hostname in the prompt changes from `wc` to `wcgpu0X`).  If (like me) you see some errors about your home directory not being accessible, that's ok; your home area is not generally accessible from the worker nodes because your Kerberos ticket won't be forwarded.

*Don't close this session* (or let your internet connection drop) until you're done for the day, if you can help it---if you do, your job will die and you'll have to start over.

##### Container setup
The ND-LAr reco has a number of dependencies that are a pain to set up.  Fortunately the SLAC group distributes Docker images that contain everything you need.
I've already grabbed one and made it available on WC.
Once you have an open interactive shell on a worker node (see above), you can start up the container by doing the following:

```bash
# set up the container software
module load singularity
module load cuda11

# unpack the container to the scratch area for this job.
# this will take about 10 minutes.
tar xf /wclustre/dune/jwolcott/singularity-images/larcv2_ub20.04-cuda11.0-pytorch1.7.1-larndsim.tar.bz2  --directory /scratch

# start the container
mkdir /scratch/work
export SINGULARITY_LOCALCACHEDIR=/scratch/.singularity
# ** be sure to replace <your principal> with your actual Kerberos principal**
singularity shell --userns --nv \
  --workdir=/scratch/work \
  --home=/work1/dune/users/<your principal>/home \
  -B /wclustre -B /scratch/work \
  /scratch/larcv2:ub20.04-cuda11.0-pytorch1.7.1-larndsim
```

At this point you should be inside the Singularity container (your prompt will probably change to `Singularity>` to let you know).

Unfortunately you will need to re-extract the tarball each time you start a new job (so don't let the session drop until you're done, if you can help it!).


##### Environment setup inside the container
To run `mlreco3d` there's some environmental setup necessary inside the container environment.  Do the following (you may want to wrap it inside a setup script):

```
. /app/root/bin/thisroot.sh
. /app/larcv2/configure.sh
export PYTHONPATH="$PYTHONPATH:$HOME/lartpc_mlreco3d"
```

##### Running ND-LAr reco tasks
You should now be ready to run training, infererence, or plotting from inside the `dune-nd-lar-reco` repository:

```
cd ~/dune-nd-lar-reco
```

Refer to the [`README`](https://github.com/chenel/dune-nd-lar-reco#readme) inside that package for more documentation.


#### 2. ND-LAr Reco

##### Running inference
"Inference" is where you take a trained model and let it run on a dataset.  This is what most people mean by "running the reco".  Sometimes it's also called "evaluation".  Refer also to the [Evaluation](https://github.com/chenel/dune-nd-lar-reco#2-evaluation) section of the `dune-nd-lar-reco`'s `README`.

Here's an example that should run in your WC environment:
```bash
python3 RunChain.py --config_file configs/config.inference.fullchain-singles.yaml \
                    --model_file /wclustre/dune/jwolcott/dune/nd/nd-lar-reco/train/track+showergnn-380Kevs-15Kits-batch32/snapshot-1499.ckpt \
                    --batch_size 1 \
                    -n 5 \
                    --input_file /wclustre/dune/jwolcott/dune/nd/nd-lar-reco/supera/singles/neutrino.0.larcv.root \
                    --output_file /scratch/work/neutrino.0.larcv.5events.npz
```

The `-n 5` in this command limits processing to the first 5 events; it can be removed to run over the whole file (which will take a while).

There is also a "pileup" file generated with neutrinos overlaid to approximate 1.2 MW beam configuration in `/wclustre/dune/jwolcott/dune/nd/nd-lar-reco/supera/1.2MW/`.  This can be processed using the `config/config.inference.fullchain-pileup.yaml` configuration.  Beware, however, that it is *much* slower than the 'singles'!

##### Exploring the output file
The output of the reco is a bunch of data products, stored in a numpy compressed format.  (I had designs of making it an HDF5 data store, but never got around to figuring out the conversion.)  You can look at it in the Python prompt:

```
Singularity> python3 
Python 3.8.5 (default, May 27 2021, 13:30:53) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> f = open("/scratch/work/neutrino.0.larcv.5events.npz", "rb")
>>> npz = np.load(f, allow_pickle=True)
>>> list(npz.keys())
['metadata', 'event_base', 'input_data', 'segment_label', 'cluster_label', 'particles_label', 'particles_raw', 'i
ndex', 'segmentation', 'ppn_feature_enc', 'ppn_feature_dec', 'points', 'ppn1', 'ppn2', 'mask_ppn1', 'mask_ppn2', 
'fragments', 'fragments_seg', 'shower_fragments', 'shower_group_pred', 'track_fragments', 'track_node_pred', 'tra
ck_edge_pred', 'track_edge_index', 'track_group_pred', 'particles', 'particles_seg', 'ppn_post', 'shower_node_pre
d', 'shower_edge_pred', 'shower_edge_index']
```
Each of these keys corresponds to a single data product.
The way to interpret each product is too much for this one tutorial---you'll get a much better feel for them by visiting the SLAC/DeepLearnPhysics group's tutorials (see links below)---but, typically, each consists of a `numpy` array for which there is one entry per event (singles) or spill (overlay).  For instance, the voxels that were input to the reconstruction are in `input_data`. So, the voxels from the first event in our sample output file are:
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
[voxel x-pos, voxel y-pos, voxel z-pos, 'batch', energy]
```
("Batch" usually corresponds to "event" or "spill" in `mlreco3d`.)

The output of the semantic segmentation algorithm on the input voxels for the events we processed is in the `segmentation` product; to once again look at the first event in our sample file from above:
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

A better way to explore these products, if you want to do that, is via a Jupyter notebook, like the event display---hopefully we'll get to that later.

You can also learn a lot about how the products are organized and what's in them from reading the plotting code discussed in the next section.

##### Making plots from the output file

To make plots from these products, one *could* intercept the products as they are made and aggregate histograms that way.  (The DeepLearnPhysics folks usually do that to make performance metrics when they're exploring event training.)
I've chosen instead to save all the products from an inference run to a file, then run plot-making on that file separately.  This makes it easier to run only the plotting over and over again (which is helpful when trying to get plot formatting etc. right).

As documented in the ("Making plots")[https://github.com/chenel/dune-nd-lar-reco#3-making-plots] section of the `dune-nd-lar-reco` package, I wrote a small histogram plotting framework to work with these files.  Here's a sample invocation from our test output above:

```bash
mkdir -p /scratch/work/plots/test
python3 Plots.py --input_file "/scratch/work/neutrino.0.larcv.5events.npz" \
                  --output_dir /scratch/work/plots/test \
                  --disable_inter
```

Here I've disabled the "interaction" plots because for these "singles" (one event == one neutrino event) the neutrino interaction segmentation isn't relevant, and wasn't run in the  `config.inference.fullchain-singles.yaml` configuration chosen for the evaluation run.  

Remember that you'll have to manually copy the output plots (as well as the `.npz` file) off of the Wilson Cluster before your session ends, otherwise you'll lose them when the `/scratch` area is purged!


##### Event display



##### Training networks

We probably won't get this far today.  But by now a lot of the stuff in the `dune-nd-lar-reco`'s [Training](https://github.com/chenel/dune-nd-lar-reco#1-training) section will look familiar, and you may be able to make headway without help.  You should also check out the DeepLearnPhysics tutorials (see links below).

## Links
* JW slides from internal consortium review: <https://indico.fnal.gov/event/48671/#57-event-reconstruction>
* JW slides on 'interaction' reconstruction: <https://indico.fnal.gov/event/50176/#1-ml-reco-update>
* Wilson Cluster: <https://computing.fnal.gov/wilsoncluster/>
* DeepLearnPhysics:
    - code <https://github.com/DeepLearnPhysics>
    - tutorials etc. <http://deeplearnphysics.org/lartpc_mlreco3d_tutorials/>
* Analysis code: <https://github.com/chenel/dune-nd-lar-reco>