#!/bin/bash

#SBATCH --job-name=dune-nd-ml-reco    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jeremy.wolcott@tufts.edu     # Where to send mail
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --output=/cluster/tufts/minos/jwolcott/data/scratch/job-output/mlreco_%j_%A-%a.out
#SBATCH --partition=ccgpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-gpu=40gb
#SBATCH --array=0-99

function set_default()
{
  varname="$1"
  val="$2"
  if [ -z "${!varname}" ]; then
    printf -v "$varname" '%s' "$val"
    echo "WARNING: var \$${varname} was not specified in shell.  Using default: $val"
  fi
}

# hopefully the user has passed these using `--export` to `sbatch`
set_default app /cluster/tufts/minos/jwolcott/app
set_default data /cluster/tufts/minos/jwolcott/data
set_default sing_images /cluster/tufts/minos/jwolcott/data/singularity-images

if [ -d "$app/dune/nd/nd-lar-reco" ]; then
  dunesoft=$app/dune/nd/nd-lar-reco
else
  dunesoft=$app
fi

echo "Assuming DUNE software is located in: $dunesoft"

script="`mktemp $data/scratch/singularity-scripts/run-mlreco-XXXXXXX.sh`"
if [ -z "$script" ]; then
  exit 1
fi
echo "writing script to tempfile: $script"

cat << EOF > $script
#!/bin/bash

. $HOME/scripts/larcv
export PYTHONPATH="\$PYTHONPATH:$dunesoft/lartpc_mlreco3d"
cd $dunesoft/dune-nd-lar-reco/ || exit 1

python3 -u RunChain.py --config_file $dunesoft/dune-nd-lar-reco/configs/config.inference.fullchain-singles.yaml \
                       --model_file $data/dune/nd/nd-lar-reco/train/track+showergnn-380Kevs-15Kits-batch32/snapshot-1499.ckpt \
                       --batch_size 1 \
                       --input_file $data/dune/nd/nd-lar-reco/supera/geom-20210623/neutrino.${SLURM_ARRAY_TASK_ID}.larcv.root \
                       --output_file $data/dune/nd/nd-lar-reco/reco-out/geom-20210623/neutrino.${SLURM_ARRAY_TASK_ID}.reco.npz \
                       --summary_hdf5 $data/dune/nd/nd-lar-reco/reco-out/geom-20210623/neutrino.${SLURM_ARRAY_TASK_ID}.summary.h5
EOF
chmod +x $script

module load singularity/3.6.1
singularity exec --nv -B $HOME -B $app -B $data \
                $sing_images/ub20.04-cuda11.0-pytorch1.7.1-larndsim.sif \
                bash $script \
  || exit $?

echo "Succeeded!  Removing temp script"
rm $script

exit 0
