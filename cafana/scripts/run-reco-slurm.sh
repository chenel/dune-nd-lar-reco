#!/bin/bash

#SBATCH --job-name=dune-nd-ml-reco    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jeremy.wolcott@tufts.edu     # Where to send mail
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --output=/cluster/tufts/minos/jwolcott/data/scratch/job-output/mlreco_%j_%A-%a.out
#SBATCH --error=/cluster/tufts/minos/jwolcott/data/scratch/job-output/mlreco_%j_%A-%a.err
#SBATCH --partition=ccgpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-gpu=40gb
#SBATCH --array=0-99

app=/cluster/tufts/minos/jwolcott/app
data=/cluster/tufts/minos/jwolcott/data

script="`mktemp $data/scratch/singularity-scripts/run-mlreco-XXXXXXX.sh`"
if [ -z "$script" ]; then
  exit 1
fi
echo "writing script to tempfile: $script"

cat << EOF > $script
#!/bin/bash

. /cluster/home/jwolco01/scripts/larcv
export PYTHONPATH="\$PYTHONPATH:$app/dune/nd/nd-lar-reco/lartpc_mlreco3d"
cd $app/dune/nd/dune-nd-lar-reco/

python3 -u RunChain.py --config_file $app/dune/nd/dune-nd-lar-reco/configs/config.inference.fullchain-singles.yaml \
                       --model_file $data/dune/nd/nd-lar-reco/train/uresnet+ppn-380Kevs-50Kits-batch32/snapshot-23999.ckpt \
                       --batch_size 1 \
                       --input_file $data/dune/nd/nd-lar-reco/supera/geom-20210623/neutrino.${SLURM_ARRAY_TASK_ID}.larcv.root \
                       --output_file $data/dune/nd/nd-lar-reco/reco-out/neutrino.${SLURM_ARRAY_TASK_ID}.reco.npz \
                       --summary_hdf5 $data/dune/nd/nd-lar-reco/reco-out/neutrino.${SLURM_ARRAY_TASK_ID}.summary.h5
EOF
chmod +x $script

module load singularity/3.6.1
singularity exec --nv -B $HOME -B $app -B $data \
                $data/singularity-images/ub20.04-cuda11.0-pytorch1.7.1-larndsim.sif \
                bash $script \
  || exit $?

echo "Succeeded!  Removing temp script"
rm $script

exit 0