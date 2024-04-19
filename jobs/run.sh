#PBS -N CL-ML-EUCLID
#PBS -q batch
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=16,mem=32gb
#PBS -t 0-9
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log

# DATA=wC50
# MODEL=gals
FOLD=$PBS_ARRAYID
WDIR=/home/mattho/git/cl-ml-euclid
cd $WDIR

# Load modules
echo "Loading modules"
module load cuda
source /data80/mattho/anaconda3/bin/activate
conda activate ili-torch

# Run the model
echo "Running model $MODEL on data $DATA"
if [ $MODEL = "gnn" ]; then
    python run_gnn.py --data $DATA --fold $FOLD --from_scratch=$FROM_SCRATCH
else
    python run.py --data $DATA --fold $FOLD --model $MODEL --from_scratch=$FROM_SCRATCH
fi
