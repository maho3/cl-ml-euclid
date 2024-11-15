
cd /home/mattho/git/cl-ml-euclid/jobs

# DATAS=(wC50 wC100 dC50 dC100)
# MODELS=(gals summ, gnn)
DATAS=(wC100)
MODELS=(gals)
FROM_SCRATCH=1

for DATA in ${DATAS[@]}; do
    for MODEL in ${MODELS[@]}; do
        PARAMS="DATA=${DATA},MODEL=${MODEL},FROM_SCRATCH=${FROM_SCRATCH}"
        echo "Submitting job for $PARAMS"
        if [ $MODEL = "gnn" ]; then
            qsub -N "${MODEL}_${DATA}" -v $PARAMS -l nodes=1:has1gpu:ppn=32,mem=64gb run.sh
        else
            qsub -N "${MODEL}_${DATA}" -v $PARAMS run.sh
        fi
    done
done