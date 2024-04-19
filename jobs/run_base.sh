
cd /home/mattho/git/cl-ml-euclid

DATAS=(wC50 wC100 dC50 dC100)


source /data80/mattho/anaconda3/bin/activate
conda activate ili-torch

for DATA in ${DATAS[@]}; do
    for fold in {0..9}; do
        echo "Running model base on data $DATA fold $fold"
        python run_base.py --data $DATA --fold $fold
    done
done
