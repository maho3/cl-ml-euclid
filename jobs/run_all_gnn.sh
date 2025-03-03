
cd /home/mattho/git/cl-ml-euclid

DATAS=(wC50 wC100 dC50 dC100)
FOLDS=(0 1 2 3 4 5 6 7 8 9)

for DATA in ${DATAS[@]}; do
    for FOLD in ${FOLDS[@]}; do
        echo "Running python run_gnn.py --data=${DATA} --fold=${FOLD}"
        python run_gnn.py --data=${DATA} --fold=${FOLD}
    done
done