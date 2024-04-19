
datas=[]

datas=("wC50" "wC100" "dC50" "dC100")
models=("gals" "summ")

for data in "${datas[@]}"
do
    for fold in {0..9}
    do
        echo "\n ~~~~~ Data: $data, Fold: $fold ~~~~~~~~"
        python run_base.py --data $data --fold $fold
        python run_gnn.py --data $data --fold $fold
        for model in "${models[@]}"
        do
            echo "Model: $model"
            python run.py --data $data --fold $fold --model $model
        done
    done
done
