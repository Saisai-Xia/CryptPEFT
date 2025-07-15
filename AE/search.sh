export PYTHONPATH=../:$PYTHONPATH
for eval_method in search
do
for ds in flowers102
do
    python3 AE/eval_model_utility.py \
    --batch_size 50 \
    --device cuda:0 \
    --resume \
    --eval_method $eval_method \
    --adapter_scaler 0.5 \
    --epochs 20 \
    --model Vit_B_16 \
    --data_path Adapter/experiments/dataset \
    --lr 0.01 \
    --dataset $ds \
    --log_dir AE/search_result
done
done
