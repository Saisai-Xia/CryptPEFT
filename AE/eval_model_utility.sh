export PYTHONPATH=../:$PYTHONPATH
for eval_method in CRYPTPEFT_Efficiency_first CRYPTPEFT_Utility_first
do
for ds in cifar10 cifar100 flowers102 svhn food101
do
    python3 AE/eval_model_utility.py \
    --batch_size 50 \
    --device cpu \
    --resume \
    --eval_method $eval_method \
    --adapter_scaler 0.5 \
    --epochs 20 \
    --model Vit_B_16 \
    --data_path Adapter/experiments/dataset \
    --lr 0.01 \
    --dataset $ds \
    --log_dir AE/eval_result
done
done
