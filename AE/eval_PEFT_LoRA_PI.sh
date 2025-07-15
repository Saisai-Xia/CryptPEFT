export PYTHONPATH=../:$PYTHONPATH
for dataset in cifar100
do
    python3 AE/eval_private_inference.py \
    --batch_size 1 \
    --rank $1 \
    --net $2 \
    --method lora \
    --mode eval_LoRA \
    --dataset $dataset
done