export PYTHONPATH=../:$PYTHONPATH
for dataset in cifar100
do
    python3 AE/eval_private_inference.py \
    --batch_size 1 \
    --rank $1 \
    --net $2 \
    --method SFT \
    --dataset $dataset \
    --transfer_scope 1
done