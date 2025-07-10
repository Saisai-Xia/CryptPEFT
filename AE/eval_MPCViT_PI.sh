export PYTHONPATH=../:$PYTHONPATH
for dataset in cifar10 cifar100
do
    python3 AE/eval_private_inference.py \
    --batch_size 1 \
    --rank $1 \
    --method mpcvit \
    --net $2 \
    --dataset $dataset
done