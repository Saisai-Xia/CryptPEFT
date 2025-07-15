export PYTHONPATH=../:$PYTHONPATH
for dataset in cifar100
do
    python3 AE/eval_private_inference.py \
    --batch_size 1 \
    --rank $1 \
    --net $2 \
    --method simple_fine_tuning \
    --dataset $dataset \
    --mode eval_Last_2_Layers \
    --transfer_scope 2
done