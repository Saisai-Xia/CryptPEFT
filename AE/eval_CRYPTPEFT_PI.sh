export PYTHONPATH=../:$PYTHONPATH
for dataset in cifar10 cifar100 flowers102 svhn food101
do
    python3 AE/eval_private_inference.py \
    --batch_size 1 \
    --rank $1 \
    --method CryptPEFT \
    --net $2 \
    --mode eval_CryptPEFT \
    --dataset $dataset
done