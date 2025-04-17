export PYTHONPATH=../:$PYTHONPATH
for batch_size in 1
do
for dataset in cifar100
do
for method in mpcvit
do
for atten_method in CryptPEFT
do
for last_layers in 1
do
    python3 benchmark/secure_inference.py \
    --batch_size $batch_size \
    --rank $1 \
    --atten_method $atten_method \
    --method $method \
    --dataset $dataset \
    --ablation \
    --transfer_scope $last_layers
done
done
done
done
done
