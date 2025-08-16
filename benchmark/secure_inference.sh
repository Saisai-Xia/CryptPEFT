export PYTHONPATH=../:$PYTHONPATH
for batch_size in 1
do
for dataset in cifar10 cifar100 food101 svhn flowers102
do
for method in CryptPEFT
do
for atten_method in CryptPEFT
do
    python3 benchmark/secure_inference.py \
    --batch_size $batch_size \
    --rank $1 \
    --atten_method $atten_method \
    --method $method \
    --dataset $dataset 
done
done
done
done

