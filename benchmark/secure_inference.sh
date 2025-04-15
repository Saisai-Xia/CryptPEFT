export PYTHONPATH=../:$PYTHONPATH
for batch_size in 32 64 96 128
do
for dataset in cifar100
do
for method in simple_fine_tuning
do
for scope in 1
do
for degree in 6
do
    python3 benchmark/secure_inference.py \
    --batch_size $batch_size \
    --rank $1 \
    --method $method \
    --dataset $dataset \
    --degree $degree \
    --transfer_scope $scope
done
done
done
done
done
