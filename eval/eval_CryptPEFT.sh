export PYTHONPATH=../:$PYTHONPATH
for adapter in CryptPEFT
do
for lr in 0.01
do
for scale in 0.5
do
for ds in flowers102 cifar10 svhn
do
for arch in CryptPEFT
do
for option in test
do
    python3 eval/eval_CryptPEFT.py \
    --batch_size 50 \
    --device cuda:0 \
    --option $option \
    --adapt_on \
    --adapter_type $adapter \
    --adapter_arch $arch \
    --adapter_scaler $scale \
    --num_repeat_blk 1 \
    --epochs 20 \
    --model Vit_B_16 \
    --data_path Adapter/experiments/dataset \
    --output_dir Adapter/experiments/output \
    --lr $lr \
    --dataset $ds \
    --log_dir eval/search_result

done
done
done
done
done
done
