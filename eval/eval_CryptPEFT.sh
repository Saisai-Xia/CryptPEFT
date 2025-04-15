export PYTHONPATH=../:$PYTHONPATH
# image
for rank in 300
do
for num_repeat_blk in 1
do
for adapter in CryptPEFT
do
for lr in 0.01
do
for scale in 4.0
do
for ds in cifar100
do
    python3 eval/eval_CryptPEFT.py \
    --batch_size 50 \
    --device cuda:1 \
    --adapt_on \
    --approx 6 \
    --adapter_type $adapter \
    --adapter_scaler $scale \
    --rank $rank \
    --num_repeat_blk $num_repeat_blk \
    --epochs 20 \
    --model Vit_B_16 \
    --data_path Adapter/experiments/dataset \
    --output_dir Adapter/experiments/output \
    --lr $lr \
    --dataset $ds \
    --log_dir eval/approx_result

done
done
done
done
done
done