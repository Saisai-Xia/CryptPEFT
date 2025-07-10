export PYTHONPATH=../:$PYTHONPATH
for atten_method in CryptPEFT MPCViT SHAFT MPCFormer
do
    python3 AE/eval_private_inference.py \
    --batch_size 1 \
    --rank $1 \
    --atten_method $atten_method \
    --method CryptPEFT \
    --net $2 \
    --dataset cifar100 \
    --ablation 
done
