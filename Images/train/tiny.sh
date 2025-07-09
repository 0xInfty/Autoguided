preset="edm2-tiny-xs"; dataset="tiny"
subdir="04_Tiny_LR/Ref/00"; ref="04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl"
subdir2="04_Tiny_LR/Early_AJEST/00"; subdir3="04_Tiny_LR/AJEST/00"; 
batchsize=256; batchsize2=128;
devices=1,2; ndevices=2
device=2

# With no data selection
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py  --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize

# With early data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir2 --dataset $dataset --preset=$preset --batch-gpu=$batchsize2 --early --acid --guide-path $ref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir2 --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref

# With data selection all the way
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir3 --dataset $dataset --preset=$preset --batch-gpu=$batchsize2 --acid --guide-path $ref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir3 --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref