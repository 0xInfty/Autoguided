preset="edm2-tiny-xxs"; dataset="tiny"
subdir="03_TinyImageNet/Ref/00"; ref="03_TinyImageNet/Ref/00/network-snapshot-0000639-0.100.pkl"
batchsize=256
devices=1,2; ndevices=2
device=2

# With no data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py  --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize

# With data selection
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref

# With early data selection
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --early --acid --guide-path $ref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref