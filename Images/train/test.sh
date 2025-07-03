preset="test-training"; dataset="cifar10"
subdir="99_Test/05"; ref="01_CIFAR10/01_Ref/network-snapshot-0025165-0.050.pkl"
batchsize=512
devices=1,2; ndevices=2
device=2

# With no data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py  --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize

# With data selection
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref