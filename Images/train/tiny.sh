dataset="tiny"; ref="xxs"; learner="xs"; 
subdir="06_CorrectEMA"; series="00"
refdir="04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl"
# refdir="04_Tiny_LR/Ref/00/network-snapshot-0001279-0.100.pkl"
# refdir="04_Tiny_LR/Ref/00/network-snapshot-0000359-0.100.pkl"
bsref=256; bslearner=128;
devices=1,2; ndevices=2
device=2

# With no data selection, small reference model
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir"/Ref/"$series --dataset $dataset --preset="edm2-"$dataset"-"$ref --batch-gpu=$bsref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py  --outdir=$subdir"/Ref/"$series --dataset $dataset --preset="edm2-"$dataset"-"$ref --batch-gpu=$bsref

#### AJEST ###########################################

# With early data selection, large learner model
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir"/Early_AJEST/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearner --early --acid --guide-path $refdir
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir"/Early_AJEST/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearner --early --acid --guide-path $refdir

# With data selection all the way, large learner model
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir"/AJEST/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearner --acid --guide-path $refdir
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir"/AJEST/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearner --acid --guide-path $refdir

#### RANDOM ##########################################

# With early data selection, large learner model
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir"/Early_Random/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearnsmall --early --selection
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir"/Early_Random/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearnsmall --early --selection

# With data selection all the way, large learner model
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir"/Random/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearnsmall --selection
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir"/Random/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearnsmall --selection

#### BASELINE ###########################################

# subdir="04_Tiny_LR"; series="04"

# With no data selection, large learner model
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir"/Baseline/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearner
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py  --outdir=$subdir"/Baseline/"$series --dataset $dataset --preset="edm2-"$dataset"-"$learner --batch-gpu=$bslearner