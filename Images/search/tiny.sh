maindir="06_CorrectEMA/Early_AJEST/00"; epoch=20500
refdir="04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl"
guidance_weights=1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5

# device=0 --> emas in [0.04, 0.06, 0.08, 0.1 , 0.12, 0.14]
# --> output="grid_search_0_0.json"
device=1
ema=0.150
# device=1 --> emas in [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
# --> output="grid_search_0_0.json"

output_filepath=$maindir"/grid_search/grid_search_"$device"_"$ema".json"
CUDA_VISIBLE_DEVICES=$device python Images/run_grid_search.py --emas=$ema --guidance-weights=$guidance_weights --super-dir=$maindir --guide-path=$refdir --out-epoch=$epoch --out-filepath=$output_filepath