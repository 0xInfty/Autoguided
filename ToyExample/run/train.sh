# Activate SCID, if needed
# conda activate SCID

# Simply run to test it works
. ToyExample/run/run_test.sh

# Train a guide model
# python ToyExample/toy_example.py train --dim 32 --total-iter 512 --outdir "ToyExample/Ref" --seed 0 --val --test --verbose --logging "ToyExample/log_Ref.txt"

# Train once: no ACID, ACID
# . run_once.sh

# Run for different random seeds
# . run_sweep_seed.sh

# Change n, compare to the default value n = 16 
# Change filter_ratio, compare to the default value filter_ratio = 0.8
# . run_sweep_ACID.sh

# Run for different batch size
# . run_sweep_batch_size.sh

# Change learnability, compare to the default value learnability = True
# python ToyExample/toy_example.py train --no-diff --outdir "ToyExample/05_ACIDParams/No_ACID_N_16_F_0.80_L_0" --no-acid --seed 0 --verbose --logging "ToyExample/05_ACIDParams/No_ACID_N_16_F_0.80_L_0.txt"
# python ToyExample/toy_example.py train --no-diff --outdir "ToyExample/05_ACIDParams/ACID_N_16_F_0.80_L_0" --acid --seed 0 --verbose --logging "ToyExample/05_ACIDParams/ACID_N_16_F_0.80_L_0.txt"

# Run a test on a trained model
# python ToyExample/toy_example.py test --acid --seed 0 --net-path "ToyExample/08_GuideAsACIDRef/InvertedGuideACID&Deactivate/iter4096learner.pkl" --ema-path "ToyExample/08_GuideAsACIDRef/InvertedGuideACID&Deactivate/iter4096.pkl" --guide-path "ToyExample/08_GuideAsACIDRef/Ref/iter0512.pkl" --logging "ToyExample/08_GuideAsACIDRef/log_inverted_guide_acid_and_deactivate.txt"