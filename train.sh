# Activate SCID, if needed
conda activate SCID

# Simply run to test it works
# python ToyExample/toy_example.py train --outdir "ToyExample/Test" --dim 32 --total-iter 128 --guidance --guide-path "ToyExample/08_GuideAsACIDRef/Ref/iter0512.pkl" --invert --acid --seed 0 --val --test --verbose --logging "ToyExample/log_test.txt"

# Train a guide model
# python ToyExample/toy_example.py train --dim 32 --total-iter 512 --outdir "ToyExample/10_EarlyStopACID/Ref" --seed 0 --val --test --verbose --logging "ToyExample/10_EarlyStopACID/log_Ref.txt"

# Run with guidance
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --no-acid --outdir "ToyExample/10_EarlyStopACID/NoACIDGuided" --seed 0 --val --test --verbose --logging "ToyExample/10_EarlyStopACID/log_NoACIDGuided.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/10_EarlyStopACID/InvertedGuideACID" --seed 0 --val --test --verbose --logging "ToyExample/10_EarlyStopACID/log_InvertedGuideACID.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --interpol --outdir "ToyExample/10_EarlyStopACID/InvertedGuideACIDInterpolation" --seed 0 --val --test --verbose --logging "ToyExample/10_EarlyStopACID/log_InvertedGuideACIDInterpolation.txt"
. run_once.sh

# Repeat training with same seed
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/17_Repetition/ACIDWithOldRef" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_ACIDWithOldRef.txt"
# python ToyExample/toy_example.py train --dim 32 --total-iter 512 --outdir "ToyExample/17_Repetition/Ref" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_Ref.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/17_Repetition/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/17_Repetition/ACIDWithNewRef" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_ACIDWithNewRef.txt"

# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/17_Repetition/ACIDWithOldRef_02" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_ACIDWithOldRef_02.txt"
# python ToyExample/toy_example.py train --dim 32 --total-iter 512 --outdir "ToyExample/17_Repetition/Ref_02" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_Ref_02.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/17_Repetition/Ref_02/iter0512.pkl" --invert --acid --outdir "ToyExample/17_Repetition/ACIDWithNewRef_02" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_ACIDWithNewRef_02.txt"

# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/17_Repetition/ACIDWithOldRef_03" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_ACIDWithOldRef_03.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/17_Repetition/ACIDWithOldRef_NoOuter" --seed 0 --val --test --verbose --logging "ToyExample/17_Repetition/log_ACIDWithOldRef_NoOuter.txt"

# Run for different random seeds
. run_sweep_ACID.sh

# Run for different batch size
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/12_BatchSize/BatchSize2048" --seed 0 --batch-size 2048 --val --test --verbose --logging "ToyExample/12_BatchSize/log_BatchSize2048.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/12_BatchSize/BatchSize1024" --seed 0 --batch-size 1024 --val --test --verbose --logging "ToyExample/12_BatchSize/log_BatchSize1024.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/12_BatchSize/BatchSize0512" --seed 0 --batch-size 512 --val --test --verbose --logging "ToyExample/12_BatchSize/log_BatchSize0512.txt"

# Run for different batch size, but adjusting also the total number of iterations to see the same amount of data points
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/12_BatchSize/BatchSize2048Long" --seed 0 --batch-size 2048 --total-iter 8192 --val --test --verbose --logging "ToyExample/12_BatchSize/log_BatchSize2048Long.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/12_BatchSize/BatchSize1024Long" --seed 0 --batch-size 1024 --total-iter 16384 --val --test --verbose --logging "ToyExample/12_BatchSize/log_BatchSize1024Long.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample/10_EarlyStopACID/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample/12_BatchSize/BatchSize0512Long" --seed 0 --batch-size 512 --total-iter 32768 --val --test --verbose --logging "ToyExample/12_BatchSize/log_BatchSize0512Long.txt"

# Change n, compare to the default value n = 16 
# Change filter_ratio, compare to the default value filter_ratio = 0.8
. run_sweep_seed.sh

# Change learnability, compare to the default value learnability = True
# python ToyExample/toy_example.py train --no-diff --outdir "ToyExample/05_ACIDParams/No_ACID_N_16_F_0.80_L_0" --no-acid --seed 0 --verbose --logging "ToyExample/05_ACIDParams/No_ACID_N_16_F_0.80_L_0.txt"
# python ToyExample/toy_example.py train --no-diff --outdir "ToyExample/05_ACIDParams/ACID_N_16_F_0.80_L_0" --acid --seed 0 --verbose --logging "ToyExample/05_ACIDParams/ACID_N_16_F_0.80_L_0.txt"

# Run a test on a trained model
# python ToyExample/toy_example.py test --acid --seed 0 --net-path "ToyExample/08_GuideAsACIDRef/InvertedGuideACID&Deactivate/iter4096learner.pkl" --ema-path "ToyExample/08_GuideAsACIDRef/InvertedGuideACID&Deactivate/iter4096.pkl" --guide-path "ToyExample/08_GuideAsACIDRef/Ref/iter0512.pkl" --logging "ToyExample/08_GuideAsACIDRef/log_inverted_guide_acid_and_deactivate.txt"