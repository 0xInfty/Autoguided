ref="/00_PreTrained"; subdir="/99_Once"; device=1

# No ACID, for comparison
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --outdir "ToyExample"$subdir"/Baseline" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_Baseline.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --invert --acid --interpol --outdir "ToyExample"$subdir"/ACIDInterpol" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDInterpol.txt" --device $device

# Random baseline
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --selection --outdir "ToyExample"$subdir"/Random" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_Random.txt" --device $device

# ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted.txt" --device $device

# Early ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --early --acid --outdir "ToyExample"$subdir"/EarlyACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted.txt" --device $device

# Late ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --late --acid --outdir "ToyExample"$subdir"/LateACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted.txt" --device $device

# Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/InterpolACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACID.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/InterpolACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACIDNonInverted.txt" --device $device

# Early Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACID.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACIDNonInverted.txt" --device $device

# Late Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACID.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACIDNonInverted.txt" --device $device