ref="/00_PreTrained"; subdir="/28_WandB"; device=0

# No ACID, for comparison
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --outdir "ToyExample"$subdir"/Baseline" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Baseline/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --interpol --outdir "ToyExample"$subdir"/pBaseline" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/pBaseline/log.txt" --device $device

# ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/iAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/iAJEST/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/AJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/AJEST/log.txt" --device $device

# Random baseline
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --selection --outdir "ToyExample"$subdir"/Random" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Random/log.txt" --device $device

# Early ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --early --acid --outdir "ToyExample"$subdir"/Early_iAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Early_iAJEST/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/Early_AJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Early_AJEST/log.txt" --device $device

# Late ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --late --acid --outdir "ToyExample"$subdir"/Late_iAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Late_iAJEST/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/Late_AJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Late_AJEST/log.txt" --device $device

# Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/piAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/piAJEST/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/pAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/pAJEST/log.txt" --device $device

# Early Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/Early_piAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Early_piAJEST/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/Early_pAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Early_pAJEST/log.txt" --device $device

# Late Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/Late_piAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Late_piAJEST/log.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/Late_pAJEST" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/Late_pAJEST/log.txt" --device $device