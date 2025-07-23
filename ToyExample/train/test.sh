ref="/00_PreTrained"; subdir="/99_Test"; device=0

python ToyExample/toy_example.py train --outdir "ToyExample"$subdir --dim 32 --total-iter 128 --guidance --guide-path "ToyExample/"$ref"/Ref/iter0512.pkl" --invert --acid --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log.txt" --device $device