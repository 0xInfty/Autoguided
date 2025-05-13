ref="/00_PreTrained"; subdir=""

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --no-acid --outdir "ToyExample"$subdir"/NoACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID.txt"
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --invert --acid --interpol --outdir "ToyExample"$subdir"/ACIDInterpol" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDInterpol.txt"