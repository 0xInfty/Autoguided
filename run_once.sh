ref="/00_PreTrained"; subdir="/23NormalizedLogits"

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted.txt"

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --late --acid --outdir "ToyExample"$subdir"/LateACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted.txt"

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --early --acid --outdir "ToyExample"$subdir"/EarlyACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted.txt"

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --no-acid --outdir "ToyExample"$subdir"/NoACID" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl"  --invert --acid --interpol --outdir "ToyExample"$subdir"/ACIDInterpol" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDInterpol.txt"