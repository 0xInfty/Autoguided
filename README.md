# Autoguided Online Data Curation 
for Diffusion Model Training

## Getting Started

### Installation

1. Create and activate an Anaconda environment with Python 3.12

    ```bash
    conda create -n SCID python=3.12
    conda activate SCID
    ```

2. Install all required packages using the installation script
    
    ```bash
    yes | . install.sh
    ```

3. Verify you have GPU support, by running...

    ```bash
    python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    ```

### Managing directories

PyVDirs can manage your directories in such a way that allows you to work from multiple PCs without modifying the code at all.

For PyVDirs to work, you will need to create a new entry inside the `dirs.json` index. This can be done automatically, just by importing the module:

```python
import pyvdirs.dirs as dirs
```

The new entry should look like...

```json
 "your-PC-user-identifier": 
    {"system_name": "UKN", 
     "system_home": "path/to/code/FilteringNSD", 
     "data_home": "path/to/code/FilteringNSD/data", 
     "models_home": "path/to/code/FilteringNSD/models", 
     "results_home": "path/to/code/FilteringNSD/results"}, 
```

Following this index...
- Any datasets and pre-trained models will be stored inside `data_home`, accessible as `dirs.DATA_HOME`
- Any training model checkpoints will be stored inside `models_home`, accessible as `dirs.MODELS_HOME`
- Any results will be stored inside `results_home`, , accessible as `dirs.MODELS_HOME`

You can manually open the file and change the `system_name` attribute to create your own nickname, to know which one is your entry. You can also manually modify each of the other path variables.

## Data selection methods

All available data selection methods are coded in the `ours.selection` module.

- `ours.selection.jointly_sample_batch(learner_loss, ref_loss, **kwargs)` is a working implementation of JEST as described by Evans et al. on ["Data curation via joint example selection further accelerates multimodal learning"](https://dl.acm.org/doi/10.5555/3737916.3742401)
- `ours.selection.random_baseline(learner_loss, **kwargs)` is a compatible implementation of random data selection.

To add any other method that works on the learner's loss, or on both the learner's and the reference's loss, you just need to define a new function inside the module:

```
def new_selection_method(learner_loss, ref_loss, ..., **kwargs):
    # Do something to get the selected indices
    return indices
```

Make the first positional argument the learner's loss. If a reference's loss is needed, then make it the second positional argument and add the function name into the `REQUIRES_REF_LOSS` list. 

The output of the function needs to be an array of indices. For super-batch size $B$ and mini-batch size $b$, the input loss will have length $B$ and the ouput indices $[i_1, i_2, ...]$ shall have length $B$ and elements such that $i_n < B, \, n \leq B$.

If you use our code to implement any other data selection methods, we'd love to hear about it! Feel free to fork our repository and ask for a pull request.

## 2D tree toy example implementation

### Running autoguidance on the toy example

This repository contains a working adaptation of the 2D tree toy example from Karras et al's ["Guiding a diffusion model with a bad version of itself"](https://dl.acm.org/doi/10.5555/3737916.3739595).

Pre-trained toy models can be automatically downloaded and tested running...

```
python ToyExample/toy_example.py plot
```

New toy models can be trained and visualized running...

```
python ToyExample/toy_example.py train
```

### Training the toy model with AJEST or random data selection

A small toy reference model can be trained running...

```
python ToyExample/toy_example.py train --dim 32 --total-iter 512 --outdir "ToyExample/Ref"
```

A larger toy model can be trained using autoguided JEST (AJEST) running...

```
python ToyExample/toy_example.py train --acid --guidance --guide-path "ToyExample/Ref/iter0512.pkl"
```

Alternatively, a toy model can be trained with random data selection running...

```
python ToyExample/toy_example.py train --selection
```

A baseline model with no data selection can be trained running...

```
python ToyExample/toy_example.py train
```

Validation metrics can be calculated during training, stored in a .txt log file. Both training loss and validation metrics can be recovered from this file with the `extract_results_from_log` and `plot_loss` auxiliary functions from `toy_example.py`.

### Testing toy models with our suite of metrics

Regardless of the training method, all toy models trained on the 2D tree task can be tested by running...

```
python ToyExample/toy_example.py test --net-path "relative/path/iter4096.pkl" --guide-path "relative/path/iter0512.pkl"
```

Metrics without guidance are always calculated. Metrics using `guidance_weight=3` as in Karras et al. will also be calculated whenever a guide is specified. If you want to test both the learner and EMA models, you can also include the path to the EMA checkpoint using the `--ema-path` flag. All relative filepaths to models are specified with respect to `dirs.MODELS`.

Basic metrics include average loss over different noise levels and L2 distance between fully denoised samples and the ground truth. Metrics on external branches can be obtained using the `external` flag. Mandala and classification metrics can be calculated using the `mandala` flag.

## Images implementation

### Running autoguidance on images

Pre-trained models can be automatically downloaded and tested to generate ImageNet-like images by running...

```
python Images/generate_images.py --preset=edm2-img512-s-autog-dino --outdir="images_test"
```

The preset configuration will determine which models and guidance weight to use according to Karras et al's results. In general...
- "autog" stands for autoguidance
- "guid" stands for classifier-free guidance

### Training on $3\times64\times64$ Tiny ImageNet with AJEST or random data selection

An XXS EDM2 model can be trained on Tiny ImageNet to be used as a reference by running a command such as...

```
torchrun --standalone --nproc_per_node=2 Images/train_edm2.py --outdir="Ref" --dataset tiny --preset="edm2-tiny-xxs" --batch-gpu=256
```

This particular command is designed to run on 2 GPUs. The total batch size is set to 2048 by default. Setting the maximum batch per GPU to 256 as in this example, there will be 4 accummulation rounds before updating the gradient.

Once this is done, an XS EDM2 model can be trained on Tiny ImageNet with AJEST by running a command such as...

```
torchrun --standalone --nproc_per_node=2 Images/train_edm2.py --outdir="AJEST" --dataset tiny --preset="edm2-tiny-xs" --batch-gpu=128 --acid --guide-path "Ref/network-snapshot-0005000-0.100.pkl"
```

To implement Early AJEST, the `--early` flag can be used. The way checkpoints are stored, this particular example loads the reference model after 5000 epochs with EMA=0.100.

Alternatively, an XS EDM2 model can be trained on Tiny ImageNet with random data selection by running...

```
torchrun --standalone --nproc_per_node=2 Images/train_edm2.py --outdir="Random" --dataset tiny --preset="edm2-tiny-xs" --batch-gpu=128 --selection
```

A baseline model with no data selection can be trained removing both the `--selection` and `--acid` flags:

```
torchrun --standalone --nproc_per_node=2 Images/train_edm2.py --outdir="Baseline" --dataset tiny --preset="edm2-tiny-xs" --batch-gpu=128
```

### Testing EDM2 models with our suite of metrics

Any EDM2 model can be tested with our suite of metrics. 

We evaluate models with FID and FD-DINOv2 as [Karras et al](https://dl.acm.org/doi/10.5555/3737916.3739595), but we also apply a pre-trained classifier on generated images and calculate top-1 and top-5 accuracy. We use the Swin-L classifier trained by Hyun et al on ["Vision Transformers in 2022: An Update on Tiny ImageNet"](https://arxiv.org/abs/2205.10660).

FID and FD-DINOv2 metrics will require you to first calculate statistics on the dataset. For Tiny ImageNet, run...

```
python ref --dataset tiny --dest "tiny.pkl"
```

## Additional information

### Authors

[Valeria Pais Malacalza](v.pais-malacalza.1@research.gla.ac.uk) from University of Glasgow, Glasgow, United Kingdom.

[Luis Oala](luis.oala@dotphoton.com) associated at the time to Dotphoton, Zug, Switzerland.

[Daniele Faccio](daniele.faccio@glasgow.ac.uk) from University of Glasgow, Glasgow, United Kingdom.

[Marco Aversa](marco.aversa@outlook.com) associated at the time to Dotphoton, Zug, Switzerland.

### License

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

Code adapted from Karras' repository was originally under the same [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license. That code was originally marked with the following copyright disclaimer: _"Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved"_. The disclaimer has been kept in all those files.

A different part of the code is an implementation of a method described by Evans, Parthasarathy et al in a publication under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). No open-source code implementation was shared in this case, but our functions have a link to their publication in the docstrings.

We use a pre-trained Tiny ImageNet classifier obtained from Huynh et al's ["TinyImageNet-Transformers"](https://github.com/ehuynh1106/TinyImageNet-Transformers) repository. The original code and models were shared under an [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0). We have kept a link to the repository on functions involving this model.

A history of changes for all code can be extracted using Git's version control tools.

### Acknowledgments

We thank T. Karras et al for sharing their ["Guiding a diffusion model with a bad version of itself"](https://dl.acm.org/doi/10.5555/3737916.3739595) research and ["EDM2 and Autoguidance"](https://github.com/NVlabs/edm2) code and pre-trained models.

We also thank T. Evans, N. Parthasarathy et al for sharing their ["Data curation via joint example selection further accelerates multimodal learning"](https://dl.acm.org/doi/10.5555/3737916.3742401) research and a detailed description of their JEST method.

Finally, our most special thanks to E. Huynh for sharing their ["Vision Transformers in 2022: An Update on Tiny ImageNet"](https://arxiv.org/abs/2205.10660) research and their ["TinyImageNet-Transformers"](https://github.com/ehuynh1106/TinyImageNet-Transformers) code and pre-trained models.