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
- Any datasets and pre-trained models will be stored inside `data_home`
- Any training model checkpoints will be stored inside `models_home`
- Any results will be stored inside `results_home`

You can manually open the file and change the `system_name` attribute to create your own nickname, to know which one is your entry. You can also manually modify each of the other path variables.

## Toy example implementation

### Running autoguidance on the toy example

This repository contains a working adaptation of the 2D tree toy example from Karras et al's ["Guiding a diffusion model with a bad version of itself"](https://arxiv.org/abs/2406.02507).

Pre-trained toy models can be automatically downloaded and tested running...

```
python ToyExample/toy_example.py plot
```

New toy models can be trained and visualized running...

```
python ToyExample/toy_example.py train
```

### Training the toy model with AJEST

This repository now contains an implementation of JEST as described by Evans, Parthasarathy et al. on ["Guiding a diffusion model with a bad version of itself"](https://arxiv.org/abs/2406.17711).

A toy model can be trained using autoguided JEST (AJEST) running...

```
python ToyExample/toy_example.py train --acid
```

## Images implementation

### Running autoguidance on images

Pre-trained models can be automatically downloaded and tested to generate ImageNet-like images by running...

```
python Images/generate_images.py --preset=edm2-img512-s-autog-dino --outdir="images_test"
```

The preset configuration will determine which models and guidance weight to use according to Karras et al's results. In general...
- "autog" stands for autoguidance
- "guid" stands for classifier-free guidance

## Additional information

### Authors

[Valeria Pais Malacalza](v.pais-malacalza.1@research.gla.ac.uk) from University of Glasgow, Glasgow, United Kingdom.

[Marco Aversa](marco.aversa@outlook.com) associated at the time to Dotphoton, Zug, Switzerland.

[Luis Oala](luis.oala@dotphoton.com) associated at the time to Dotphoton, Zug, Switzerland.

### License

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

Code adapted from Karras' repository was originally under the same [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license. That code was originally marked with the following copyright disclaimer: _"Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved"_. The disclaimer has been kept in all those files.

Another part of the code is an implementation described by Evans, Parthasarathy et al in a publication under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

A history of changes for all code can be extracted using Git's version control tools.

### Acknowledgments

We thank T. Karras et al for sharing their ["Guiding a diffusion model with a bad version of itself"](https://arxiv.org/abs/2406.02507) research and ["EDM2 and Autoguidance"](https://github.com/NVlabs/edm2) code, licensed under CC BY-NC-SA 4.0.