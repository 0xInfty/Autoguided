# SCID
Synthetic Curation as Implicit Distillation

## Getting Started

### Installation
1. Create a Python environment, using a command such as...

    ```conda create -n scid python```

2. Install PyTorch, using a command from the [official website](https://pytorch.org/get-started/locally/) such as...
    
    ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

3. Verify you have GPU support, by running...

    ```python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"```

## Additional information

### Authors

[Valeria Pais Malacalza](v.pais-malacalza.1@research.gla.ac.uk) from University of Glasgow, Glasgow, United Kingdom.

[Marco Aversa](marco.aversa@dotphoton.com) from Dotphoton, Zug, Switzerland.

[Luis Oala](luis.oala@dotphoton.com) from Dotphoton, Zug, Switzerland.

### License

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

Part of the code was originally marked with the following disclaimer: "Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved". Those files have kept this disclaimer and a history of changes can be extracted using Git's version control tools.

### Acknowledgments

We thank T. Karras et al for sharing their ["Guiding a diffusion model with a bad version of itself"](https://arxiv.org/abs/2406.02507) research and ["EDM2 and Autoguidance"](https://github.com/NVlabs/edm2/blob/main/toy_example.py) code, licensed under CC BY-NC-SA 4.0.