# SCID
Synthetic Curation as Implicit Distillation

## Installation
1. Create a Python environment, using a command such as...

    ```conda create -n scid python```

2. Install PyTorch, using a command from the [official website](https://pytorch.org/get-started/locally/) such as...
    
    ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

3. Verify you have GPU support, by running...

    ```python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"```