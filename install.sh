# conda create -n SCID python; conda activate SCID

Linux
pip install torch torchvision torchaudio \
    huggingface_hub==0.25.2 diffusers==0.26.3 accelerate==0.27.2 \
    matplotlib click tqdm colorlog \
    pyvtools pyvdirs --no-cache

# Windows
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install huggingface_hub==0.25.2 diffusers==0.26.3 accelerate==0.27.2 \
#     matplotlib click tqdm colorlog \
#     pyvtools pyvdirs --no-cache