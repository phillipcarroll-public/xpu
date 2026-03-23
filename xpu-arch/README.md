# Arch/Garuda XPU Setup

**Tested on Garuda Linux 6.19**

- Assuming you having installed using Mesa drivers.

```bash
sudo pacman -S intel-compute-runtime intel-media-driver level-zero-loader level-zero-headers ocl-icd
```

```bash
sudo gpasswd -a $USER render
sudo gpasswd -a $USER video
```

```bash
newgrp render
```

Either re-source your shell or simply close Konsole and re-open.

Install uv:

```bash
pip install uv
```

Re-source your shell or simply close Konsole and re-open.

Create uv venv folder/s:

```bash
mkdir ~/AI
cd ~/AI
mkdir xpu
cd xpu
```

Create a venv with a hidden folder named `.venv`:

```bash
uv venv .venv python 3.12
source .venv/bin/activate.fish
```

Install PyTorch with XPU support:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
```

Install Intel Extension for PyTorch:

```bash
uv pip pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Note: If the intel-extension does not install you may need to download it manually from: <a href="https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-2.8.10%2Bxpu-cp312-cp312-linux_x86_64.whl">https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-2.8.10%2Bxpu-cp312-cp312-linux_x86_64.whl</a>

Install other needed packages:

```bash
uv pip install jupyterlab accelerate diffusers tqdm IProgress transformers scikit-learn matplotlib pillow numpy pandas safetensors ipywidgets
```

Open Jupyter Lab:

```bash
jupyter lab
```

Running the following to check for xpu support:

```python
import torch
import intel_extension_for_pytorch as ipex

# This will return True if everything is working
print(torch.xpu.is_available())
print(torch.__version__)

# Validate ipex version
print(ipex.__version__)
```

You should see the following output:

```bash
True
2.8.0+xpu
2.8.10+xpu
```

Garuda/Arch is now setup for pytorch with xpu support.