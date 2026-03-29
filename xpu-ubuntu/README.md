# Native XPU Ubuntu 24.04 LTS

I find this has the best performance, albeit only slightly better than the containers. However between containers and venv/conda I strongly prefer creating a `uv venv` over containers. 

### Steps to install

Kernel drivers exist already but the system will not have all the specific oneapi and mkl-sycl drivers that can further accelerate Arc GPUs in pytorch.

Have Ubuntu up to date. These instructions will create a python venv using `uv`, if you want to follow along you must install `uv`: Use `curl -LsSf https://astral.sh/uv/install.sh | sh`.

`uv` will allow us to create simple venv's but also allow us to have the benefit of a separate python installation than our base system. It has many other features but I like to think of it as having conda like features in a simple venv package.

Create a folder for all projects/things:

```bash
mkdir ~/Venvs
```

Create a folder for this specific test/project:

```bash
mkdir ~/Venvs/xpubase
cd ~/Venvs/xpubase
```

Create the python virtual environment using `uv` and activate the environment:

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
```

Install the required Intel packages:

<a href="https://pytorch-extension.intel.com/installation?request=platform">Intel® Extension for PyTorch* Installation Guide</a>

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu

# If the ipex install fails (seems to happen due to a FE website issue) use the whl package directly with the line below
# uv pip install https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-2.8.10%2Bxpu-cp313-cp313-linux_x86_64.whl
uv pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

uv pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

This is the base installation complete, please be aware the repository for these packages can sometimes be extremely slow. 

Validate the base installation: 

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

You should see a line for xpu 2.8 and ipex 2.8 along with a `true` meaning xpu is available via pytorch.

Install additional packages:

```bash
uv pip install jupyterlab accelerate diffusers tqdm IProgress transformers scikit-learn matplotlib numpy pandas safetensors ipywidgets
```

This will be about all the packages needed for all basic machine learning in Pytorch. 


