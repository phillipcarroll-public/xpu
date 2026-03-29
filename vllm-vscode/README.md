# Goal: Setup vLLM to serve models to VS Code

Things we need:

- VS Code
- Continue extension
- vLLM
- openVINO
- Model
- Arc GPU (B580) and related drivers
- oneAPI base toolkit
- Python
- uv


## Install oneAPI base toolkit

```bash
sudo apt update -y
sudo apt install cmake pkg-config build-essential -y
# Verify
which cmake pkg-config make gcc g++
```

Edit your GRUB

```bash
sudo vim /etc/default/grub
```

Look for the line: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"`

We need to add `i915.enable_hangcheck=0` at the end after `quiet splash`

The line should look like:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash i915.enable_hangcheck=0"
```

Save and exit

Update grub and reboot

```bash
sudo update-grub
sudo reboot
```

Verification

We should see "i915.enable_hangcheck=0"

```bash
cat /proc/cmdline
```

```bash
$ cat /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-6.17.0-19-generic root=UUID=~~~~~~ ro quiet splash i915.enable_hangcheck=0 vt.handoff=7
```

Install oneAPI with the online installer: <a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=online">Intel® oneAPI Base Toolkit</a>

Download the script and execute. 

```bash
$ bash intel-oneapi-base-toolkit-2025.3.1.36.sh
```

This will popup a gui based installation, follow the prompts.

Verify Installation, in my case it was installed in my home dir under `intel`

Set the environment vars:

```bash
. ~/intel/oneapi/oneapi-vars.sh
```

You should see output similar to:

```bash
:: initializing oneAPI environment ...
   bash: BASH_VERSION = 5.2.21(1)-release
   args: Using "$@" for oneapi-vars.sh arguments: 
:: advisor -- processing etc/advisor/vars.sh
:: ccl -- processing etc/ccl/vars.sh
:: compiler -- processing etc/compiler/vars.sh
:: dal -- processing etc/dal/vars.sh
:: debugger -- processing etc/debugger/vars.sh
:: dnnl -- processing etc/dnnl/vars.sh
:: dpct -- processing etc/dpct/vars.sh
:: dpl -- processing etc/dpl/vars.sh
:: ipp -- processing etc/ipp/vars.sh
:: ippcp -- processing etc/ippcp/vars.sh
:: mkl -- processing etc/mkl/vars.sh
:: mpi -- processing etc/mpi/vars.sh
:: tbb -- processing etc/tbb/vars.sh
:: vtune -- processing etc/vtune/vars.sh
:: oneAPI environment initialized ::
```

Launch the oneAPI samples browser: `oneapi-cli`

Create a project -> cpp -> Base:Vector Add -> Create

This will create a folder where you specified. Lets compile the cpp.

`cd` to `vector-add/src`

We should have the required compilers and tools installed:

```bash
# dpcpp is deprecated use icpx
#dpcpp vector-add-buffers.cpp -o vector_add_buff_app

icpx -fsycl vector-add-buffers.cpp -o vector_add_buff_app
```

You should see similar to the following output: 

```bash
$ ./vector_add_buff_app 
Running on device: Intel(R) Arc(TM) B580 Graphics
Vector size: 10000
[0]: 0 + 0 = 0
[1]: 1 + 1 = 2
[2]: 2 + 2 = 4
...
[9999]: 9999 + 9999 = 19998
Vector add successfully completed on device.
```

## Setup your uv venv Environment

I keep my uv venvs / projects in a directory: `~/Venvs` you can put them where you like but I will reference this folder.

Create a new project dir to home the venv: `mkdir vllm-vscode`

cd to the dir: `cd vllm-vscode`

Create a venv with uv for python 3.13: `uv venv .vllm-intel --python 3.13`

Activate the venv: `source .vllm-intel/bin/activate`

Update pip: `uv pip install --upgrade pip`

These can take a very long time to install, be patient. 

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu

# If the ipex install fails (seems to happen due to a FE website issue) use the whl package directly with the line below
# uv pip install https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-2.8.10%2Bxpu-cp313-cp313-linux_x86_64.whl
uv pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

uv pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Validate the installation: 

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

If you run into errors regarding being unable to import `...Venvs/vllm-vscode/.vllm-intel/lib/python3.13/site-packages/torch/lib/../../../../libsycl.so.8`

You may need to export the correct path from your venv: `export LD_LIBRARY_PATH=$HOME/Venvs/vllm-vscode/.vllm-intel/lib:$LD_LIBRARY_PATH`

Rerun the python validation:

You should see similar to: 

```bash
True
2.8.0+xpu
2.8.10+xpu
```

## Install vLLM

Clone the repo: `git clone --depth 1 https://github.com/vllm-project/vllm.git`

**Note:** `--depth 1` will just pull the latest commit and not the entire life of the project

Change to the dir: `cd vllm`

Install the following:

```bash
uv pip install -r requirements/common.txt
uv pip install setuptools wheel cmake ninja pybind11
uv pip install -r requirements/xpu.txt
#uv pip install cmake>=3.26.1 wheel packaging ninja setuptools-scm
```

Build vLLM targeting XPU: `VLLM_TARGET_DEVICE=xpu uv pip install --no-deps -v -e .`

**Note:** The first time this may take an extremely long time to download and resolve depedencies.

## Start the vLLM server

