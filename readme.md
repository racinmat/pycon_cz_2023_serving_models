# Examples of serving with some non-trivial requirements

For torchserve
```bash
cd torch_serve/win_env
conda create -n serving_demos_torch_serve python=3.10 poetry=1.6.1 cudatoolkit=11.2 cudnn=8.1.0
conda activate serving_demos_torch_serve
poetry install
gcloud auth login
gcloud auth application-default login
gcloud set project xxx
```
windows natively does not have gpu support for tensorflow > 2.10, see https://www.tensorflow.org/install/pip#step-by-step_instructions
newer versions, like `cudatoolkit=11.8.0 cudnn=8.8.0` do not work.

For some reason fetching data using my account through tf.io did not work, it was not using my credentials.

In WSL:
```bash
ubuntu2004.exe
cd /mnt/c/Projects/serving_demos
cd torch_serve
conda create -n serving_demos_torch_serve python=3.10 poetry=1.6.1
 cudatoolkit=11.2 cudnn=8.1.0
conda activate serving_demos_torch_serve
poetry install
gcloud auth login
gcloud auth application-default login
gcloud set project xxx
```
for fastapi
```bash
cd fastapi
conda create -n serving_demos_fastapi python=3.10 poetry=1.6.1
conda activate serving_demos_fastapi
poetry install
```
