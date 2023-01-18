<div align="center">

<h1> reliability-score 🎯 </h1>

<p align="center">
  <a href="http://149.169.30.58:8000/">[reliability-score documentation]</a>
  <br> <br>
</p>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A tool for performing the reliability tests on NLP models.

## How to run

Install dependencies

```bash
# clone project
git clone git@github.com:Maitreyapatel/reliability-score.git
cd reliability-score

# [OPTIONAL] create conda environment
conda create -n venv python=3.8
conda activate venv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Evaluate example model/data with default configuration

```bash
# train on CPU
python src/eval.py

# train on GPU
python src/eval.py trainer=gpu
```

Evaluate model with chosen dataset-specific experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/eval.py experiment=<experiment_name>
```

Specify the custom model_name as shown in following MNLI example

```bash
# if model_name is used for tokenizer as well.
python src/eval.py experiment=mnli custom_model="bert-base-uncased-mnli"

# if model_name is different for tokenizer then
python src/eval.py experiment=mnli custom_model="bert-base-uncased-mnli" custom_model.tokenizer.model_name="ishan/bert-base-uncased-mnli"
```

## Documentation:

The locally hosted documentation can be found at: [LINK](http://149.169.30.58:8000/)
