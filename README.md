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

A suite of reliability tests for NLP models.

## How to install

Install the package

```bash
pip install git+https://github.com/Maitreyapatel/reliability-score
```

Evaluate example model/data with default configuration

```bash
# eval on CPU
rs

# eval on GPU
rs trainer=gpu
```

Evaluate model with chosen dataset-specific experiment configuration from [reliability_score/configs/experiment/](reliability_score/configs/experiment/)

```bash
rs experiment=<experiment_name>
```

Specify the custom model_name as shown in following MNLI example

```bash
# if model_name is used for tokenizer as well.
rs experiment=mnli custom_model="bert-base-uncased-mnli"

# if model_name is different for tokenizer then
rs experiment=mnli custom_model="bert-base-uncased-mnli" custom_model.tokenizer.model_name="ishan/bert-base-uncased-mnli"
```

## add custom_model config
```bash
# create config folder structure similat to reliability_score/configs/
mkdir ./configs/
mkdir ./configs/custom_model/

# run following command after creating new config file inside ./configs/custom_model/<your-config>.yaml
rs experiment=mnli custom_model=<your-config>
```

## Documentation:

The locally hosted documentation can be found at: [LINK](http://149.169.30.58:8000/)
