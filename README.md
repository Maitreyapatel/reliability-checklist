<div align="center">

![reliability-score](extras/logo-no-background.png)

<p align="center">
  <a href="http://149.169.30.58:8000/">[reliability-score documentation]</a>
  <br> <br>
</p>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

# Description

`reliability-score` is a Python framework (available via `CLI`) for reliability tests of Large Language Models.
> `reliability-score` accepts any model and dataset as input and runs the test-suit of consisting all the important metrices to guide you to decide the most reliable model for your deployments. 

**Why you might want to use it:**

<b>✅ No coding needed</b><br>
Pre-defined templates available to easily integrate your models/datasets via commandline only.

<b>✅ Bring Your own Model (BYoM)</b><br>
Your model template is missing!? We have got you covered: Checkout [BYoM]() to create your own model specific config file.

<b>✅ Bring Your own Data (BYoD)</b><br>
Your dataset template is missing!? Again we have got you covered: Checkout [BYoD]() to create your own dataset specific config file.

<b>✅ Reliability metrics</b><br>
Currently, we support various reliability metrics specific to the classification tasks:
- <b>Standard metrics:</b> Accuracy/F1/Precision/Recall
- <b>Calibration tests:</b> Expected Calibration Error (ECE), Expected Overconfidence Error (EOE)
- <b>Selective Prediction:</b> Selective Prediction Error (SPE), Risk-Coverage Curve (RCC)
- <b>🌟 Proposed new metrics:</b> Sensitivity (our very own), and Stability

## **Upcoming feature releases:**
- <b>Adversarial Attack:</b> Model in the loop discrete adversarial attacks to learn more about failures
- <b>Out-of-Distribution:</b> Support to have many relevant OOD datasets
- <b>Task Specific Augmentations:</b> Task specific augmentations to check the reliability on highly optimized test cases

## Workflow

<b>✅ Want to integrate features?</b><br>
Our easy-to-develop infrastructure allows developers to contribute any models, datasets, augmentations, and evaluation metrics seamlessly to the whole workflow.

WORKFLOW-IMAGE

# How to install

Install the package

```bash
pip install git+https://github.com/Maitreyapatel/reliability-score
```

Evaluate example model/data with default configuration

```bash
# eval on CPU
rs

# eval on GPU
rs trainer=gpu +trainer.gpus=[1,2,3]
```

Evaluate model with chosen dataset-specific experiment configuration from [reliability_score/configs/task/](reliability_score/configs/task/)

```bash
rs tasl=<task_name>
```

Specify the custom model_name as shown in following MNLI example

```bash
# if model_name is used for tokenizer as well.
rs task=mnli custom_model="bert-base-uncased-mnli"

# if model_name is different for tokenizer then
rs task=mnli custom_model="bert-base-uncased-mnli" custom_model.tokenizer.model_name="ishan/bert-base-uncased-mnli"
```

## Add custom_model config

```bash
# create config folder structure similar to reliability_score/configs/
mkdir ./configs/
mkdir ./configs/custom_model/

# run following command after creating new config file inside ./configs/custom_model/<your-config>.yaml
rs task=mnli custom_model=<your-config>
```

# 🤝 Contributing to `reliability-score`
Any kind of positive contribution is welcome! Please help us to grow by contributing to the project.

If you wish to contribute, you can work on any features/issues [listed here](https://github.com/Maitreyapatel/reliability-score/issues) or create one on your own. After adding your code, please send us a Pull Request.

> Please read [`CONTRIBUTING`](CONTRIBUTING.md) for details on our [`CODE OF CONDUCT`](CODE_OF_CONDUCT.md), and the process for submitting pull requests to us.

---

<h3 align="center">
A ⭐️ to <b>reliability-score</b> is to build the reliability of LLMs.
</h3>