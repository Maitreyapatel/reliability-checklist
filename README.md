<div align="center">

<img src="extras/logo-no-background.png" width="50%">

<p align="center">
  <a href="#">[documentation]</a>
  <br> <br>
</p>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

# Description

`reliability-checklist` is a Python framework (available via `CLI`) for reliability tests of Large Language Models.

> `reliability-checklist` accepts any model and dataset as input and runs the test-suit of consisting all the important metrices to guide you to decide the most reliable model for your deployments.

**Why you might want to use it:**

<b>✅ No coding needed</b><br>
Pre-defined templates available to easily integrate your models/datasets via commandline only.

<b>✅ Bring Your own Model (BYoM)</b><br>
Your model template is missing!? We have got you covered: Checkout [BYoM](<>) to create your own model specific config file.

<b>✅ Bring Your own Data (BYoD)</b><br>
Your dataset template is missing!? Again we have got you covered: Checkout [BYoD](<>) to create your own dataset specific config file.

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

<b>✅ Want to integrate more features?</b><br>
Our easy-to-develop infrastructure allows developers to contribute any models, datasets, augmentations, and evaluation metrics seamlessly to the workflow.

![workflow](extras/recheck_workflow.jpg)

# How to install?

```bash
pip install git+https://github.com/Maitreyapatel/reliability-checklist

python -m spacy download en_core_web_sm
python -c "import nltk;nltk.download('wordnet')"
```

# How to use?

Evaluate example model/data with default configuration

```bash
# eval on CPU
recheck

# eval on GPU
recheck trainer=gpu +trainer.gpus=[1,2,3]
```

Evaluate model with chosen dataset-specific experiment configuration from [reliability_checklist/configs/task/](reliability_checklist/configs/task/)

```bash
recheck tasl=<task_name>
```

Specify the custom model_name as shown in following MNLI example

```bash
# if model_name is used for tokenizer as well.
recheck task=mnli custom_model="bert-base-uncased-mnli"

# if model_name is different for tokenizer then
recheck task=mnli custom_model="bert-base-uncased-mnli" custom_model.tokenizer.model_name="ishan/bert-base-uncased-mnli"
```

## Add custom_model config

```bash
# create config folder structure similar to reliability_checklist/configs/
mkdir ./configs/
mkdir ./configs/custom_model/

# run following command after creating new config file inside ./configs/custom_model/<your-config>.yaml
recheck task=mnli custom_model=<your-config>
```

# 🤝 Contributing to `reliability-checklist`

Any kind of positive contribution is welcome! Please help us to grow by contributing to the project.

If you wish to contribute, you can work on any features/issues [listed here](https://github.com/Maitreyapatel/reliability-checklist/issues) or create one on your own. After adding your code, please send us a Pull Request.

> Please read [`CONTRIBUTING`](CONTRIBUTING.md) for details on our [`CODE OF CONDUCT`](CODE_OF_CONDUCT.md), and the process for submitting pull requests to us.

______________________________________________________________________

<h1 align="center">
A ⭐️ to <b>reliability-checklist</b> is to build the reliability of LLMs.
</h1>
