# Use any pre-trained model

To add a new model for evaluation, we just need to define the configuration files for this model.

Note: To learn on how to define the new model config, please refer to the documentation here.

## Install `reliability-checklist`

```bash
pip install git+https://github.com/Maitreyapatel/reliability-checklist

python -m spacy download en_core_web_sm
python -c "import nltk;nltk.download('wordnet')"
```

## Steps to follow

Assuming that you know how to create the new model config and suppose that we have created a new config file `distillbert_base.yaml` for the existing `mnli` dataset by following the steps from here.

1. Create the configs folder to store the any new configurations.

```bash
mkdir -p configs/custom_models
```

2. Put your `distillbert_base.yaml` inside the `./configs/custom_models` folder.

3. Copy existing parrot augmentation file inside `data` folder to save the time.

4. Run following command to evalaute your model on this new dataset:

```bash
recheck task=mnli custom_model=distillbert_base 'hydra.searchpath=[file://./configs/]'
```
