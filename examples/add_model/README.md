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

Assuming that you know how to create the new model config and suppose that we have created a new config file `mt5_custom.yaml` for the MT5 dataset by following the steps from here.

Steps:

1. Create the configs folder to store the any new configurations.
```bash
mkdir configs
```

2. Put your `mt5_custom.yaml` inside the `./configs` folder.

3. Run following command to evalaute your model on this new dataset:
```bash
recheck task=boolq custom_model=mt5_custom 'hydra.searchpath=[file://./configs/]'
```