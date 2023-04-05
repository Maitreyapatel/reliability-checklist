# Use any new dataset

To add a new dataset, we just need to define the configuration files for the dataset and modify the configuration file for the model to adapt it for the given dataset.

Note: To learn on how to define the new dataset config, please refer to the documentation here.

## Install `reliability-checklist`
```bash
pip install git+https://github.com/Maitreyapatel/reliability-checklist

python -m spacy download en_core_web_sm
python -c "import nltk;nltk.download('wordnet')"
```

## Steps to follow

Assuming that you know how to create the new dataset config and suppose that we have created a new config file for the BoolQ dataset by following the steps from here.

Steps:

1. Create the configs folder to store the any new configurations.
```bash
mkdir configs
```

2. Put your `booq.yaml` inside the `./configs` folder.

3. Modify your model configuration file to support the three class classification problem (if necessary) and put it also inside the `./configs` folder. Documentation on modifying model config can be cound here.

4. Run following command to evalaute your model on this new dataset:
```bash
recheck task=boolq datamodule=boolq 'hydra.searchpath=[file://./configs/]'
```