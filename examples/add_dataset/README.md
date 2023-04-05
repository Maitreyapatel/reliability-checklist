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

Goal: Use the `snli` dataset instead of `mnli` dataset from existing recipes.

As we already have models pre-defined for `mnli` task, we do not need to explicitly define them again.

1. Create the configs folder to store the any new configurations.

```bash
mkdir -p configs/datamodule
```

2. Put your `snli.yaml` inside the `./configs/datamodule` folder.

3. Run following command to evalaute your model on this new dataset:

```bash
recheck task=mnli datamodule=snli 'hydra.searchpath=[file://./configs/]'
```

**Note:** This will throw following error: `KeyError: "Column hypothesis_parse not in the dataset. Current columns in the dataset: ['premise', 'hypothesis', 'label', 'primary_key']"`

4. Let's remove the INV_PASS augmentation because it requires the `hypothesis_parse` from input dataset.

```bash
recheck task=mnli datamodule=snli ~augmentation.inv_augmentation 'hydra.searchpath=[file://./configs/]'
```

**Note:** This will again throw following the error: `KeyError: -1` Because of the dataset inconsistency on HuggingFace Space. But this is how you can add new dataset support. But make sure that your dataset is clean and should not have such inconsistencies as `snli` from huggingface.
