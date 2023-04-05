# Define your own recipe

When you have a fix set of evaluation strategy then you can define this collection as **recipe**.
You can use this recipe in your deployment pipeline without any efforts.

Before going through the steps of defining a new recipe, we highly suggest to go through other examples on how to add different parts such as augmentation/dataset/model/metrics.

## Install `reliability-checklist`
```bash
pip install git+https://github.com/Maitreyapatel/reliability-checklist

python -m spacy download en_core_web_sm
python -c "import nltk;nltk.download('wordnet')"
```