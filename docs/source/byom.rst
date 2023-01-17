.. _byom:

BYoM (Bring Your own Model)
===========================

reliability-score allows users to bring their own pre-trained models by just configuring the single .yaml file.
Checkout the `configs/custom_models/ <https://github.com/Maitreyapatel/reliability-score/tree/develop/configs/custom_model>`_.

**Note:** reliability-score requires models to be created using `transformers <https://huggingface.co/docs/transformers/index>`_ library.


How to specify your model to reliability-score for tests?
---------------------------------------------------------

Suppose, we created `roberta_large_mnli.yaml <https://github.com/Maitreyapatel/reliability-score/blob/develop/configs/custom_model/roberta_large_mnli.yaml>`_ file
and we want to use this config for reliability tests on MNLI dataset.
To do this, we just need to specify the name of the config in cli and we are done:

.. code-block:: shell

    python src/eval.py experiment=mnli custom_model=roberta_large_mnli

Pre-defined list of templates:
------------------------------

#. `bert_base_uncased <https://github.com/Maitreyapatel/reliability-score/blob/develop/configs/custom_model/bert_base_uncased.yaml>`_
#. `roberta_large_mnli <https://github.com/Maitreyapatel/reliability-score/blob/develop/configs/custom_model/roberta_large_mnli.yaml>`_
#. `mt5-large-finetuned-mnli-xtreme-xnli <https://github.com/Maitreyapatel/reliability-score/blob/develop/configs/custom_model/mt5-large-finetuned-mnli-xtreme-xnli.yaml>`_
#. `zero-shot-classification <https://github.com/Maitreyapatel/reliability-score/blob/develop/configs/custom_model/zero-shot-classification.yaml>`_


If you have a custom trained model and if it fits with any of the above templates then you can either specify either model_name or model_path on cli as shown below:

.. code-block:: shell

    python src/eval.py experiment=mnli custom_model=roberta_large_mnli custom_model.model_name=bert_base_uncased

    python src/eval.py experiment=mnli custom_model=roberta_large_mnli custom_model.model_name=bert_base_uncased custom_model.model_path=</path/to/your/model/>

How to create template from scratch?
------------------------------------

reliability-score allows various configurations of models including prompt/instruction enginerring.
Following example shows how the standard template looks-like:

.. code-block:: yaml

    model_name: "roberta-large-mnli"
    model_type: "discriminative" 
    decoder_model_name: null
    model_path: null 
    tie_embeddings: false
    label: null
    tie_encoder_decoder: false
    pipeline: null

    additional_model_inputs: null 

    tokenizer:
        model_name: ${..model_name} ## modify this only if tokenizer is a different then the model
        label2id: 
            contradiction: 0
            neutral: 1
            entailment: 2
        args:
            truncation: true

    data_processing:
        header: null 
        footer: null 
        separator: " [SEP] " 
        columns:
            null


