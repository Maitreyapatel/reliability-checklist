.. _byom:

.. role:: python(code)
  :language: python
  :class: highlight

.. role:: yaml(code)
  :language: yaml
  :class: highlight

BYoM (Bring Your own Model)
===========================

reliability-checklist allows users to bring their own pre-trained models by just configuring the single .yaml file.
Checkout the `reliability_checklist/configs/custom_models/ <https://github.com/Maitreyapatel/reliability-checklist/tree/release-prep/reliability_checklist/configs/custom_model>`_.

**Note:** reliability-checklist requires models to be created using `transformers <https://huggingface.co/docs/transformers/index>`_ library.


How to specify your model to reliability-checklist for tests?
---------------------------------------------------------

Suppose, we created `roberta_large_mnli.yaml <https://github.com/Maitreyapatel/reliability-checklist/tree/release-prep/reliability_checklist/configs/custom_model/roberta_large_mnli.yaml>`_ file
and we want to use this config for reliability tests on MNLI dataset.
To do this, we just need to specify the name of the config in cli and we are done:

.. code-block:: shell

    recheck task=mnli custom_model=roberta_large_mnli

Pre-defined list of templates:
------------------------------

#. `bert_base_uncased <https://github.com/Maitreyapatel/reliability-checklist/tree/release-prep/reliability_checklist/configs/custom_model/bert_base_uncased.yaml>`_
#. `roberta_large_mnli <https://github.com/Maitreyapatel/reliability-checklist/tree/release-prep/reliability_checklist/configs/custom_model/roberta_large_mnli.yaml>`_
#. `mt5-large-finetuned-mnli-xtreme-xnli <https://github.com/Maitreyapatel/reliability-checklist/tree/release-prep/reliability_checklist/configs/custom_model/mt5-large-finetuned-mnli-xtreme-xnli.yaml>`_
#. `zero-shot-classification <https://github.com/Maitreyapatel/reliability-checklist/tree/release-prep/reliability_checklist/configs/custom_model/zero-shot-classification.yaml>`_


If you have a custom trained model and if it fits with any of the above templates then you can either specify either model_name or model_path on cli as shown below:

.. code-block:: shell

    recheck task=mnli custom_model=roberta_large_mnli custom_model.model_name=bert_base_uncased

    recheck task=mnli custom_model=roberta_large_mnli custom_model.model_name=bert_base_uncased custom_model.model_path=</path/to/your/model/>

How to create template from scratch?
------------------------------------

reliability-checklist allows various configurations of models including prompt/instruction enginerring.
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


As shown above yaml files for custom_model contains various parameters which allows different level of flexibility without touching the source-code.
Below we explain each of the parameters in details.

Level-1 set of parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^

* :yaml:`model_name`: str: give model name from huggingface spaces
    * example: `navteca/bart-large-mnli <https://huggingface.co/navteca/bart-large-mnli>`_
* :yaml:`model_type`: str: provide the type of the model
    * choices: :python:`["encode-decode","decoder-only","bart","discriminative","shared","hybrid","t5"]`
    * BERT/RoBERTa are 'discriminative' models, while MT5 is T5 based model which works as discriminative model for MNLI dataset.
    * Similarly, :yaml:`pipeline=zero-shot-classification` is discriminative type even if the base :yaml:`model_name` contains generative model (given that transformers supports this).
* :yaml:`decoder_model_name`: str: provide the decoder model name if it is different than the :yaml:`mode_name`
    * default: keep default to :yaml:`null`
* :yaml:`model_path`: str: provide the path to the custom-trained model on local.
    * default: keep default to :yaml:`null`
* :yaml:`tie_embeddings`: bool: feature in progress
    * default: keep default to :yaml:`false`
* :yaml:`label`: feature in progress
    * default: keep default to :yaml:`null`
* :yaml:`tie_encoder_decoder`: bool: feature in progress
    * default: keep default to :yaml:`false`
* :yaml:`pipeline`: support of different huggingface pipelines
    * choices: :python:`["zero-shot-classification"]`
    * default: keep default to :yaml:`null`
* :yaml:`additional_model_inputs`: dict: define the additional fixed inputs used while inference
    * default: :yaml:`null`
    * example: generative model uses different inputs such as :python:`num_beams=1`
    * this is a level-2 type parameter
* :yaml:`tokenizer`: dict: define tokenizer specific arguments
    * this is a level-2 type parameter
* :yaml:`data_processing`: dict: define the custom data pre-processing steps.
    * you can use this for prompt/instruction enginerring
    * this is a level-2 type parameter


Level-2 set of parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^

**additional_model_inputs**:

This is a great example of unrestricted additional input arguments. Model like BERT/RoBERTa do not require any extra arguments apart from the :python:`**inputs` which is direct output from the tokenizer.
However, models like T5 will require the extra input arguments and that can be defined as:

.. code-block:: yaml

    additional_model_inputs:
        output_scores: true
        return_dict_in_generate: true
        num_beams: 1

Similarly, if you are using :yaml:`pipeline` then it also takes additional arguments such as:

.. code-block:: yaml

    additional_model_inputs:
        multi_label: false


**tokenizer**:

Tokenization can vary a lot based on the selected model or even the data.
It is important to define the proper mapping between your trained version vs the reliability-checklist requirements.
:yaml:`tokenizer` parameter contains the several reuqired parameters and again some unrestricted set of parameters:

* :yaml:`model_name`: str: define the name of the tokenizer name
    * default: keep the default to :yaml:`{..model_name}` if you are not using different tokenizer else provide the string of the tokenizer_name from the huggingface.
* :yaml:`label2id`: dict: this is the most important part of the tokenizer, as :yaml:`label2id` within :python:`model.config` form the transformer might assume different ground truth labels
    * For example, MNLI dataset contains three classes: entailment, contradiction, and neutral. Hence, define this mapping.
    * **Note:** Please refer to your selected dataset.
    * Consider the below snippet for sample:

.. code-block:: yaml

    label2id:
        contradiction: 0
        neutral: 1
        entailment: 2

* :yaml:`args`: dict: define the unrestricted set of arguments for the tokenizer from huggingface.
    * For example, it can contain :python:`max_len:512`, :python:`truncation:false` or any other custom arguments.

The final :yaml:`tokenizer` level-2 config looks like:

.. code-block:: yaml

    tokenizer:
        model_name: ${..model_name}
        label2id:
            contradiction: 0
            neutral: 1
            entailment: 2
        args:
            truncation: true


**data_processing:**

This is by far the most important and latest feature which should be carefully defined.
Suppose your model is trained using prompt enginerring or instruction learning. And in these cases it is important to define the prompts/instructions.
At the same time, some models do not require any of these like BERT/RoBERTa and in this case we can ignore these parameters except for the :yaml:`separator`.

* :yaml:`header`: str: define the global instruction
    * default: keep the default to :yaml:`{null}` if you are not using any instruction.
* :yaml:`footer`: str: define the signal to signal the model to generate
    * default: keep the default to :yaml:`{null}` if you are not using any instruction.
* :yaml:`separator`: str: define the separator string depending on your model for mixing the different columns of the dataset such as premise and hypothesis
    * For BERT/RoBERTa: :yaml:`separator=" [SEP] "`
    * For generative model: :yaml:`separator="\n"`
* :yaml:`columns`: dict: this requires the good level of understanding of the dataset being used
    * default: keep the default to :yaml:`null` if your are not using prompting.
    * Else define the prefix string for each column in the dataset.
    * consider the following code snippet for the MT5 prompt enginerring based model:


.. code-block:: yaml

    data_processing:
        header: null
        footer: null
        separator: " "
        columns:
            premise: "xnli: premise:"
            hypothesis: "hypothesis:"

Where to store new templates?
------------------------------------

Create the following folder inside your project director:

.. code-block:: shell

    # create config folder structure similar to reliability_checklist/configs/
    mkdir ./configs/
    mkdir ./configs/custom_model/

    # run following command after creating new config file inside ./configs/custom_model/<your-config>.yaml
    recheck task=mnli custom_model=<your-config>
