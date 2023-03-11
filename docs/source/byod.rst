.. _byod:

.. role:: python(code)
  :language: python
  :class: highlight

.. role:: yaml(code)
  :language: yaml
  :class: highlight

BYoD (Bring Your own Data)
===========================

reliability-score allows users to bring their own datasets by just configuring the single .yaml file.
Checkout the `reliability_score/configs/datamodule/ <https://github.com/Maitreyapatel/reliability-score/tree/release-prep/reliability_score/configs/datamodule>`_.

**Note:** reliability-score requires datasets to be on `datasets <https://huggingface.co/docs/datasets/index>`_ library.


How to specify your dataset to reliability-score for tests?
---------------------------------------------------------

Suppose, we created `sentiment140.yaml <https://github.com/Maitreyapatel/reliability-score/blob/release-prep/reliability_score/configs/datamodule/sentiment140.yaml>`_ file
and we want to use this config for reliability tests on sentiment140 dataset.
To do this, we just need to specify the name of the config in cli and we are done:

.. code-block:: shell

    rs task=mnli datamodule=sentiment140

Pre-defined list of templates:
------------------------------

#. `mnli <https://github.com/Maitreyapatel/reliability-score/blob/release-prep/reliability_score/configs/datamodule/mnli.yaml>`_
#. `sentiment140 <https://github.com/Maitreyapatel/reliability-score/blob/release-prep/reliability_score/configs/datamodule/sentiment140.yaml>`_


If you have a new dataset and if it fits with any of the above templates then you can either specify either model_name or model_path on cli as shown below:

.. code-block:: shell

    rs task=mnli datamodule=mnli datamodule.dataset_specific_args.name="LysandreJik/glue-mnli-train" datamodule.dataset_specific_args.split="validation"

How to create template from scratch?
------------------------------------

reliability-score allows various configurations of datasets.
Following example shows how the standard template looks-like:

.. code-block:: yaml

    _target_: reliability_score.datamodules.common_datamodule.GeneralDataModule
    data_dir: ${paths.data_dir}
    batch_size: 1
    num_workers: 0
    pin_memory: False
    tokenizer_data: ${custom_model.tokenizer}
    model_type: ${custom_model.model_type}
    data_processing: ${custom_model.data_processing}
    dataset_specific_args:
        label_conversion:
            0: 0
            2: 1
            4: 2
        label2id:
            0: "negative"
            1: "neutral"
            2: "positive"
        cols: ["text"]
        name: sentiment140
        split: test
        remove_cols: ["query", "user", "date"]
        label_col: "sentiment"

As shown above yaml files for datamodule contains various parameters which allows different level of flexibility without touching the source-code.
Below we explain each of the parameters in details.

Level-1 set of parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^

* :yaml:`_target_`: str: path to the datamodule class: **DO NOT CHANGE**
* :yaml:`data_dir`: str: `${paths.data_dir}`: **DO NOT CHANGE**
* :yaml:`batch_size`: int: default batch size for the inference
    * default: keep default to :yaml:`1`
* :yaml:`num_workers`: int: provide the number of workers to use for dataloader.
    * default: keep default to :yaml:`0`
* :yaml:`pin_memory`: bool: pass true if you want to do caching
    * default: keep default to :yaml:`false`
* :yaml:`tokenizer_data`: **DO NOT CHANGE**
* :yaml:`model_type`: **DO NOT CHANGE**
* :yaml:`data_processing`: **DO NOT CHANGE**
* :yaml:`dataset_specific_args`: dict: define the important aspects to pre-process the dataset

Level-2 set of parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^

**dataset_specific_args**:

This is a great example of restricted additional input arguments. Datasets vary a lot based on how they are stored.
However, we have defined general classification processing pipeline to help you ease the pain of adding new datasets:

.. code-block:: yaml

    dataset_specific_args:
        label_conversion:
            0: 0
            2: 1
            4: 2
        label2id:
            0: "negative"
            1: "neutral"
            2: "positive"
        cols: ["text"]
        name: sentiment140
        split: test
        remove_cols: ["query", "user", "date"]
        label_col: "sentiment"



* :yaml:`label_conversion`: dict: provide the mapping to convert the labels into ordered int
    * default: keep default to :yaml:`null`
    * Here, you can provide any number of valid conversions. We only ask you to keep this conversions to integer and start it with 0 and only do +1 for classification tasks.
* :yaml:`label2id`: dict: After processing labels, let's define the corresponding class to represent the evaluation results.
* :yaml:`cols`: list: provide the list of feature columns to use them as input data
    * default: keep default to :yaml:`1`
* :yaml:`name`: str: provide the name of dataset from huggingface spaces.
* :yaml:`split`: str: provide the specific split to use for evals (either test or validation)
    * default: keep default to :yaml:`false`
* :yaml:`remove_cols`: list: provide the list of unnecessary feature column names which we can remove
* :yaml:`label_col`: str: provide the name of the target label column



Where to store new templates?
------------------------------------

Create the following folder inside your project director:

.. code-block:: shell

    # create config folder structure similar to reliability_score/configs/
    mkdir ./configs/
    mkdir ./configs/datamodule/

    # run following command after creating new config file inside ./configs/custom_model/<your-config>.yaml
    rs task=mnli datamodule=<your-config>
