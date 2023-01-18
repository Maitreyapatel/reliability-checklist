.. _tool101:

reliability-tool 101
====================

This section walks through an example of how to do reliability tests on one pre-trained model. Later, we will go over some cool tips and tricks about the tool.

Run a sample reliability tests on MNLI dataset:
-----------------------------------------------
.. code-block:: shell

    python src/eval.py experiment=mnli

It is important to note that, we can only run the experiments on the pre-defined set of datasets that are listed inside `configs/experiments/ <https://github.com/Maitreyapatel/reliability-score/tree/develop/configs/experiment>`.

Using on different devices:
---------------------------

.. code-block:: shell

    # train on CPU
    python src/eval.py

    # train on 1 GPU
    python src/eval.py trainer=gpu

    # train on 2 GPU
    python src/eval.py trainer=gpu trainer.gpus=2

    # train on 2 GPU with specific ids
    python src/eval.py trainer=gpu trainer.gpus=[1, 5]

    # train on TPU
    python src/eval.py +trainer.tpu_cores=8

    # train with DDP (Distributed Data Parallel) (4 GPUs)
    python src/eval.py trainer=ddp trainer.devices=4

    # train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
    python src/eval.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

    # simulate DDP on CPU processes
    python src/eval.py trainer=ddp_sim trainer.devices=2

    # accelerate training on mac
    python src/eval.py trainer=mps


Saving the output of the reliability tests:
-------------------------------------------


.. code-block:: shell

    python src/eval.py logger=csv



Going beyond user and configuring each experiments:
---------------------------------------------------

Please refer to the `hydra <https://hydra.cc>`_ package and `configs/ <https://github.com/Maitreyapatel/reliability-score/tree/develop/configs>`_ folder to understand the different parameters and features.
Once you understand them then you can modify them on cli (for example, how different devices are used in above example).
