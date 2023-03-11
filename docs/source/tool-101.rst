.. _tool101:

reliability-tool 101
====================

This section walks through an example of how to do reliability tests on one pre-trained model. Later, we will go over some cool tips and tricks about the tool.

Run a sample reliability tests on MNLI dataset:
-----------------------------------------------
.. code-block:: shell

    rs task=mnli

It is important to note that, we can only run the experiments on the pre-defined set of datasets that are listed inside `configs/experiments/ <https://github.com/Maitreyapatel/reliability-score/tree/develop/configs/experiment>`.

Using on different devices:
---------------------------

.. code-block:: shell

    # eval on CPU
    rs

    # eval on 1 GPU
    rs trainer=gpu

    # eval on 2 GPU
    rs trainer=gpu +trainer.gpus=2

    # eval on 2 GPU with specific ids
    rs trainer=gpu +trainer.gpus=[1, 5]

    # eval on TPU
    rs trainer=tpu +trainer.tpu_cores=8

    # eval with DDP (Distributed Data Parallel) (4 GPUs)
    rs trainer=ddp trainer.devices=4

    # eval with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
    rs trainer=ddp trainer.devices=4 trainer.num_nodes=2

    # simulate DDP on CPU processes
    rs trainer=ddp_sim trainer.devices=2

    # accelerate training on mac
    rs trainer=mps


Saving the output of the reliability tests:
-------------------------------------------


.. code-block:: shell

    rs logger=csv



Going beyond user and configuring each experiments:
---------------------------------------------------

Please refer to the `hydra <https://hydra.cc>`_ package and `configs/ <https://github.com/Maitreyapatel/reliability-score/tree/develop/configs>`_ folder to understand the different parameters and features.
Once you understand them then you can modify them on cli (for example, how different devices are used in above example).
