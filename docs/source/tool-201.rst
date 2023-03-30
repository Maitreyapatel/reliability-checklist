.. _tool201:

reliability-tool 201
====================

This section walks through an example of how to do **customize** reliability tests. This tutorial/ability requires the understanding of all `.yaml` files within the reliability-checklist.

Remove any specific functionality:
-----------------------------------

Quick solution is to add `~` before the functionality name:

.. code-block:: shell

    recheck task=mnli ~<functionality-name>.<name-of-the-feature>



For example, our task is to remove `inv_augmentation` augmentation defined at `inv_pass.yaml <https://github.com/Maitreyapatel/reliability-checklist/blob/release-prep/reliability_checklist/configs/augmentation/inv_pass.yaml>`_ then

.. code-block:: shell

    recheck task=mnli ~augmentation.inv_augmentation

Another example is to remove `calibration_callback` evaluation defined at `calibration.yaml <https://github.com/Maitreyapatel/reliability-checklist/blob/release-prep/reliability_checklist/configs/callbacks/calibration.yaml>`_ then

.. code-block:: shell

    recheck task=mnli ~callbacks.calibration_callback


Pass different value to specific parameters:
--------------------------------------------

The simple solution is to pass the value via command line argument:

.. code-block:: shell

    recheck task=mnli callbacks.calibration_callback.num_bins=20
