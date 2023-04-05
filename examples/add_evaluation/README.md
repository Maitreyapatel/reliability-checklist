# Define new evaluation metric

This is an important situation, what if we want to add a new metric but we don't understand the `reliability-checklist` infrastructure?

**Good news:** You don't need to have perfect knoweldge of the `reliability-checklist`. In this example, we will learn how to add new evaluation metric by following the `reliability-checklist` standards. 

In this example, let's assume that we want to add the F1 Score in our evaluation pipeline. Now follow below steps to implement this feature:

1. Let's first create the necessary folders.
```bash
mkdir -p ./configs/callbacks
mkdir ./src/
```

2. Create the new metric class inside the `./src/new_metric.py` which inherites the `reliability-checklist` wrapper `MonitorBasedMetric`. Now, you need to define at least three functions which contains the main logic of your code.

    * `init_logic` to define the variables to store the information in the from of dict.
    * `batch_logic` define the batch logic that returns the dict having values to the same variables defined above.
    * `end_logic` to get the final evaluation number from the `saved` input dictionary. This returns the two variables `results` and `extra`. `results` should be another dict containing the `metric_name: value`. While, you have a choice to store any extra variables/data from `extra` variable.
    * `save_logic` can be used to store custom results or plots.

3. Define the `f1score.yaml` inside the `./configs/callbacks`.

4. Define the `new_metric.yaml` inside the `./configs/callbacks` to load the default callbacks and f1score.

5. Now add this directory into python search path and initiate the inference by defining `callbacks=new_eval`:
```bash
export PYTHONPATH="${PYTHONPATH}:${pwd}"
recheck task=mnli trainer=gpu callbacks=new_eval augmentation=default 'hydra.searchpath=[file://./configs/]' +trainer.limit_test_batches=10
```

Note: above example only performs inference on 10 data instances without any augmentation.