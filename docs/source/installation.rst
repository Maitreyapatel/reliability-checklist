.. _installation:
Installation
============


Installing reliability-score is simple and quick. It has been tested on python3.8 and ubuntu. But it should be able to work on different >python3.5 versions and any OS.


Clone the project:
------------------------
.. code-block:: shell

    git clone git@github.com:Maitreyapatel/reliability-score.git
    cd reliability-score


Setup the anaconda environment:
-------------------------------
.. code-block:: shell

    conda create -n venv python=3.8
    conda activate venv

Install the requirements:
-------------------------
# install pytorch according to instructions

# https://pytorch.org/get-started/

.. code-block:: shell

    pip install -r requirements.txt
