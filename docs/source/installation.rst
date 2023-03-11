.. _installation:
Installation
============


Installing reliability-score is simple and quick. It has been tested on python3.8 and ubuntu. But it should be able to work on different >python3.5 versions and any OS.

.. code-block:: shell

    pip install git+https://github.com/Maitreyapatel/reliability-score

    python -m spacy download en_core_web_sm
    python -c "import nltk;nltk.download('wordnet')"


Develop installation guidelines
-------------------------------

Clone the project:
~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    git clone git@github.com:Maitreyapatel/reliability-score.git
    cd reliability-score


Setup the anaconda environment:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    conda create -n venv python=3.8
    conda activate venv

Install the requirements:
~~~~~~~~~~~~~~~~~~~~~~~~~
install pytorch according to instructions: https://pytorch.org/get-started/

.. code-block:: shell

    pip install -r requirements.txt

    python -m spacy download en_core_web_sm
    python -c "import nltk;nltk.download('wordnet')"