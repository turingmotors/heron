Installation
---------------

1. Clone this repository
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/turingmotors/heron.git
   cd heron


2. Install Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend using virtual environment to install the required packages. If you want to install the packages globally, use `pip install -r requirements.txt` instead.

2-a. Poetry (Recommended)
""""""""""""""""""""""""""""""""""""""""

`pyenv <https://github.com/pyenv/pyenv>`_ and `Poetry <https://python-poetry.org/>`_ , you can install the required packages as follows:

.. code-block:: bash

   # install pyenv environment
   pyenv install 3.10
   pyenv local 3.10

   # install packages from pyproject.toml
   poetry install

   # install local package
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .

   # for development, install pre-commit
   pre-commit install


2-b. Anaconda
""""""""""""""""""""

`Anaconda <https://www.anaconda.com/>`_ , you can install the required packages as follows:

.. code-block:: bash

   conda create -n heron python=3.10 -y
   conda activate heron
   pip install --upgrade pip  # enable PEP 660 support

   pip install -r requirements.txt
   pip install -e .

   # for development, install pre-commit
   pre-commit install


.. attention::

   To use Llama-2 models, you need to register for the models.
   First, you request access to the llama-2 models, in `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b>`_ and `Meta <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_ .

   Please sign-in the Hugging Face account.

   .. code-block:: bash

      huggingface-cli login

