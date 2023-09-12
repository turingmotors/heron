如何安装
---------------

1. 获取存储库
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/turingmotors/heron.git
   cd heron


2. 设置 Python 环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

建议使用虚拟环境安装所需软件包. 如果要全局安装软件包, 请使用 `pip install -r requirements.txt` 代替.

2-a. Poetry (Recommended)
""""""""""""""""""""""""""""""""""""""""

对于  `pyenv <https://github.com/pyenv/pyenv>`_ 和 `Poetry <https://python-poetry.org/>`_ , 请按照以下步骤安装必要的软件包.

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

对于 `Anaconda <https://www.anaconda.com/>`_ , 请按照以下步骤安装必要的软件包.

.. code-block:: bash

   conda create -n heron python=3.10 -y
   conda activate heron
   pip install --upgrade pip  # enable PEP 660 support

   pip install -r requirements.txt
   pip install -e .

   # for development, install pre-commit
   pre-commit install


.. attention::

   ## 3. 预申请 Llama-2 模型
   要使用 Llama-2 模型, 您需要注册您的模型.
   首先，请访问 `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b>`_ 和 `Meta <https://ai.meta.com/resources/models-and-libraries/llama- downloads/>`_ 并申请访问 llama-2 模型.

   申请通过后, 使用您的 HaggingFace 账户登录.

   .. code-block:: bash

      huggingface-cli login
