インストール方法
-----------------------

1. リポジトリの取得
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/turingmotors/heron.git
   cd heron


1. Python環境のセットアップ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

必要なパッケージのインストールには仮想環境を使用することを推奨します。グローバルにパッケージをインストールしたい場合は、代わりに `pip install -r requirements.txt` を使ってください。


2-a. Poetry (Recommended)
""""""""""""""""""""""""""""""""""""""""

`pyenv <https://github.com/pyenv/pyenv>`_ と `Poetry <https://python-poetry.org/>`_ の場合、次の手順で必要なパッケージをインストールしてください。

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

`Anaconda <https://www.anaconda.com/>`_ の場合、次の手順で必要なパッケージをインストールしてください。

.. code-block:: bash

   conda create -n heron python=3.10 -y
   conda activate heron
   pip install --upgrade pip  # enable PEP 660 support

   pip install -r requirements.txt
   pip install -e .

   # for development, install pre-commit
   pre-commit install


.. attention::

   Llama-2モデルを使用するには、アクセスの申請が必要です。
   まず、 `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b>`_ と `Meta <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_ のサイトから、llama-2モデルへのアクセスをリクエストしてください。

   リクエストが承認されたら、HaggingFaceのアカウントでサインインしてください。

   .. code-block:: bash

      huggingface-cli login
