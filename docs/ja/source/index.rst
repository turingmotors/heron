.. heron documentation master file, created by
   sphinx-quickstart on Tue Sep 12 17:08:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Heron
=================================

[日本語] | `[English] </en/latest/>`_ | `[中文] </zh/latest/>`_


Heronは、複数の画像/動画モデルと言語モデルをシームレスに統合するライブラリです。日本語のVision and Language (V&L)モデルをサポートしており、さらに様々なデータセットで学習された事前学習済みウェイトも提供します。

異なるLLMで構築されたマルチモーダルのデモページはこちらをご覧ください。（ともに日本語対応）

* `BLIP + Japanese StableLM Base Alpha <https://huggingface.co/spaces/turing-motors/heron_chat_blip>`_
* `GIT + ELYZA-japanese-Llama-2 <https://huggingface.co/spaces/turing-motors/heron_chat_git>`_

.. image:: ../../../images/heron_image.png
   :scale: 25%


Heronでは、様々なモジュールを組み合わせた独自のV&Lモデルを構成することができます。Vision Encoder、Adopter、LLMを設定ファイルで設定できます。分散学習方法やトレーニングに使用するデータセットも簡単に設定できます。

.. image:: ../../../images/build_train_model.png


組織情報
------------

`Turing株式会社 <https://www.turing-motors.com/>`_

ライセンス
------------

Apache License 2.0において公開されています。

参考情報
------------

* `GenerativeImage2Text <https://github.com/microsoft/GenerativeImage2Text>`_: モデルの構成方法の着想はGITに基づいています。
* `Llava <https://github.com/haotian-liu/LLaVA>`_ : 本ライブラリはLlavaプロジェクトを参考にしています。
* `GIT-LLM <https://github.com/Ino-Ichan/GIT-LLM>`_ 



.. toctree::
   :maxdepth: 2
   :caption: Contents

   ./installation
   ./training
   ./inference
   ./dataset



* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
