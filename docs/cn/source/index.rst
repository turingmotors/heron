.. heron documentation master file, created by
   sphinx-quickstart on Tue Sep 12 17:08:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Heron
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


`[日本語] </ja/latest/>`_ | `[English] </en/latest/>`_ | [中文]

Heron是一个可无缝集成多种图像/视频和语言模型的库. 此外, 它还提供在各种数据集上训练的预训练权重.

点击此处查看使用不同 LLM 制作的多模态演示页面.（均有日语版本）

* `BLIP + Japanese StableLM Base Alpha <https://huggingface.co/spaces/turing-motors/heron_chat_blip>`_
* `GIT + ELYZA-japanese-Llama-2 <https://huggingface.co/spaces/turing-motors/heron_chat_git>`_

.. image:: ../../../images/heron_image.png
   :scale: 25%

Heron允许您结合各种模块配置自己的V&L模型. 可以在配置文件中配置视觉编码器, Adopter和LLM. 用于训练的分布式学习方法和数据集也可以轻松配置.

.. image:: ../../../images/build_train_model.png


组织信息
------------

`Turing株式会社 <https://www.turing-motors.com/>`_

许可
------------

License 2.0.

Acknowledgements
------------------------

* `GenerativeImage2Text <https://github.com/microsoft/GenerativeImage2Text>`_
* `Llava <https://github.com/haotian-liu/LLaVA>`_ 
* `GIT-LLM <https://github.com/Ino-Ichan/GIT-LLM>`_ 


.. toctree::
   :maxdepth: 2
   :caption: Contents

   ./installation
   ./training
   ./inference
   ./dataset


Index
------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
