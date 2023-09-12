.. heron documentation master file, created by
   sphinx-quickstart on Tue Sep 12 17:08:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Heron
=================================


`[日本語] </ja/latest/>`_ | [English] | `[中文] </zh/latest/>`_

Welcome to "heron" repository. Heron is a library that seamlessly integrates multiple Vision and Language models, as well as Video and Language models. One of its standout features is its support for Japanese V&L models. Additionally, we provide pretrained weights trained on various datasets.

Please click here to see the multimodal demo pages built with different LLMs. (Both are available in Japanese)

* `BLIP + Japanese StableLM Base Alpha <https://huggingface.co/spaces/turing-motors/heron_chat_blip>`_
* `GIT + ELYZA-japanese-Llama-2 <https://huggingface.co/spaces/turing-motors/heron_chat_git>`_

.. image:: ../../../images/heron_image.png
   :scale: 25%

Heron allows you to configure your own V&L models combining various modules. Vision Encoder, Adopter, and LLM can be configured in the configuration file. The distributed learning method and datasets used for training can also be easily configured.

.. image:: ../../../images/build_train_model.png




Organization
------------

`Turing株式会社 <https://www.turing-motors.com/>`_

License
------------

Released under the Apache License 2.0.

Acknowledgements
------------------------

* `GenerativeImage2Text <https://github.com/microsoft/GenerativeImage2Text>`_: The main idia of the model is based on original GIT.
* `Llava <https://github.com/haotian-liu/LLaVA>`_ : This project is learned a lot from the great Llava project.
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
