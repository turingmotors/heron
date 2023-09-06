# Copyright 2023 Turing Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import find_namespace_packages, setup


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="heron",
    version="0.0.1",
    author="Yuichi Inoue, Kotaro Tanahashi",
    description="Heron - Multiple Vision/Video and Language models",
    url="https://github.com/turingmotors/heron",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Vision-Language, Multimodal, Image Captioning, Generative AI, Deep Learning, Library, PyTorch",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="heron.*"),
    install_requires=fetch_requirements("requirements.txt"),
)
