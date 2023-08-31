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


import abc

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, is_inference: bool = False):
        super(BaseDataset, self).__init__()
        self.is_inference = is_inference

    @classmethod
    @abc.abstractmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        if self.is_inference:
            return self._get_item_inference(index)
        else:
            return self._get_item_train(index)

    @abc.abstractmethod
    def _get_item_train(self, index):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_item_inference(self, index):
        raise NotImplementedError
