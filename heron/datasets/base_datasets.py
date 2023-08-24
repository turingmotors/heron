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

    @abc.abstractmethod
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
