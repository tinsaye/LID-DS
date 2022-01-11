from algorithms.features.base_feature import BaseFeature
from algorithms.features.util.Singleton import Singleton
from dataloader.syscall import Syscall


class UnknownFlags(BaseFeature, metaclass=Singleton):
    def __init__(self):
        super().__init__()
        self._flag_dict = {}

    def depends_on(self) -> list:
        return []

    def train_on(self, syscall: Syscall, features: dict):
        """
            builds dictionary with all known flags seen in training for each syscall
        """
        if 'flags' in syscall.params().keys():
            if syscall.name() in self._flag_dict:
                self._flag_dict[syscall.name()].append(syscall.param('flags'))
            else:
                self._flag_dict[syscall.name()] = []
                self._flag_dict[syscall.name()].append(syscall.param('flags'))

    def extract(self, syscall: Syscall, features: dict):
        """
            lookup of syscall flag in know flags
            if unknown -> returns 1 else 0
        """
        if 'flags' in syscall.params().keys():
            if syscall.param('flags') in self._flag_dict[syscall.name()]:
                features[self.get_id()] = 0
            else:
                features[self.get_id()] = 1
        else:
            features[self.get_id()] = 0
