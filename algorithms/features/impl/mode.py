from algorithms.features.base_feature import BaseFeature
from algorithms.features.util.Singleton import Singleton
from dataloader.syscall import Syscall


class Mode(BaseFeature, metaclass=Singleton):

    def __init__(self):
        super().__init__()

    def extract(self, syscall: Syscall, features: dict):
        """
        extract mode parameter from syscall
        eg: mode=0
        """
        params = syscall.params()
        if "mode" in params:
            features[self.get_id()] = params["mode"]
        else:
            features[self.get_id()] = "0"

    def depends_on(self):
        return []