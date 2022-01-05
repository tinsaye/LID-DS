import math
from collections import deque

from algorithms.features.base_feature import BaseFeature
from algorithms.features.impl.threadID import ThreadID
from dataloader.syscall import Syscall


class Sum(BaseFeature):
    """
    gives the sum value from a stream of system call features
    """

    def __init__(self, feature: BaseFeature, thread_aware: bool, window_length: int):
        """
        feature: the sum should be calculated on feature
        thread_aware: True or False
        window_length: length of the window considered
        """
        super().__init__()
        self._window_buffer = {}
        self._sum_values = {}
        self._feature = feature
        self._thread_aware = thread_aware
        self._window_length = window_length

        self._dependency_list = []
        if thread_aware:
            self._dependency_list.append(ThreadID())
        self._dependency_list.append(feature)
        self._feature_id = feature.get_id()

    def depends_on(self):
        return self._dependency_list

    def extract(self, syscall: Syscall, features: dict):
        """
        returns the sum over feature in the window if the feature is in the current set of features
        """
        thread_id = 0
        if self._thread_aware:
            try:
                thread_feature_id = ThreadID().get_id()
                thread_id = features[thread_feature_id]
            except Exception:
                raise KeyError('No thread id in features')
        if thread_id not in self._window_buffer:
            self._window_buffer[thread_id] = deque(maxlen=self._window_length)
            self._sum_values[thread_id] = 0

        if self._feature_id in features:
            dropout_value = 0
            if len(self._window_buffer[thread_id]) == self._window_length:
                dropout_value = self._window_buffer[thread_id][0]
            new_value = features[self._feature_id]
            self._window_buffer[thread_id].append(new_value)
            self._sum_values[thread_id] += new_value - dropout_value
            features[self.get_id()] = self._sum_values[thread_id]

    def new_recording(self):
        """
        empty buffer so ngrams consist of same recording only
        """
        self._window_buffer = {}
        self._sum_values = {}