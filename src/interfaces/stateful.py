import abc


class Stateful(metaclass=abc.ABCMeta):

    def __init__(self):
        self.state_data = []

    @property
    def state_data(self):
        return self._state_data
    @state_data.setter
    def state_data(self,val):
        self._state_data=val

    def add_state(self, data):
        self.state_data.append(data)

    @abc.abstractmethod
    def calc(self):
        return
    @abc.abstractmethod
    def print(self):
        return