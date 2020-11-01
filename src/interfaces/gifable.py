import abc
import imageio


class GIFable(metaclass=abc.ABCMeta):
    def __init__(self):
        self.images = []

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, val):
        self._images = val

    def add_image(self, image_location):
        self.images.append(imageio.imread(image_location))

    @abc.abstractmethod
    def prepare_images(self):
        return

    def create_gif(self,path,duration):
        imageio.mimsave(path, self.images,duration=duration)