from src.algorithms.convolution import Convolution
import numpy as np
import pathlib
import os

output_file_name="conv.gif"
img = np.random.rand(7,7)
kernel=np.random.rand(2,2)
padding=0
strides=1
savelocation=os.path.join(pathlib.Path(__file__).parent.absolute(),output_file_name)

c2d = Convolution(image=img, padding=padding, strides=strides, kernel=kernel)
c2d.calc()
c2d.prepare_images()
c2d.create_gif(savelocation,duration=1.5)


