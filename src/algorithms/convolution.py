from ..interfaces.stateful import Stateful
from ..interfaces.gifable import GIFable
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import tempfile

'''
2D Convolution representation class extending Stateful.
image - image path
kernel_size - tuple of x,y sizes of the kernel
stride - number of strides
padding - true / false
filter - if defined input as 2d array
'''


class Convolution(Stateful,GIFable):

    def __init__(self,image,padding,strides=1,kernel=None):
        Stateful.__init__(self)
        GIFable.__init__(self)
        self.image = image
        self.strides=strides
        self.padding=padding
        if kernel is None:
            self.kernel=self.random_filter()
        else:
            self.kernel=kernel
        self.kernel_size=self.kernel.shape
        self.out_size = self.calc_out_size()
        self.output = np.zeros(self.out_size)
    def calc(self):
        if self.padding != 0: # if padding defined then we need to resize the image and add zeros
            pad = cv2.copyMakeBorder(self.image, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT)
        else:
            pad=self.image #without padding the image stays the same
        imgXrange = pad.shape[0]
        imgYrange= pad.shape[1]
        kernXrange = self.kernel.shape[0]
        kernYrange = self.kernel.shape[1]
        out_gen = self.output_cordinates(self.output)

        for x in range(imgXrange):
            if x > imgXrange - kernXrange:
                break
            if x % self.strides == 0:
                for y in range(imgYrange):
                    if y > imgYrange - kernYrange:
                        break
                    try:
                        if y % self.strides == 0:
                            res=(self.kernel * pad[x: x + kernXrange, y: y + kernYrange]).sum()
                            x_out , y_out = out_gen.__next__()
                            self.output[x_out, y_out] = res
                            self.add_state(
                                {'pad': pad, 'x': x, 'kernX': kernXrange, 'kernY': kernYrange, 'y': y, 'res': res,
                                 'out': self.output.copy(), 'x_out':x_out,'y_out':y_out})
                    except Exception as e:
                        print(e)
                        break
                    '''
                    Convolution State : 
                    image - pad
                    ___current image positions :
                    start position - (x,y)
                    end position - (kernXRange,kernYrange)
                    
                    __output 
                        output itself - res
                        output modification - x,y
                    '''
        return self.output

    def get_filter_size(self) -> int :
        return self.filter.size()

    '''
    this method calculates image output size after applying 2d convolution.
    :returns tuple of x and y sizes as (int,int)
    '''

    def calc_out_size(self) :
        return ((int((self.image.shape[0] - self.kernel_size[0] +2 * self.padding) / self.strides) + 1),
               (int((self.image.shape[1] - self.kernel_size[1] +2 * self.padding) / self.strides) + 1))


    def prepare_images(self):
        # self.add_state((pad, x, kernXrange, y, kernYrange, res))
        annot_kws = {'fontsize': 10}
        cmap = sns.diverging_palette(250, 250, as_cmap=True) #color map
        title_font_size = 15
        props = dict(boxstyle='round', facecolor='red', alpha=0.7)
        for entry in self.state_data: # for each state we create a plot
            fig = plt.figure(figsize=(20, 20)) # main figure
            image_ax = fig.add_subplot(3, 3, 1) #subplots
            kernel_ax = fig.add_subplot(3, 3, 2)
            output_ax = fig.add_subplot(3, 3, 3)

            # image  subplot
            image_ax.add_patch(Rectangle((entry['y'], entry['x']), entry['kernX'], entry['kernY'], fill=False, edgecolor='gold', lw=5))
            image_ax.set_title("Image", color='red', fontsize=title_font_size)
            sns.heatmap(entry['pad'], annot=True, annot_kws=annot_kws, linewidths=.5, cbar=False, ax=image_ax, cmap=cmap)

            # kernel subplot
            kernel_ax.add_patch(Rectangle((0,0), self.kernel.shape[0], self.kernel.shape[1], fill=False, edgecolor='gold', lw=5))
            kernel_ax.set_title("Kernel", color='red', fontsize=title_font_size)
            sns.heatmap(self.kernel, annot=True, annot_kws=annot_kws, linewidths=.5, cbar=False, ax=kernel_ax, cmap=cmap)

            #formula as string calculation
            pad_cols=entry['pad'][entry['x']:entry['x']+entry['kernX'],entry['y']:entry['y']+entry['kernY']]
            formula='+'.join(f'{x:.2f}*{y:.2f}' for row in zip(self.kernel, pad_cols) for x, y in zip(*row))
            formula+=" = {:.1f}".format(entry['res'])
            plt.text(0.3, -0.2, formula, transform=image_ax.transAxes, fontsize=10,verticalalignment='center', bbox=props)

            #output subplot
            output_ax.add_patch(Rectangle((entry['y_out'], entry['x_out']), 1,1, fill=False, edgecolor='gold', lw=5))
            output_ax.set_title("Output", color='red', fontsize=title_font_size)
            sns.heatmap(entry['out'], annot=True, annot_kws=annot_kws, linewidths=.5, cbar=False, ax=output_ax, cmap=cmap)

            #saving image and adding to images list
            temp_file_loc = tempfile.gettempdir()
            temp_file_loc=os.path.join(temp_file_loc,"fig.png")
            plt.savefig(temp_file_loc, bbox_inches='tight')
            self.add_image(temp_file_loc)
            os.remove(temp_file_loc)
            plt.close(fig)

    def print(self):
        plt.imshow(self.kernel, cmap="gray")
        plt.show()

    ''''
    create a random filter based on a given sizes from the constructor
    '''
    def random_filter(self):
        return np.random.rand(self.kernel_size[0],self.kernel_size[1])
    ''''
    generator for the output cordinates
    '''
    def output_cordinates(self,output):
        for i in range(len(output)):
            for j in range(len(output[0])):
                yield i, j