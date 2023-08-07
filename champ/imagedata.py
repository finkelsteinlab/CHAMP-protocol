import numpy as np
from champ import misc


class ImageData(object):
    """A class for image data to be correlated with fastq coordinate data."""
    def __init__(self, filename, um_per_pixel, image):
        assert isinstance(image, np.ndarray), 'Image not numpy ndarray'
        self.fname = str(filename)
        self.fft = None
        self.image = image
        self.median_normalize()
        self.um_per_pixel = um_per_pixel
        self.um_dims = self.um_per_pixel * np.array(self.image.shape)

    def median_normalize(self):
        med = np.median(self.image)
        self.image = self.image.astype('float', copy=False, casting='safe')
        self.image /= float(med)
        self.image -= 1.0

    # Perform FFT on the TIFF images. Since the cross-correlation computation is more efficient when matrices size is power of 2. We pad constant 0 to the original TIFF image to fulfill this requirement. 
    def set_fft(self, padding):
        totalx, totaly = np.array(padding) + np.array(self.image.shape)
        dimension = int(max(misc.next_power_of_2(totalx),
                        misc.next_power_of_2(totaly))) # Call the "next_power_of_2" method in the misc.py file to calculate the dimension needed to fit in the TIFF images.
        padded_im = np.pad(self.image,
                           ((int(padding[0]), dimension - int(totalx)), (int(padding[1]), dimension - int(totaly))),
                           mode='constant')
        if padded_im.shape != (dimension, dimension):
            raise ValueError("FFT of microscope image is not a power of 2, this will cause the program to stall.")
        self.fft = np.fft.fft2(padded_im)