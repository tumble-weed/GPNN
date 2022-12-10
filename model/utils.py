from imageio import imread, imsave
from skimage.util import img_as_ubyte
import os
import timeit
import time
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def img_read(path):
	im = imread(path)
	if im.shape[2] > 3:
		im = im[:, :, :3]
	return im


def img_save(im, path):
	dir = os.path.splitext(os.path.dirname(path))[0]
	if not os.path.isdir(dir):
		os.mkdir(dir)
	imsave(path, img_as_ubyte(im))

class Timer():
	def __init__(self,name):
		self.name = name

	def __enter__(self):
		self.tic = time.time()
	def __exit__(self,*others):
		self.toc = time.time()
		self.elapsed = self.toc - self.tic
		print(f'{self.name} took {self.elapsed}')
	pass