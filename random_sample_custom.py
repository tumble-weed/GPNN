#%%
import os
import numpy as np
# os.chdir('/root/evaluate-saliency-5/GPNN')
#%%
# faiss doesnt work without gpu
# !python random_sample.py -in database/balloons.png --faiss
import skimage.io
from matplotlib import pyplot as plt
from model.utils import Timer
# original_imname = 'images/ILSVRC2012_val_00000013.JPEG'
original_imname = 'database/balloons.png'
# original_imname = 'database/volacano.png'

output_imname = os.path.join('output',os.path.basename(original_imname))
output_imname_root,ext = output_imname.split('.')
output_imname = output_imname_root + '_random_sample' +'.png'
original_im = skimage.io.imread(original_imname)
# !python random_sample.py -in  {original_imname} --faiss
# assert False
from model.my_gpnn import gpnn
# from model.gpnn import gpnn
config = {
    'out_dir':'output',
    # 'iters':10,
    'iters':1,#10
    # 'coarse_dim':14,#
    'coarse_dim':100,#
    'out_size':0,
    'patch_size':7,
    'stride':1,
    'pyramid_ratio':4/3,
    'faiss':True,
    # 'faiss':False,
    'no_cuda':False,
    #---------------------------------------------
    'in':None,
    'sigma':1*0.75,
    'alpha':0.005,
    'task':'random_sample',
    #---------------------------------------------
#     'input_img':original_im,
    'input_img':original_imname,

}

model = gpnn(config)
with Timer('model run'):
    augmentations = model.run(to_save=True)
import gradcam
cams = gradcam.gradcam(augmentations.permute(0,3,1,2),target=1)

# output_im = skimage.io.imread(output_imname)

if False:
    plt.figure()
    plt.imshow(np.array(original_im[...,:3]))
    plt.show()

    plt.figure()
    plt.imshow(np.array(output_im))
    plt.show()
#%%
