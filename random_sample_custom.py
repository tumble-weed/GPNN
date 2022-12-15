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
original_imname = 'images/ILSVRC2012_val_00000013.JPEG'
# original_imname = 'database/balloons.png'
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
    'iters':10,
    # 'iters':1,#10
    'coarse_dim':14,#
    # 'coarse_dim':28,
    # 'coarse_dim':100,#
    'out_size':0,
    'patch_size':7,
    # 'patch_size':15,
    'stride':1,
    'pyramid_ratio':4/3,
    'faiss':True,
    # 'faiss':False,
    'no_cuda':False,
    #---------------------------------------------
    'in':None,
    'sigma':4*0.75,
    # 'sigma':0.3*0.75,
    'alpha':0.005,
    'task':'random_sample',
    #---------------------------------------------
#     'input_img':original_im,
    'input_img':original_imname,

}
for d in ['output','camoutput','unpermuted_camsoutput','maskoutput']:
    os.system(f'rm -rf {d}')
model = gpnn(config)
with Timer('model run'):
    augmentations,I = model.run(to_save=True)
import gradcam
cams = gradcam.gradcam(augmentations.permute(0,3,1,2),target=370)
from model.utils import *
# import pdb;pdb.set_trace()


for ii,ci in enumerate(cams):
    # assert ci.shape[-1] == 3
    # img_save(tensor_to_numpy(ci), 'cam'+model.out_file[:-len('.png')] + str(ii) + '.png' )
    img_save(ci, 'cam'+model.out_file[:-len('.png')] + str(ii) + '.png' )
# output_im = skimage.io.imread(output_imname)
#====================================================
device ='cuda'
import torch
assert not isinstance(cams,torch.Tensor)
cams = torch.tensor(cams).unsqueeze(1).to(device) #1,1,333,500
# dummy = torch.ones_like(cams).requires_grad_(True)
#==========================================================
# for the masks
from model.my_gpnn  import extract_patches,combine_patches
dummy = torch.zeros(model.batch_size,*cams.shape[-2:],1).to(device).requires_grad_(True)
# dummy_values = extract_patches(dummy, model.PATCH_SIZE, model.STRIDE)
dummy_values = torch.stack([extract_patches(di,  model.PATCH_SIZE, model.STRIDE) for di in dummy],dim=0)
# import pdb;pdb.set_trace()
# mask_keys_flat = mask_keys.reshape((mask_keys.shape[0], -1)).contiguous()
dummy_values = torch.cat([dummy_values[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(model.batch_size,I.shape[0]//model.batch_size,*I.shape[1:]))],dim=0)
# dummy_values = dummy_values.squeeze(0)
# dummy_values = dummy_values.reshape(model.y_pyramid[0].shape[0],
#                         dummy_values.shape[0]//model.y_pyramid[0].shape[0],*dummy_values.shape[1:])
assert dummy_values.ndim == 5,'1,npatches,nchan,7,7'
# import pdb;pdb.set_trace()
augmented_dummy = torch.stack([combine_patches(v, model.PATCH_SIZE, model.STRIDE, dummy.shape[1:3]+(3,),as_np=False) for v in dummy_values],dim=0)
augmented_dummy = augmented_dummy.permute(0,3,1,2).contiguous()
#==========================================================
(augmented_dummy * cams).sum().backward()
for ii,di in enumerate(dummy.grad):
    di = tensor_to_numpy(di)[...,0]
    di = di/di.max()
    img_save(di, 'unpermuted_cams'+model.out_file[:-len('.png')] + str(ii) + '.png' )
if False:
    plt.figure()
    plt.imshow(np.array(original_im[...,:3]))
    plt.show()

    plt.figure()
    plt.imshow(np.array(output_im))
    plt.show()
#%%
