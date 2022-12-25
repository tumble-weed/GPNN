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

from model.my_gpnn  import extract_patches,combine_patches
import torch
import gradcam
from model.utils import *
import skimage.io
import skimage.transform
# original_imname = 'images/ILSVRC2012_val_00000013.JPEG'; imagenet_target=370
# original_imname = 'database/balloons.png'
# original_imname = 'database/volacano.png'
print('is it class 6 for voc?')
original_imname = 'images/cars.png'; imagenet_target = 751#6#
# original_imname = 'images/n01443537_16.JPEG'; imagenet_target = 1
# original_imname = 'images/data/feature_inversion/building.jpg'; imagenet_target = 538

output_imname = os.path.join('output',os.path.basename(original_imname))
output_imname_root,ext = output_imname.split('.')
output_imname = output_imname_root + '_random_sample' +'.png'
original_im = skimage.io.imread(original_imname)
# !python random_sample.py -in  {original_imname} --faiss
# assert False
from model.my_gpnn import gpnn
# from model.gpnn import gpnn
im = skimage.io.imread(original_imname)
if im.ndim == 3 and im.shape[-1] > 3:
    im = im[...,:-1]
overshoot = min(im.shape[:2])/256
resize_aspect_ratio = 1/overshoot
print(im.shape)

    
im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))
print(im.shape)
# time.sleep(5)
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
    # 'input_img':original_imname,
    'input_img':im,
    'batch_size':10,
}
for d in ['output','camoutput','unpermuted_camsoutput','maskoutput']:
    os.system(f'rm -rf {d}')
model = gpnn(config)
with Timer('model run'):
    augmentations,I = model.run(to_save=True)
# import pdb;pdb.set_trace()
if False and 'identity I':
    I = torch.tile(torch.arange(I.max()+1).to('cuda'),(augmentations.shape[0],))
    I = I[:,None]

# import torch
torch.cuda.empty_cache()
if 'gradcam' and True:
    extra_for_cam = {}
    # import pdb;pdb.set_trace()
    augmentations.requires_grad_(True)
    # import pdb;pdb.set_trace()    
    cams,scores,probs = gradcam.gradcam(augmentations.permute(0,3,1,2),target=imagenet_target,model_type='imagenet')
if 'gradients' and False:
    from captum.attr import Saliency
    from torchvision.models import resnet50
    cnn = resnet50(pretrained=True).to('cuda')
    saliency = Saliency(cnn)
    # Computes saliency maps for class 3.
    cams = saliency.attribute(augmentations.permute(0,3,1,2), target=imagenet_target)
    cams = cams.abs()
    cams = cams.mean(dim = 1); print('averaging the gradients in channel')
    # cams = cams.max(dim = 1); print('averaging the gradients in channel')
    cams = tensor_to_numpy(cams)
    scores = torch.ones(cams.shape[0]).to('cuda'); print('prob = 1')
    probs = torch.ones(cams.shape[0]).to('cuda'); print('scores = 1')
    
full_shape = cams.shape[1:]
valid_shape_for_ps1 = full_shape[0] - 2*(model.PATCH_SIZE[0]//2),full_shape[1] - 2*(model.PATCH_SIZE[0]//2)
# cams =  np.ones((cams.shape[0],)+full_shape)
#==============================================
# import pdb;pdb.set_trace()
if 'cams for sanity check' and False:
    assert cams.ndim == 3, 'n_cams,H,W'
    sanity_cams0 = torch.tile(torch.linspace(0.1,1.,valid_shape_for_ps1[0])[None,:,None],
                            (cams.shape[0],1,valid_shape_for_ps1[-1])).to('cuda')
    # assert sanity_cams.shape == cams.shape
    sanity_cams_values = torch.stack([extract_patches(di,  
                                                    (1,1),
                                                    #  model.PATCH_SIZE, 
                                                    model.STRIDE) for di in sanity_cams0.unsqueeze(-1)],dim=0)
    # import pdb;pdb.set_trace()
    # mask_keys_flat = mask_keys.reshape((mask_keys.shape[0], -1)).contiguous()
    sanity_cams_values = torch.cat([sanity_cams_values[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(model.batch_size,I.shape[0]//model.batch_size,*I.shape[1:]))],dim=0)
    # sanity_cams_values = torch.cat([sanity_cams_values[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(I.shape[0]//model.batch_size,model.batch_size,*I.shape[1:]).permute(1,0,2))],dim=0)
    # dummy_values = dummy_values.squeeze(0)
    # dummy_values = dummy_values.reshape(model.y_pyramid[0].shape[0],
    #                         dummy_values.shape[0]//model.y_pyramid[0].shape[0],*dummy_values.shape[1:])
    assert sanity_cams_values.ndim == 5,'1,npatches,nchan,7,7'
    # import pdb;pdb.set_trace()
    sanity_cams = torch.stack([combine_patches(v,
                                            (1,1),
                                            #   model.PATCH_SIZE, 
                                            model.STRIDE, 
                                            valid_shape_for_ps1 +(1,),as_np=False,use_divisor=True) for v in sanity_cams_values],dim=0)
    sanity_cams = sanity_cams.permute(0,3,1,2).contiguous()

    assert sanity_cams.max() <= 1
    assert sanity_cams.min() >= 0.1
    sanity_cams = sanity_cams.squeeze(1)
    # if cams.shape != sanity_cams.shape:
    #     import pdb;pdb.set_trace()
    cams = sanity_cams
    cams = tensor_to_numpy(cams)
    # cams =  np.ones(cams.shape)
assert cams.ndim == 3,'cams.ndim == 3'
#==============================================
for ii,ci in enumerate(cams):
    # assert ci.shape[-1] == 3
    # img_save(tensor_to_numpy(ci), 'cam'+model.out_file[:-len('.png')] + str(ii) + '.png' )
    img_save(ci, 'cam'+model.out_file[:-len('.png')] + str(ii) + '.png' )
# output_im = skimage.io.imread(output_imname)
#====================================================
device ='cuda'


assert not isinstance(cams,torch.Tensor)
cams = torch.tensor(cams).unsqueeze(1).to(device) #1,1,333,500
# dummy = torch.ones_like(cams).requires_grad_(True)
#==========================================================
# for the masks
def arrange(dummy,I,output_shape):
    # dummy_values = extract_patches(dummy, model.PATCH_SIZE, model.STRIDE)
    dummy_values0 = torch.stack([extract_patches(di,(1,1),
                                                # model.PATCH_SIZE, 
                                                model.STRIDE) for di in dummy],dim=0)
    # import pdb;pdb.set_trace()
    # mask_keys_flat = mask_keys.reshape((mask_keys.shape[0], -1)).contiguous()
    I1 = I.view(model.batch_size,I.shape[0]//model.batch_size,*I.shape[1:])
    dummy_values = torch.cat([dummy_values0[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(model.batch_size,I.shape[0]//model.batch_size,*I.shape[1:]))],dim=0)
    # dummy_values = torch.cat([dummy_values[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(I.shape[0]//model.batch_size,model.batch_size,*I.shape[1:]).permute(1,0,2))],dim=0)
    # dummy_values = dummy_values.squeeze(0)
    # dummy_values = dummy_values.reshape(model.y_pyramid[0].shape[0],
    #                         dummy_values.shaped[0]//model.y_pyramid[0].shape[0],*dummy_values.shape[1:])
    assert dummy_values.ndim == 5,'1,npatches,nchan,7,7'
    # import pdb;pdb.set_trace()
    # transposed_output_shape = output_shape[1],output_shape[0],output_shape[-1]
    augmented_dummy = torch.stack([combine_patches(v, 
                                                #    model.PATCH_SIZE, 
                                    (1,1),
                                    model.STRIDE,output_shape,as_np=False,use_divisor=True) for v in dummy_values],dim=0)
    augmented_dummy = augmented_dummy.permute(0,3,1,2).contiguous()
    # import pdb;pdb.set_trace()
    return augmented_dummy
dummy = torch.ones(model.batch_size,*valid_shape_for_ps1,1).to(device).requires_grad_(True)
if I.shape[0]//model.batch_size == (cams.shape[-2] - 2*(model.PATCH_SIZE[0]//2)) * (cams.shape[-1] - 2*(model.PATCH_SIZE[1]//2)):
    dummy_shape = (cams.shape[-2] - 2*(model.PATCH_SIZE[0]//2),
                            cams.shape[-1] - 2*(model.PATCH_SIZE[1]//2))+(1,)
    cropped_cams = cams[...,model.PATCH_SIZE[0]//2:-(model.PATCH_SIZE[0]//2),model.PATCH_SIZE[0]//2:-(model.PATCH_SIZE[0]//2)]
    
elif I.shape[0]//model.batch_size == (cams.shape[-2]) * (cams.shape[-1]):
    # assert False,'should not be here'
    dummy_shape = (cams.shape[-2],cams.shape[-1],1)
    cropped_cams = cams

augmented_dummy = arrange(dummy,I,dummy_shape
                        )
dummy1 = torch.ones(model.batch_size,*valid_shape_for_ps1,1).to(device).requires_grad_(True)
augmented_dummy1 = arrange(dummy1,I,
                        dummy_shape)
#==========================================================
assert torch.isclose(augmented_dummy.mean(),torch.ones_like(augmented_dummy.mean()))
assert torch.isclose(augmented_dummy.std(),torch.zeros_like(augmented_dummy.std()))
# (augmented_dummy).sum().backward()
(augmented_dummy * cropped_cams).sum().backward()
# (augmented_dummy * 1/(augmented_dummy.detach())*cams).sum().backward()
# (augmented_dummy).sum().backward()
(augmented_dummy1).sum().backward()
# import pdb;pdb.set_trace()
assert augmented_dummy.min() != 0
avg_cam = 0
avg_cam2 = 0
denom = 0
for ii,(di,ddi) in enumerate(zip(dummy.grad,dummy1.grad)):
    if probs[ii].ndim > 0:
        probs[ii] = probs[ii].mean()
    # ddi = torch.ones_like(ddi)
    di = di/(ddi + (ddi==0).float())
    di_ = tensor_to_numpy(di)[...,0]
    # denom = di
    # assert di.max() >= 0
    # di = di/di.max()
    if False:
        denom = (di_.max() - di_.min())
        img_save((di_ - di_.min())/(denom + (denom==0).astype(np.float32)), 'unpermuted_cams'+model.out_file[:-len('.png')] + str(ii) + '.png' )
    else:
        img_save(di_ , 'unpermuted_cams'+model.out_file[:-len('.png')] + str(ii) + '.png' )
    
    avg_cam = avg_cam + (di * ddi) * probs[ii]
    avg_cam2 = avg_cam2 + (di**2 * ddi) * probs[ii]
    denom = (denom + ddi * probs[ii])
avg_cam = avg_cam / denom
avg_cam2 = avg_cam2/denom
std_cam = (avg_cam2 - avg_cam**2).sqrt()
avg_cam_ = tensor_to_numpy(avg_cam)[...,0]
std_cam_ = tensor_to_numpy(std_cam)[...,0]
if False:
    img_save(
    (avg_cam_ -avg_cam_.min())/ (avg_cam_.max() -avg_cam_.min()), 
    'unpermuted_cams'+model.out_file[:-len('.png')] + 'avg' + '.png' )
else:
    img_save(
    avg_cam_, 
    'unpermuted_cams'+model.out_file[:-len('.png')] + 'avg' + '.png' )
img_save(std_cam_, 'unpermuted_cams'+model.out_file[:-len('.png')] + 'std' + '.png' )
sampling = denom/dummy1.grad.shape[0]
sampling = sampling/sampling.max()
# import pdb;pdb.set_trace()
img_save(tensor_to_numpy(sampling), 
        os.path.join('output','sampling.png') )
if False:
    plt.figure()
    plt.imshow(np.array(original_im[...,:3]))
    plt.show()

    plt.figure()
    plt.imshow(np.array(output_im))
    plt.show()
#%%
import pdb;pdb.set_trace()
