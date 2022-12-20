from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from torchvision.models import resnet50
import cv2 
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os 


def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy())
model = resnet50(pretrained=True)
target_layers = [model.layer4]
def normalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    return out
    
def gradcam(img_tensor,target = None,target_layers=target_layers):
    
    input_tensor = normalize_tensor(img_tensor)
    
    targets = [ClassifierOutputTarget(target) for _ in range(img_tensor.shape[0])]
    with GradCAM(model=model,
                target_layers=target_layers,
                use_cuda= ('cuda' in str(img_tensor.device))) as cam:
        extra = {}
        cam.batch_size = 32
        grayscale_cam0 = cam(input_tensor=input_tensor,
                            targets=targets,
                            extra=extra)
    return grayscale_cam0,extra['scores'],extra['probs']

if __name__ == '__main__':
    model = resnet50(pretrained=True)

    imgs = os.listdir("output")
    target_layers = [model.layer4]
    image_path = 'output/{}'.format(imgs[0])
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    input_tensor = torch.zeros([len(imgs),3,rgb_img.shape[0], rgb_img.shape[1]])

    count = 0
    for img in imgs:
        image_path = 'output/{}'.format(img)
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        img_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor[count] = img_tensor
        count+= 1

    targets = None

    with GradCAM(model=model,
                        target_layers=target_layers,
                        use_cuda=True) as cam:

        cam.batch_size = 32
        grayscale_cam0 = cam(input_tensor=input_tensor,
                            targets=targets)

    for i in range(len(imgs)):
        grayscale_cam = grayscale_cam0[i, :]
        image_path = 'output/{}'.format(imgs[i])
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])

        image_name = imgs[i].split(".")[0]

        cv2.imwrite('gradcamOutput/{}_cam.png'.format(image_name), cam_image)


