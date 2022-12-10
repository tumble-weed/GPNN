import numpy as np
import torch
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
from torch.nn.functional import fold, unfold
from .utils import *
from .pca import PCA
TODO = None
class gpnn:
    def __init__(self, config):
        # general settings
        self.T = config['iters']
        self.PATCH_SIZE = (config['patch_size'], config['patch_size'])
        self.COARSE_DIM = (config['coarse_dim'], config['coarse_dim'])
        if config['task'] == 'inpainting':
            mask = img_read(config['mask'])
            mask_patch_ratio = np.max(np.sum(mask, axis=0), axis=0) // self.PATCH_SIZE
            coarse_dim = mask.shape[0] / mask_patch_ratio
            self.COARSE_DIM = (coarse_dim, coarse_dim)
        self.STRIDE = (config['stride'], config['stride'])
        self.R = config['pyramid_ratio']
        self.ALPHA = config['alpha']

        # cuda init
        global device
        if config['no_cuda']:
            device = torch.device('cpu')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print('cuda initialized!')

        # faiss init
        self.is_faiss = config['faiss']
        if self.is_faiss:
            global faiss, res
            import faiss
            res = faiss.StandardGpuResources()
            print('faiss initialized!')

        # input image
        import os
        img_path = None
        if config['task'] == 'structural_analogies':
            img_path = config['img_a']
        else:
            if isinstance(config['input_img'],str):
                img_path = config['input_img']
            else:
                self.input_img = config['input_img']
        
        if isinstance(config['input_img'],str):
            self.input_img = img_read(img_path)

        
        
        if config['out_size'] != 0:
            if self.input_img.shape[0] > config['out_size']:
                self.input_img = rescale(self.input_img, config['out_size'] / self.input_img.shape[0], multichannel=True)

        # pyramids
        pyramid_depth = np.log(min(self.input_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
        self.add_base_level = True if np.ceil(pyramid_depth) > pyramid_depth else False
        pyramid_depth = int(np.ceil(pyramid_depth))
        self.x_pyramid = list(
            tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))
        # import pdb;pdb.set_trace()
        if self.add_base_level is True:
            self.x_pyramid[-1] = resize(self.x_pyramid[-2], self.COARSE_DIM)
        self.y_pyramid = [0] * (pyramid_depth + 1)

        # out_file
        # filename = os.path.splitext(os.path.basename(img_path))[0]
        filename = 'out_img'
        self.out_file = os.path.join(config['out_dir'], "%s_%s.png" % (filename, config['task']))

        # coarse settings
        if config['task'] == 'random_sample':
            noise = np.random.normal(0, config['sigma'], self.COARSE_DIM)[..., np.newaxis]
            self.coarse_img = self.x_pyramid[-1] + noise
        elif config['task'] == 'structural_analogies':
            self.coarse_img = img_read(config['img_b'])
            self.coarse_img = resize(self.coarse_img, self.x_pyramid[-1].shape)
        elif config['task'] == 'inpainting':
            self.coarse_img = self.x_pyramid[-1]
            mask_img = img_read(config['mask'])
            self.mask_pyramid = [0] * len(self.x_pyramid)
            for i in range(len(self.mask_pyramid)):
                mask = resize(mask_img, self.x_pyramid[i].shape) != 0
                mask = extract_patches(mask, self.PATCH_SIZE, self.STRIDE)
                if self.input_img.shape[2] > 1:
                    mask = torch.all(mask, dim=3)
                mask = torch.all(mask, dim=2)
                mask = torch.all(mask, dim=1)
                self.mask_pyramid[i] = mask
        self.n_pca_components = 10
        self.use_pca = False
        print('init done')

    def run(self, to_save=True):
        import tqdm
        for i in tqdm.tqdm_notebook(reversed(range(len(self.x_pyramid)))):
            if i == len(self.x_pyramid) - 1:
                queries = self.coarse_img
                keys = self.x_pyramid[i]
            else:
                queries = resize(self.y_pyramid[i + 1], self.x_pyramid[i].shape)
                keys = resize(self.x_pyramid[i + 1], self.x_pyramid[i].shape)
            new_keys = True
            for j in tqdm.tqdm_notebook(range(self.T)):
                if self.is_faiss:
                    self.y_pyramid[i] = self.PNN_faiss(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                       self.ALPHA, mask=None, new_keys=new_keys)
                else:
                    self.y_pyramid[i] = self.PNN(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                 self.ALPHA)
                queries = self.y_pyramid[i]
                keys = self.x_pyramid[i]
                if j > 1:
                    new_keys = False
        if to_save:
            img_save(self.y_pyramid[0], self.out_file)
        else:
            return self.y_pyramid[0]

    def PNN(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None):
        queries = extract_patches(y_scaled, patch_size, stride)
        keys = extract_patches(x_scaled, patch_size, stride)
        values = extract_patches(x, patch_size, stride)
        if mask is None:
            dist = torch.cdist(queries.view(len(queries), -1), keys.view(len(keys), -1))
        else:
            m_queries, m_keys = queries[mask], keys[~mask]
            dist = torch.cdist(m_queries.view(len(m_queries), -1), m_keys.view(len(m_keys), -1))
        norm_dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
        NNs = torch.argmin(norm_dist, dim=1)  # find_NNs
        if mask is None:
            values = values[NNs]
        else:
            values[mask] = values[~mask][NNs]
            # O = values[NNs]  # replace_NNs(values, NNs)
        y = combine_patches(values, patch_size, stride, x_scaled.shape)
        return y

    def PNN_faiss(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None, new_keys=True):
        print('using faiss')
        queries = extract_patches(y_scaled, patch_size, stride)
        print('extracted query',queries.shape)
        keys = extract_patches(x_scaled, patch_size, stride)
        print('extracted keys')
        values = extract_patches(x, patch_size, stride)
        print('extracted values')
        if mask is not None:
            queries = queries[mask]
            keys = keys[~mask]
        #====================================================================

        queries_flat = np.ascontiguousarray(queries.reshape((queries.shape[0], -1)).cpu().numpy(), dtype='float32')
        keys_flat = np.ascontiguousarray(keys.reshape((keys.shape[0], -1)).cpu().numpy(), dtype='float32')
        if new_keys:
            if self.use_pca:
                self.pca = PCA(self.n_pca_components)
                keys_proj = self.pca.fit_transform(keys_flat)
                keys_proj = np.ascontiguousarray(keys_proj)
            else:
                keys_proj = keys_flat
            n_patches = keys_flat.shape[-1]
            print(n_patches)
            
            self.index = faiss.IndexFlatL2(keys_proj.shape[-1])
            # import pdb;pdb.set_trace()
            print('created index')
            # import pdb;pdb.set_trace()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print('pushed index to gpu')
            self.index.add(keys_proj)
        if self.use_pca:
            queries_proj = self.pca.transform(queries_flat)
            queries_proj = np.ascontiguousarray(queries_proj)
        else:
            queries_proj = queries_flat
        print('searching')
        D, I = self.index.search(queries_proj, 1)
        if mask is not None:
            values[mask] = values[~mask][I.T]
        else:
            values = values[I.T]
            #O = values[I.T]
        #====================================================================
        y = combine_patches(values, patch_size, stride, x_scaled.shape)
        print('combined')
        return y


def extract_patches(src_img, patch_size, stride):
    channels = 3
    img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
    return torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
        .squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])


def compute_distances(queries, keys):
    dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float16, device=device)
    for i in range(len(queries)):
        dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3))
    return dist_mat


def combine_patches(O, patch_size, stride, img_shape):
    channels = 3
    O = O.permute(1, 0, 2, 3).unsqueeze(0)
    patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
    combined = fold(patches, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones((1, img_shape[2], img_shape[0], img_shape[1]), dtype=O.dtype, device=device)
    divisor = unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = fold(divisor, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()

config = {
    'out_dir':None,
    'iters':10,
    'coarse_dim':14,
    'out_size':0,
    'patch_size':7,
    'stride':1,
    'pyramid_ratio':4/3,
    'faiss':True,
    'no_cuda':False,
    #---------------------------------------------
    'in':None,
    'sigma':0.75,
    'alpha':0.005,
    'task':None,
    #---------------------------------------------


}


'''
import argparse
from model.gpnn import gpnn
from model.parser import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parser_general(parser)
    parser = parser_sample(parser)
    config = vars(parser.parse_args())
    model = gpnn(config)
    model.run()


def parser_general(parser):
    parser.add_argument('-out', '--out_dir', type=str, required=False, default='./output', help='path of output dir')
    parser.add_argument('--coarse_dim', type=int, default=14, required=False, help='the height of the coarsest pyramid level. default is 14 (int)')
    parser.add_argument('--out_size', type=int, default=0, required=False, help='output image height. should be smaller than original image. default is 0 - as input (int)')
    parser.add_argument('--patch_size', type=int, default=7, required=False, help='the size of the square patches to use in nearest neighbors. default is 7 (int)')
    parser.add_argument('--stride', type=int, default=1, required=False, help='the stride between patches in the nearest neighbros method. default is 1 (int)')
    parser.add_argument('--iters', type=int, default=10, required=False, help='number of refinement iterations in each pyramid scale. default is 10 (int)')
    parser.add_argument('--pyramid_ratio', type=float, default=4 / 3, required=False, help='the ratio between pyramid scales. default is 4/3 (float)')
    parser.add_argument('--faiss', action='store_true', default=False, help='indicate to use faiss approximate nearest-neighbor. default is False (boolean)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='indicate to run only on cpu. default is False (boolean)')
    return parser


def parser_sample(parser):
    parser.add_argument('-in', '--input_img', type=str, required=True, help='path of input image')
    parser.add_argument('--sigma', type=float, default=0.75, required=False, help='noise level to adjust the variatonality of the new sample. default is 0.75 (float)')
    parser.add_argument('--alpha', type=float, default=0.005, required=False, help='alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 0.005 (float)')
    parser.add_argument('--task', type=str, default='random_sample')
    return parser


def parser_analogies(parser):
    parser.add_argument('-a', '--img_a', type=str, required=True, help='path of image A - the content')
    parser.add_argument('-b', '--img_b', type=str, required=True, help='path of image B - the structure')
    parser.add_argument('--alpha', type=float, default=0.005, required=False, help='alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 0.005 (float)')
    parser.add_argument('--task', type=str, default='structural_analogies')
    return parser


def parser_inpainting(parser):
    parser.add_argument('-in', '--input_img', type=str, required=True, help='path of input image')
    parser.add_argument('-m', '--mask', type=str, required=True, help='path of an image with ones where the inpainting is in the input image and zeroes elsewhere')
    parser.add_argument('--alpha', type=float, default=1, required=False, help='alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 1.0 (float)')
    parser.add_argument('--task', type=str, default='inpainting')
    return parser
'''