import copy
import numpy as np

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image

## feature dimensions by layer_ind
## input: 3 x 224 x 224
## 0: [64, 112, 112] = 802,816
## 1: [128, 56, 56] = 401,408
## 2: [256, 28, 28] = 200,704
## 3: [512, 14, 14] = 100,352
## 4: [512, 7, 7] = 50,176
## 5: [1, 4096]
## 6: [1, 4096]

use_cuda=False
class VGG19Embeddings(nn.Module):
    """Splits vgg19 into separate sections so that we can get
    feature embeddings from each section.
    :param vgg19: traditional vgg19 model
    """
    def __init__(self, vgg19, layer_index=-1, spatial_avg=True):
        super(VGG19Embeddings, self).__init__()
        self.conv1 = nn.Sequential(*(list(vgg19.features.children())[slice(0, 5)]))
        self.conv2 = nn.Sequential(*(list(vgg19.features.children())[slice(5, 10)]))
        self.conv3 = nn.Sequential(*(list(vgg19.features.children())[slice(10, 19)]))
        self.conv4 = nn.Sequential(*(list(vgg19.features.children())[slice(19, 28)]))
        self.conv5 = nn.Sequential(*(list(vgg19.features.children())[slice(28, 37)]))
        self.linear1 = nn.Sequential(*(list(vgg19.classifier.children())[slice(0, 2)]))
        self.linear2 = nn.Sequential(*(list(vgg19.classifier.children())[slice(3, 5)]))
        self.linear3 = nn.Sequential(list(vgg19.classifier.children())[-1])
        layer_index = int(float(layer_index)) # bll
        assert layer_index >= -1 and layer_index < 8
        self.layer_index = layer_index
        self.spatial_avg = spatial_avg

    def _flatten(self, x):
        if (self.spatial_avg==True) & (self.layer_index<5):
            x = x.mean(3).mean(2)
        return x.view(x.size(0), -1)

    def forward(self, x):
        # build in this ugly way so we don't have to evaluate things we don't need to.
        x_conv1 = self.conv1(x)
        if self.layer_index == 0:
            return [self._flatten(x_conv1)]
        x_conv2 = self.conv2(x_conv1)
        if self.layer_index == 1:
            return [self._flatten(x_conv2)]
        x_conv3 = self.conv3(x_conv2)
        if self.layer_index == 2:
            return [self._flatten(x_conv3)]
        x_conv4 = self.conv4(x_conv3)
        if self.layer_index == 3:
            return [self._flatten(x_conv4)]
        x_conv5 = self.conv5(x_conv4)
        x_conv5_flat = self._flatten(x_conv5)
        if self.layer_index == 4:
            return [x_conv5_flat]
        x_linear1 = self.linear1(x_conv5_flat)
        if self.layer_index == 5:
            return [x_linear1]
        x_linear2 = self.linear2(x_linear1)
        if self.layer_index == 6:
            return [x_linear2]
        x_linear3 = self.linear3(x_linear2)
        if self.layer_index == 7:
            return [x_linear3]
        return [self._flatten(x_conv1), self._flatten(x_conv2),
                self._flatten(x_conv3), self._flatten(x_conv4),
                self._flatten(x_conv5), x_linear1, x_linear2, x_linear3]

class FeatureExtractor():

    def __init__(self,paths,layer=6, use_cuda=False, imsize=224, batch_size=64, cuda_device=0, data_type='images',spatial_avg=True):
        self.layer = layer
        self.paths = paths
        self.num_images = len(self.paths)
        self.use_cuda = use_cuda
        self.imsize = imsize
        self.padding = 10
        self.batch_size = batch_size
        self.cuda_device = cuda_device
        self.data_type = data_type ## either 'images' or 'sketches'
        self.spatial_avg = spatial_avg ## if true, collapse across spatial dimensions to just preserve channel activation

    def extract_feature_matrix(self):

        def load_image(path, imsize=224, padding=self.padding, volatile=True, use_cuda=False):
            im = Image.open(path)

            loader = transforms.Compose([
                transforms.Pad(padding),                
                transforms.Resize(imsize),
                transforms.ToTensor()])
            
            im = Variable(loader(im), requires_grad=False) # used to be volatile = volatile instead of requires_grad = False
            
            if use_cuda:
                im = im.cuda(self.cuda_device)
            return im

        def load_vgg19(layer_index=self.layer,use_cuda=False,cuda_device=self.cuda_device):
            if use_cuda:
                vgg19 = models.vgg19(pretrained=True).cuda(self.cuda_device)
            else:
                vgg19 = models.vgg19(pretrained=True)#.cuda(self.cuda_device)
            vgg19 = VGG19Embeddings(vgg19,layer_index,spatial_avg=self.spatial_avg)
            vgg19.eval()  # freeze dropout

            # freeze each parameter
            for p in vgg19.parameters():
                p.requires_grad = False

            return vgg19

        def flatten_list(x):
            return np.array([item for sublist in x for item in sublist])

        def get_metadata_from_path(path):
            parsed_path = path.split('/')[-1].split('.')[0]
            return parsed_path

        def generator(paths, imsize=self.imsize, use_cuda=use_cuda):
            for path in paths:
                image = load_image(path)
                fname = get_metadata_from_path(path)
                yield (image, fname)

        # define generator
        generator = generator(self.paths,imsize=self.imsize,use_cuda=self.use_cuda)
        
        # initialize image and label matrices
        Features = []
        ImageIDs = []

        n = 0        

        # load appropriate extractor
        extractor = load_vgg19(layer_index=self.layer)

        # generate batches of images and labels
        if generator:
            while True:
                batch_size = self.batch_size
                image_batch = Variable(torch.zeros(batch_size, 3, self.imsize, self.imsize))
                if use_cuda:
                    image_batch = image_batch.cuda(self.cuda_device)
                imageid_batch = []

                if (n+1)%1==0:
                    print('Extracting features from batch {}'.format(n + 1))
                
                for b in range(batch_size):
                    try:
                        image,imageid = next(generator)
                        image_batch[b] = image
                        imageid_batch.append(imageid)                        

                    except StopIteration:                        
                        print('stopped!')
                        break
                
                n += 1                  
                if n == self.num_images//self.batch_size:
                    image_batch = image_batch.narrow(0,0,b + 1)
                    imageid_batch = imageid_batch[:b + 1]                                                  

                # extract features from batch
                image_batch = extractor(image_batch)
                image_batch = image_batch[0].cpu().data.numpy()

                if len(Features)==0:
                    Features = image_batch                    
                else:
                    Features = np.vstack((Features,image_batch))

                ImageIDs.append(imageid_batch)
                print('Shape of Features',np.shape(Features))

                if n == self.num_images//batch_size:
                    break

        ImageIDs = flatten_list(ImageIDs)
        return Features,ImageIDs