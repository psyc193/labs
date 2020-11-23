import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

from glob import glob
import os

import numpy as np
import pandas as pd
import json
import re

from PIL import Image
import base64

from embeddings import *

'''
To extract features, run, e.g.:
python extract_features.py --data=/images/ --layer_ind=5 --data_type=images --spatial_avg=True --channel_norm=False --out_dir=/features/
'''

def list_files(path, ext='png'):
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result

def make_dataframe(ImageIDs):    
    Y = pd.DataFrame([ImageIDs])
    Y = Y.transpose()
    Y.columns = ['image_id']
    return Y

def normalize(X):
    X = X - X.mean(0)
    X = X / np.maximum(X.std(0), 1e-5)
    return X

def preprocess_features(Features, Y, channel_norm=True):
    _Y = Y.sort_values(['image_id'])
    inds = np.array(_Y.index)
    if channel_norm==True:
        _Features = normalize(Features[inds])
    else:
        _Features = Features[inds]
    _Y = _Y.reset_index(drop=True) # reset pandas dataframe index
    return _Features, _Y
  

def save_features(Features, Y, layer_num, data_type,out_dir='features'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    layers = ['P1','P2','P3','P4','P5','FC6','FC7']
    np.save(os.path.join(out_dir,'FEATURES_{}_{}.npy'.format(layers[int(layer_num)], data_type)), Features)
    Y.to_csv(os.path.join(out_dir,'METADATA_{}.csv'.format(data_type)), index=True, index_label='feature_ind')
    print('Saved features out to {}'.format(out_dir))
    return layers[int(layer_num)]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    import argparse        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='full path to images', default='images')
    parser.add_argument('--layer_ind', help='fc6 = 5, fc7 = 6', default=5)
    parser.add_argument('--num_pcs', help='number of principal components', default=128)    
    parser.add_argument('--data_type', help='"images" or "sketch"', default='images')
    parser.add_argument('--out_dir', help='path to save features to', default='features')    
    parser.add_argument('--spatial_avg', type=bool, help='collapse over spatial dimensions, preserving channel activation only if true', default=True) 
    parser.add_argument('--channel_norm', type=str2bool, help='apply channel-wise normalization?', default='True')    
    parser.add_argument('--test', type=str2bool, help='testing only, do not save features', default='False')  
    parser.add_argument('--ext', type=str, help='image extension type (e.g., "png")', default="png")   
    

    args = parser.parse_args()
    print(f'data type is {args.data_type}')
    print(f'Spatial averaging is {args.spatial_avg}')
    print(f'Channel norm is {args.channel_norm}')
    print(f'Testing mode is {args.test}')
    print(f'VGG layer index is {args.layer_ind}')
    print(f'Output directory is {args.out_dir}')
    
    ## make out_dir if it does not already exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    ## get list of all sketch paths
    image_paths = sorted(list_files(args.data,args.ext))
    print(f'Path to data is {args.data}. Image extension type is {args.ext}')
    print(f'Length of image_paths before filtering: {len(image_paths)}')
    
    ## extract features
    layers = ['P1','P2','P3','P4','P5','FC6','FC7']
    extractor = FeatureExtractor(image_paths,layer=args.layer_ind,\
                                 data_type=args.data_type,\
                                 spatial_avg=args.spatial_avg)
    Features,ImageIDs = extractor.extract_feature_matrix()   
    
    # organize metadata into dataframe
    Y = make_dataframe(ImageIDs)    

    # save out features and meta
    if args.test==False:        
        layer = save_features(Features, Y, args.layer_ind, args.data_type,out_dir = args.out_dir)  
    
       