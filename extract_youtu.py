import numpy as np
import matplotlib.pyplot as plt
import cv2

import pywt
from scipy.signal import wiener
import skimage as skimg 

from sklearn.base import (BaseEstimator, TransformerMixin)
from sklearn.pipeline import (make_pipeline, make_union,)

import os
from tqdm import tqdm


class LoadImageRoseYoutu():
    def __init__(self, path_prefix, use_test = True, colorcvt=None):
        assert os.path.exists(path_prefix), "LoadImage, Path does not exist"
        self.path_prefix = path_prefix
        self.colorcvt = colorcvt

        self.to_use_folder = 'test' if use_test else 'adaptation'
        self.txt_file = self.to_use_folder + '_list.txt'
        self.img_path_prefix = os.path.join(self.path_prefix, 'rgb', self.to_use_folder)
        
        self.img_paths, self.folder_no, self.labels = self._read_txt_file()
        self.att_type = np.array([path.split('/')[1][0] for path in self.img_paths])

        self.next_idx = 0
        self.batch_size = 64

    def _read_txt_file(self):
        txt_path = os.path.join(self.path_prefix, self.txt_file)
        with open(txt_path, 'r') as f:
            text = f.readlines()
            lines = np.array([s.split() for s in text])
        return np.moveaxis(lines, -1, 0)
    
    def next_batch(self):
        images = []
        if self.next_idx >= len(self.labels):
            return None
        end_idx = min(len(self.labels), self.next_idx + self.batch_size)

        for i in range(self.next_idx, end_idx):
            file = os.path.join(self.img_path_prefix, self.img_paths[i] + '.jpg')
            img = cv2.imread(file)
            if img is not None:
                images.append(img)
        
        self.next_idx = end_idx
        return images


class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wtname = 'haar', level = 3):
        '''
        waveletname = ['haar', 'db3', 'db5', 'sym2', 'bior5.5', etc.]
        level: total number of decomposite level
        '''
        self.wtname = wtname
        self.level = level
    
    def fit(self, X, y):
        return self

    def transform(self, X):
        features = []
        for img in X:
            img_features = []
            for img_channel in np.moveaxis(img, -1, 0):
                wt = pywt.wavedec2(data=img_channel, wavelet=self.wtname, level=self.level)
                appr = wt[0]
                details = wt[1:]
                wt = [appr]
                for levels in details:
                    for detail in levels:
                        wt.append(detail)
                for _wt in wt:
                    img_features.append(np.mean(_wt))
                    img_features.append(np.var(_wt))
            features.append(img_features)
        return features


class LBPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_points = 8, radius = 1, gray=False, noise=False):
        self.num_points = num_points
        self.radius = radius
        self.gray = gray
        self.noise = noise

    def fit(self, X, y):
        return self
    
    def transform(self, X):
        features = []
        for img in X:
            if self.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if self.noise:
                    img = img - wiener(img, 5)
                features.append(self.get_lbp_features((img,)))
            else:
                features.append(self.get_lbp_features(np.moveaxis(img, -1, 0)))
        return features

    def local_binary_pattern(self, img, normalized=True):
        lbp = skimg.feature.local_binary_pattern(
            img, self.num_points, self.radius, method="nri_uniform").ravel()
        (hist, bins) = np.histogram(lbp.ravel(), bins=59)
        
        if normalized is False:
            return hist
        hist = hist / len(lbp)
        return hist

    def get_lbp_features(self, img_channels):
        lbp_features = np.zeros(59)
        for img in img_channels:
            lbp_features += self.local_binary_pattern(img)
        return lbp_features / len(img_channels)


def dump_feature(path, filename, A, b, folder, att_type):
    np.save(os.path.join(path, filename+".npy"), A)
    np.save(os.path.join(path, filename+"_label.npy"), b)
    np.save(os.path.join(path, filename+"_folder.npy"), folder)
    np.save(os.path.join(path, filename+"_att_type.npy"), att_type)



###################################################

class Query():
    def __init__(self, extractor, colorcvt, num_feature, dump_filename):
        self.extractor = extractor
        self.colorcvt = colorcvt
        self.num_feature = num_feature  # channels * #feature * (#high_img * levels + #low-img)
        self.dump_filename = dump_filename
        

# ----------------------------------------------- #
def main(query: Query):
    extractor = query.extractor
    loader = LoadImageRoseYoutu(PATH_PREFIX, colorcvt=query.colorcvt)
    A = np.empty((1, query.num_feature))

    n = loader.labels.shape[0]
    n = n//64 + (1 if n%64 else 0)
    # print(n)
    for batch in tqdm(range(n)):
        images = loader.next_batch()
        if images is None:
            break
        A = np.concatenate((A, extractor.transform(images)), axis=0)

    A = A[1:]
    print(A.shape)

    dump_feature(PATH_DUMP, query.dump_filename, 
                 A, loader.labels, 
                 loader.folder_no, loader.att_type)


if __name__ == "__main__":
    PATH_PREFIX = '/home/thienn17/Documents/Rose - Youtu/client'
    assert os.path.exists(PATH_PREFIX), "Path does not exist"

    PATH_DUMP = "./object dump/rose-youtu"
    assert os.path.exists(PATH_DUMP), "Path does not exist"

    queries = (

        Query(WaveletTransformer(wtname='db5',level=3), 
              None, 
              3*2*(3*3+1),
              'wt_BGR_3lv_db5'),

        Query(LBPTransformer(gray=True), 
              None, 
              59,
              'lbp_gray'),

        Query(LBPTransformer(), 
              cv2.COLOR_BGR2HSV, 
              59,
              'lbp_HSV'),

        Query(LBPTransformer(gray=True, noise=True), 
              None, 
              59,
              'lbp_noise'),

        Query(WaveletTransformer(wtname='db11',level=3), 
              cv2.COLOR_BGR2YCrCb,
              3*2*(3*3+1),
              'wt_YCC_3lv_db11'),
    ) 

    for q in tqdm(queries):
        print(q.dump_filename)
        main(q)