import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image

import json

class CITYBboxDataset:
    def __init__(self, data_dir, train=True,
                 use_difficult=False, return_difficult=False,
                 ):
        self.train=train
        self.img_filenames, self.annotation_filenames = self._get_valid_data(data_dir)
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = CITYPERSON_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.img_filenames)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        # Load a image
        img_file = self.img_filenames[i]
        img = read_image(img_file, color=True)

        ## Load annotation
        annotation = self.annotation_filenames[i]
        with open(annotation, 'r') as f:
            annot = json.load(f)
        bbox_list = list()
        for i in annot['objects']:
            if i['label'] != 'ignore':
                x, y, w, h = i['bbox']
                bbox_list += [[y, x, y + h, x + w]]
        bbox = np.stack(bbox_list).astype(np.float32)
        # Get label.
        label = np.zeros(bbox.shape[0], dtype=np.int32)

        difficult = list()
        

        ##### bbox와 bboxVis가 다른걸 이용해서 occlued 인지 아닌지 판단 가능
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example

    def _get_valid_data(self, data_dir):
        """Get all valid images and annotations, which contain people.

        Args:
            annotation_path: annotation path
            img_path: image path

        Returns:
            valid annotation list and image list.
        """
        annotation_path = os.path.join(data_dir, 'gtBboxCityPersons')
        img_path = os.path.join(data_dir, 'leftImg8bit')
        if self.train:
            img_path = os.path.join(img_path, 'train')
            annotation_path = os.path.join(annotation_path, 'train')
        else:
            img_path = os.path.join(img_path, 'val')
            annotation_path = os.path.join(annotation_path, 'val')

        annotation_list, img_list = list(), list()
        for city in os.listdir(annotation_path):
            city_list = os.path.join(annotation_path, city)
            for a in os.listdir(city_list):
                annot_path = os.path.join(city_list, a)
                with open(annot_path, 'r') as f:
                    annot_ = json.load(f)

                valid_index = 0
                for i in annot_['objects']:
                    if i['label'] != 'ignore':
                        valid_index += 1
                if valid_index > 0:
                    annotation_list += [os.path.join(city_list, a)]
                    img_name_ = a.split('.')[0].split('_')[:-1]
                    img_name = ''
                    for n in img_name_:
                        img_name += (n + '_')
                    img_name += 'leftImg8bit.png'
                    img_list += [os.path.join(img_path, city, img_name)]
        return img_list, annotation_list



CITYPERSON_BBOX_LABEL_NAMES = (
    'pedestrian',
    'rider',
    'sitting person',
    'person (other)',
    'person group')
