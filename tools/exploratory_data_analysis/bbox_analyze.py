# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et


CLASSES = ('ore carrier', 'bulk cargo carrier', 'container ship',
           'general cargo ship', 'fishing boat', 'passenger ship')


class BboxAnalyze:
    def __init__(self, data_root, *, idx_file='trainval'):
        """
        :param data_root:
        """
        self.root = os.path.join(data_root)
        idx_file_ = open(os.path.join(self.root, 'ImageSets',
                                      'Main', f'{idx_file}.txt')).readlines()
        self.idx_list = [i.rstrip('\n') for i in idx_file_]
        self.label_names = CLASSES

    @property
    def get_collect_data(self):
        collected = {'img_size': [],
                     'img_path': [],
                     'bbox': [],
                     'labels': []}
        for idx in self.idx_list:
            img_size, img_path, bbox, labels = self._get_collect(idx)
            collected['img_size'].append(img_size)
            collected['bbox'].append(bbox)
            collected['labels'].append(labels)
            collected['img_path'].append(img_path)
        return collected

    # 锚框长宽比，锚框区域大小
    def analyze_bbox_dist(self, bbox: np.array):
        h, w = bbox[:, 3] - bbox[:, 1], bbox[:, 2] - bbox[:, 0]
        area = h * w
        area_count, area_bins = np.histogram(area)

        try:
            ratio = h / w
        except Warning:
            print(f'{w}: dividee by zero, height: {h}, width: {w}')
        n, bins, _, = plt.hist(area, bins=50)

        # 标注最高那条bar所占数量
        n_max = np.argmax(n)
        bins_max = bins[n_max]
        plt.text(bins[n_max] + (bins[1] - bins[0]) / 2, n[n_max] * 1.01, int(n[n_max]), ha='center', va='bottom')
        plt.text(bins[n_max] + (bins[1] - bins[0]) / 2, -0.01, int(bins_max), ha='center', va='bottom')

        # 将xticks设为plt.hist返回的bins分布，bins位置为柱状图最左侧，且bins数量为n+1，其中包含最后一个bins的左侧坐标和右侧坐标
        plt.xticks(bins[:-1:10])
        plt.xlabel = 'area'
        plt.show()
        n, bins, _ = plt.hist(ratio, bins=10)
        for i in range(len(n)):
            plt.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.01, int(n[i]), ha='center', va='bottom')
        plt.xticks(bins[:-1])
        plt.xlabel = 'ratio'
        plt.show()

    def analyze_label_dist(self, labels: np.array):
        n, bins, _ = plt.hist(labels, bins=len(self.label_names), align='mid')
        for i in range(len(n)):
            plt.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.01, int(n[i]), ha='center', va='bottom')
        plt.xlabel = 'label distribution'
        plt.ylabel = 'times'

        plt.xticks(bins[:-1],
                   list(self.label_names),
                   color='blue',
                   rotation=60)
        plt.show()

    def analyze_bboxnums(self, bbox: list):
        bbox_bincount = np.zeros(len(bbox))
        for idx, bbox_tmp in enumerate(bbox):
            bbox_bincount[idx] = len(bbox_tmp)

        n, bins, _ = plt.hist(bbox_bincount, bins=np.array(bbox_bincount, dtype=np.uint8).max(),
                              align='mid')
        for i in range(len(n)):
            plt.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.01, int(n[i]), ha='center', va='bottom')
        plt.xticks(bins[:-1])
        plt.xlabel = 'bbox_nums'
        plt.show()

    def bbox_vision(self, img_path: str, bbox: np.array, labels: np.array):
        pass

    def _get_collect(self, id_):
        bbox = []
        labels = []

        annotation = et.parse(os.path.join(self.root, 'Annotations', f'{id_}.xml'))
        size = annotation.find('size')
        img_width = size.find('width').text
        img_height = size.find('height').text
        img_channel = size.find('depth').text
        img_size = [img_height, img_width, img_channel]
        for anno in annotation.findall('object'):
            name = anno.find('name').text
            try:
                labels.append(self.label_names.index(name))
            except ValueError as e:
                print(f'error xml name: {name}')
            # difficult.append(anno.find('difficult').text)
            bbox_info = anno.find('bndbox')
            xmin = bbox_info.find('xmin').text
            ymin = bbox_info.find('ymin').text
            xmax = bbox_info.find('xmax').text
            ymax = bbox_info.find('ymax').text
            bbox.append([xmin, ymin, xmax, ymax])

        bbox = np.array(bbox, dtype=np.uint16)
        labels = np.array(labels, dtype=np.uint16)
        img_path = os.path.join(self.root, 'JPEGImages', f'{id_}.jpg')
        return img_size, img_path, bbox, labels


if __name__ == '__main__':
    bb_analy = BboxAnalyze('../../SeaShips', idx_file='test')
    collect_data = bb_analy.get_collect_data
    bbox = collect_data['bbox']
    label = collect_data['labels']

    # 对所有图像中的bbox, labels进行concat
    bbox_cat = np.concatenate(bbox, axis=0)
    label_cat = np.concatenate(label, axis=0)
    bb_analy.analyze_bbox_dist(bbox_cat)
    bb_analy.analyze_label_dist(label_cat)
    bb_analy.analyze_bboxnums(bbox)



