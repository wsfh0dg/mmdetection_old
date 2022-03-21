# -*- coding: utf-8 -*-
import json
import mmcv
from tqdm import tqdm

train_file_path = "instances_train2017.json"
val_file_path = "instances_val2017.json"

train = "train2017.json"
val = "val2017.json"

if __name__ == "__main__":
    file_list = [train_file_path, val_file_path]
    file_saver = [train, val]
    for idx, path in enumerate(file_list):
        ann = mmcv.load(path)
        json_list = tqdm(ann['annotations'])
        for item in json_list:
            item['segmentation'] = []
        with open(file_saver[idx], "w") as fp:
            json.dump(ann, fp)



    
