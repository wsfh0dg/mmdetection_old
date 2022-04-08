# -*-  coding=utf-8 -*-
# @Time : 2022/4/8 9:39
# @Author : Scotty1373
# @File : custom_inerence.py
# @Software : PyCharm
import time

import mmcv
import torch
import argparse

from mmcv.runner import init_dist, wrap_fp16_model, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.models import build_detector
from mmdet.apis.test import single_gpu_test, multi_gpu_test
from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Custom Data Inference')
    parser.add_argument('config',
                        help='train/test config path')
    parser.add_argument('checkpoint',
                        help='pretrined chkp')
    parser.add_argument('out',
                        help='inference result output file')
    # 可选参数
    parser.add_argument('--visual',
                        help='visualization needed',
                        default=False)
    parser.add_argument('--from_config_datasets',
                        help='inference from test dataset',
                        default=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.from_config_datasets:
        # 从config文件中读取数据用于从配置文件中获得test数据集
        cfg = mmcv.Config.fromfile(args.config)

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # 创建基于配置文件的test数据集
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                      args.show_score_thr)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
        print(f'\nwrite test dataset result to {args.out}')
        mmcv.dump(outputs, args.out)
    else:
        root = 'imgs/'
        img_path = '000012.jpg'
        model = init_detector(config=args.config,
                              checkpoint=args.checkpoint,
                              device='cuda:0')
        result = inference_detector(model=model, imgs=root+img_path)
        pass
        time.time()





