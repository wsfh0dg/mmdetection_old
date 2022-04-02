# -*-  coding=utf-8 -*-
# @Time : 2022/4/1 16:52
# @Author : Scotty1373
# @File : msg_hook.py
# @Software : PyCharm
import torch
from mmcv.runner import Hook, HOOKS
import requests


def msg_send(name, ep, log_dict):
    content_str = f"Epoch={ep}, " \
                  f"loss_rpn_cls={log_dict['loss_rpn_cls'][-1]:.4f}, " \
                  f"loss_rpn_bbox={log_dict['loss_rpn_bbox'][-1]:.4f}, " \
                  f"loss_roi_cls={log_dict['loss_cls'][-1]:.4f}, " \
                  f"loss_roi_bbox={log_dict['loss_bbox'][-1]:.4f}, " \
                  f"acc={log_dict['acc'][-1]}, " \
                  f"total_loss={log_dict['loss'][-1]:.4f}"
    resp = requests.get(
        "https://www.autodl.com/api/v1/wechat/message/push?token={token}&title={title}&name={name}&content={content}".format(
            token="e571d1889d78",
            title="Wechat MSG",
            name=name,
            content=content_str))
    print(resp.content.decode())
    del resp


@HOOKS.register_module()
class SendMsgInfoHook(Hook):
    def __init__(self, send=True):
        self.send = send

    """Send message to WeChat Terminal"""
    def after_train_epoch(self, runner):
        epoch = runner.epoch
        model_name = runner.model_name
        log_dict = runner.log_buffer.val_history
        # content_str = f"Epoch={epoch}, " \
        #               f"loss_rpn_cls={log_dict['loss_rpn_cls'][-1]:.4f}, " \
        #               f"loss_rpn_bbox={log_dict['loss_rpn_bbox'][-1]:.4f}, " \
        #               f"loss_roi_cls={log_dict['loss_cls'][-1]:.4f}, " \
        #               f"loss_roi_bbox={log_dict['loss_bbox'][-1]:.4f}, " \
        #               f"acc={log_dict['acc'][-1]}, " \
        #               f"total_loss={log_dict['loss'][-1]:.4f}"
        if not epoch % 4:
            # print(content_str)
            msg_send(model_name, epoch, log_dict)




