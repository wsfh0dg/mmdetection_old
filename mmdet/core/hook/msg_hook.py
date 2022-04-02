# -*-  coding=utf-8 -*-
# @Time : 2022/4/1 16:52
# @Author : Scotty1373
# @File : msg_hook.py
# @Software : PyCharm
import torch
from mmcv.runner import Hook, HOOKS
import requests

@HOOKS.register_module()
class SendMsgInfoHook(Hook):
    """Send message to WeChat Terminal"""
    def after_train_epoch(self, runner):
        epoch = runner.epochs
        model_name = runner.model_name
        log_dict = runner.log_buffer.val_history
        if epoch % 4:
            self.msg_send(model_name, epoch, log_dict)

    @staticmethod
    def msg_send(name, ep, log_dict):
        content_str = f"Epoch={ep}, " \
                      f"loss_rpn_cls={log_dict['loss_rpn_cls']}, " \
                      f"loss_rpn_bbox={log_dict['loss_rpn_bbox']}, " \
                      f"loss_roi_cls={log_dict['loss_cls']}, " \
                      f"loss_roi_bbox={log_dict['loss_bbox']}, " \
                      f"acc={log_dict['acc']}, " \
                      f"total_loss={log_dict['loss']}"
        resp = requests.get("https://www.autodl.com/api/v1/wechat/message/push?token={token}&title={title}&name={name}&content={content}".format(
                token="e571d1889d78",
                title="Wechat MSG",
                name=name,
                content=content_str))
        print(resp.content.decode())

