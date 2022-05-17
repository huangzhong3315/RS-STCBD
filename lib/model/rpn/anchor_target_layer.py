from __future__ import absolute_import
"""
主要功能是产生anchor， 并对anchor进行评分等操作
anchor_target_layer 主要针对RPN的输出进行处理， 对RPN的输出结果加工，对anchor打上标签
然后通过与groundtrue的信息比对， 计算出与真实框的偏差，这些都指向了为计算loss误差做准备
"""

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from lib.model.utils.config import cfg
from .generate_anchors import generate_anchors
from .twin_transform import clip_twins, twins_overlaps_batch, twin_transform_batch

import pdb
# 挑选好的anchor，用于训练RPN网络
# 区分前景和背景
# 为前景边界框生成好的边界回归系数
DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and time window regression targets.
    """
    def __init__(self, feat_stride, scales):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(base_size=feat_stride, scales=np.array(scales))).float()
        # generate_anchors函数根据scale产生坐标变换， 这些坐标变换是让中心的产生不同的anchor
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0


    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted twin deltas at cell i to each of the 9 anchors
        # filter out-of-video anchors
        # measure GT overlap

        rpn_cls_score = input[0]
        # GT boxes (batch_size, n, 3), each row of gt box contains(x1, x2, label)
        gt_twins = input[1]
        # im_info = input[2]
        # num_boxes = input[3]

        # map of shape (..., L, H, W)
        length, height, width = rpn_cls_score.shape[-3:]  # 得到特征提取层最后一层feature map的长度、高度和宽度

        batch_size = gt_twins.size(0)
        # Enumerate all shifts
        shifts = np.arange(0, length) * self._feat_stride  # 长度上的偏移值
        shifts = torch.from_numpy(shifts.astype(float))
        shifts = shifts.contiguous().type_as(rpn_cls_score)

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 1) to get
        # shift anchors (K, A, 2)
        # reshape to (K*A, 2) shifted anchors
        A = self._num_anchors  # A是anchor的数量
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(rpn_cls_score)  # move to specific gpu.
        # 将generate_anchors.py得到的anchor (最左上角的那个anchor), 依次加上偏移
        all_anchors = self._anchors.view(1, A, 2) + shifts.view(K, 1, 1)
        all_anchors = all_anchors.view(K * A, 2)  # 所有anchors的坐标值
        total_anchors = int(K * A)
        # 去除超过视频边界的proposals, 只保留那些在视频之内的anchor索引
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] < long(length * self._feat_stride) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors inds_inside 一维索引
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_twins.new(batch_size, inds_inside.size(0)).fill_(-1)
        twin_inside_weights = gt_twins.new(batch_size, inds_inside.size(0)).zero_()    # 计算前景Anchor的损失
        twin_outside_weights = gt_twins.new(batch_size, inds_inside.size(0)).zero_()
        # print("anchors {}".format(anchors.shape)) #(876, 2)
        # print("gt_twins {}".format(gt_twins.shape)) #(1, 6, 3)
        # assume anchors(batch_size, N, 2) and gt_wins(batch_size, K, 2), respectively, overlaps will be (batch_size, N, K)
        overlaps = twins_overlaps_batch(anchors, gt_twins)
        # find max_overlaps for each dt: (batch_size, N)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)  # argmax_overlaps是每个anchor对应最大overlap的gt_twins的下标
        # find max_overlaps for each gt: (batch_size, K)
        gt_max_overlaps, _ = torch.max(overlaps, 1)  # gt_max_overlaps存储每一个gt框和视频中某一个anchor的最大重叠率

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # 首先分配bg标签，这样积极地标签可以打击他们 cfg.TRAIN.RPN_NEGATIVE_OVERLAP=0.3 小于0.3为负样本,label=0
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1  # 对于每一个gt, 最大重叠率最大的那个anchor为fg

        # fg label: above threshold IOU  对于每个gt, 最大重叠率大于0.7的为fg
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # 分配bg标签，以便负标签可以打击积极
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # 要求的fg数量

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_twins).long()
                # np.random.permutation随机排列一个序列，或者数组
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]] # 随机抽取fg_inds.size(0)-num_fg个样本丢弃
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_twins).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_twins.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        twin_targets = _compute_targets_batch(anchors, gt_twins.view(-1,3)[argmax_overlaps.view(-1), :].view(batch_size, -1, 3))

        # use a single value instead of 2 values for easy index.
        twin_inside_weights[labels==1] = cfg.TRAIN.RPN_TWIN_INSIDE_WEIGHTS[0]

        # 对正负样本的坐标的权值做初始化 样本的均匀加权（给定非均匀抽样）
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT
            negative_weights = 1 - positive_weights
        # 正负样本都设置twin_outside_weights
        twin_outside_weights[labels == 1] = positive_weights
        twin_outside_weights[labels == 0] = negative_weights

        # total_anchors所有图像中所有anchors的数量为K*A, labels一维数组， _unmap赋值
        # 给所有视频中的anchors打上label,正负样本分别为0,1，其他为-1
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        twin_targets = _unmap(twin_targets, total_anchors, inds_inside, batch_size, fill=0) # 给所有的anchors赋上四个偏移量，其余为0
        twin_inside_weights = _unmap(twin_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        twin_outside_weights = _unmap(twin_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []
        # 将labels数组变形
        labels = labels.view(batch_size,length, height, width, A).permute(0,4,1,2,3).contiguous()
        labels = labels.view(batch_size, 1, A * length, height, width)
        outputs.append(labels)

        twin_targets = twin_targets.view(batch_size,length, height, width, A*2).permute(0,4,1,2,3).contiguous()
        outputs.append(twin_targets)

        anchors_count = twin_inside_weights.size(1)
        twin_inside_weights = twin_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 2)

        twin_inside_weights = twin_inside_weights.contiguous().view(batch_size,length, height, width, 2*A)\
                            .permute(0,4,1,2,3).contiguous()

        outputs.append(twin_inside_weights)

        twin_outside_weights = twin_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 2)
        twin_outside_weights = twin_outside_weights.contiguous().view(batch_size,length, height, width, 2*A)\
                            .permute(0,4,1,2,3).contiguous()
        outputs.append(twin_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    # 实现将之前在所有图像上产生的anchor都赋上label, twin_targets, twin_outside_weights, twin_inside_weights属性

    if data.dim() == 2:
        ret = data.new(batch_size, count).fill_(fill)
        ret[:, inds] = data
    # for twin_targets
    else:
        ret = data.new(batch_size, count, data.size(2)).fill_(fill)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an video. 计算视频的边界框回归目标"""

    return twin_transform_batch(ex_rois, gt_rois[:, :, :2])
