from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from lib.model.utils.config import cfg
from lib.model.rpn.generate_anchors import generate_anchors
from lib.model.rpn.twin_transform import twin_transform_inv, clip_twins
from lib.model.nms.nms_wrapper import nms

import pdb
# 根据边界框回归系数变换锚点以生成变换锚点。 然后通过使用锚点作为前景区域的概率应用非最大抑制来修剪锚点的数量。
DEBUG = False
# 根据RPN的输出结果，提取出所需的目标框（roi)
class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular twins (called "anchors").
    属性包括锚框的生成，通常在特征图每个点生成10个锚框，这里记录了特征图的下采样倍数，因此生成的锚框坐标是对应原图而言的，运行文件generate_anchors.py，可以看到生成的锚框示例。
    [-4 , 11]
    [-12, 19]
    [-16, 23]
    [-20, 27]
    [-28, 35]
    [-32, 39]
    [-36, 43]
    [-44, 51]
    [-52, 59]
    [-60, 67]
    """

    def __init__(self, feat_stride, scales, out_scores=False):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(base_size=feat_stride, scales=np.array(scales))).float()
        self._num_anchors = self._anchors.size(0)   # 10个anchors
        self._out_scores = out_scores
        # TODO: add scale_ratio for video_len ??
        # rois blob: holds R regions of interest, each is a 3-tuple
        # (n, x1, x2) specifying an video batch index n and a
        # rectangle (x1, x2)
        # top[0].reshape(1, 3)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    # 通过矩阵操作将生成的k个anchor对应到特征图每个点对应的原图每个位置上
    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors:, :, :, :]
        twin_deltas = input[1]
        cfg_key = input[2]

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N   # 在将NMS应用于RPN之前，需要保留的顶级评分框数量为12000个
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 在将NMS应用于RPN之后，需要保留的顶级评分框数量为2000
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH   # nms阈值0.8
        min_size = cfg[cfg_key].RPN_MIN_SIZE   # 8

        # 1. Generate proposals from twin deltas and shifted anchors
        length, height, width = scores.shape[-3:]  # 16 1 1

        if DEBUG:
            print('score map size:{}'.format(scores.shape))

        batch_size = twin_deltas.size(0)  # 1

        # Enumerate all shifts
        shifts = np.arange(0, length) * self._feat_stride
        shifts = torch.from_numpy(shifts.astype(float))
        shifts = shifts.contiguous().type_as(scores)   # contiguous()把tensor变为连续分布形式

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 1) to get
        # shift anchors (K, A, 2)
        # reshape to (1, K*A, 2) shifted anchors
        # expand to (batch_size, K*A, 2)
        A = self._num_anchors    # 10
        K = shifts.size(0)    # 16
        # 原始anchor+shifts，就可以实现逐行算anchor，得到所有anchor
        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 2) + shifts.view(K, 1, 1)
        anchors = anchors.view(1, K * A, 2).expand(batch_size, K * A, 2)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # twin deltas will be (batch_size, 2 * A, L, H, W) format
        # transpose to (batch_size, L, H, W, 2 * A)
        # reshape to (batch_size, L * H * W * A, 2) where rows are ordered by (l, h, w, a)
        # in slowest to fastest order
        # 利用坐标偏移量twin_deltas，得到修正后的anchor，再将超出边界的anchor部分裁剪至边界，得到proposals

        twin_deltas = twin_deltas.permute(0, 2, 3, 4, 1).contiguous()
        twin_deltas = twin_deltas.view(batch_size, -1, 2)

        # Same story for the scores:
        # scores are(batch_size, A, L, H, W) format
        # transpose to (batch_size, L, H, W, A)
        # reshape to (batch_size, L * H * W * A) where rows are ordered by (l, h, w, a)
        scores = scores.permute(0, 2, 3, 4, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via twin transformations
        # 根据anchor和偏移量计算proposal
        proposals = twin_transform_inv(anchors, twin_deltas, batch_size)

        # 2. clip predicted boxes to video, 将proposals限制在视频范围内，超出边界，则将边界赋值
        proposals = clip_twins(proposals, length * self._feat_stride)  # 矩阵求逆

        # 将前景得分排序，测试阶段取前12000个得分较高的anchor，得到相应的坐标和得分，通过nms方法计算得分较高并且去掉不符合阈值的anchor.
        # 在符合nms阈值的anchor中再取前排名前2000的anchor，若nms返回的anchor数量不足300，则用0填充。
        # 3. remove predicted twins with either length < threshold
        # assign the score to 0 if it's non keep.
        no_keep = self._filter_twins_reverse(proposals, min_size)
        scores[no_keep] = 0

        scores_keep = scores
        proposals_keep = proposals
        # sorted in descending order
        _, order = torch.sort(scores_keep, 1, True)

        # print ("scores_keep {}".format(scores_keep.shape))
        # print ("proposals_keep {}".format(proposals_keep.shape))
        # print ("order {}".format(order.shape))

        output = scores.new(batch_size, post_nms_topN, 3).zero_()

        if self._out_scores:
            output_score = scores.new(batch_size, post_nms_topN, 2).zero_()

        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]  # 按score排序后nms

            # numel 函数返回元素个数
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]  # 测试阶段取前6000个得分的索引

            # 取前6000 的索引对应的区域和得分[6000,4],[6000,1]，会重新生成proposals_single的下标0:5999
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            # torch.cat 在第一维度拼接区域和得分矩阵[6000,5]
            # 进行非极大值抑制
            # keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            # keep_idx_i 返回通过nms阈值限制之后的索引，该索引基于6000的下标
            keep_idx_i = keep_idx_i.long().view(-1)

            # 挑选NMS之后的数值最大proposal, 按设置好的nms后框数量限制参数 --RPN_POST_NMS_TOP_N, 截取RPN_POST_NMS_TOP_N个proposals
            # 取该索引的前2000个建议区域
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.将不足2000的区域建议补0
            num_proposal = proposals_single.size(0)
            # print ("num_proposal: ", num_proposal)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single

            if self._out_scores:
                output_score[i, :, 0] = i
                output_score[i, :num_proposal, 1] = scores_single

        if self._out_scores:
            return output, output_score
        else:
            return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_twins_reverse(self, twins, min_size):
        """get the keep index of all twins with length smaller than min_size.
        twins will be (batch_size, C, 2), keep will be (batch_size, C)"""
        ls = twins[:, :, 1] - twins[:, :, 0] + 1
        no_keep = (ls < min_size)
        return no_keep
