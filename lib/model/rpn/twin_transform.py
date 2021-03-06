import torch
import numpy as np
import pdb

# 在计算坐标变化的时候是将anchor的表示形式变为中心坐标与长度
def twin_transform(ex_rois, gt_rois):
    # ex_rois will be (C, 2)
    # gt_rois will be (C, 2)
    # targets will be (C, 2)
    ex_lengths = ex_rois[:, 1] - ex_rois[:, 0] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_lengths  # 计算得到的每个anchor的中心坐标和长度

    gt_lengths = gt_rois[:, 1] - gt_rois[:, 0] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_lengths  # 计算得到的每个anchor对应的ground truth box，对应中心坐标和长度

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_lengths  # 计算两个坐标变换值
    targets_dl = torch.log(gt_lengths / ex_lengths)

    targets = np.stack(
        (targets_dx, targets_dl), 1)  # 对于每一个anchor，得到两个关系值shape:[4, num_anchor]
    return targets

# 结合rpn输出对所有初始框进行坐标变换
def twin_transform_inv(wins, deltas, batch_size):
    # wins will be (batch_size, C, 2)
    # deltas will be (batch_size, C, 2)
    # pred_wins will be (batch_size, C, 2)
    # 获得初始proposal的中心和长度信息
    lengths = wins[:, :, 1] - wins[:, :, 0] + 1.0
    ctr_x = wins[:, :, 0] + 0.5 * lengths
    # after the slice operation, dx will be (batch_size, C, 1), Not (batch_size, C)
    # 获得坐标变换信息
    dx = deltas[:, :, 0::2]
    dl = deltas[:, :, 1::2]
    # 获得变换后的proposal的中心和长度信息
    pred_ctr_x = dx * lengths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_l = torch.exp(dl) * lengths.unsqueeze(2)
    # 将改变后的proposal的中心和长度信息还原成中心和长度的版本
    pred_wins = deltas.clone()
    # x1
    pred_wins[:, :, 0::2] = pred_ctr_x - 0.5 * pred_l
    # x2
    pred_wins[:, :, 1::2] = pred_ctr_x + 0.5 * pred_l

    return pred_wins

# 严格限制proposal的两个边界在视频边界内
def clip_twins(wins, video_length):
    """
    Clip wins to video boundaries.
    """
    # wins.clamp_(0, video_length-1)
    # x1 >= 0
    wins[:, 0::2] = np.maximum(np.minimum(wins[:, 0::2], video_length - 1), 0)
    # x2 < im_shape[1]
    wins[:, 1::2] = np.maximum(np.minimum(wins[:, 1::2], video_length - 1), 0)
    return wins

def twin_transform_batch(ex_rois, gt_rois):
    # ex_rois will be (C, 2) -- anchors are the same for all the batchs.
    #              or (batch_size, C, 2)
    # gt_rois will be (batch_size, C, 2) or (batch_size, C, 3)
    # targes will be (batch_size, C, 2)
    # Note that the outer dimension equals to 1 will be discarded
    if ex_rois.dim() == 2:
        ex_lengths = ex_rois[:, 1] - ex_rois[:, 0] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_lengths

        gt_lengths = gt_rois[:, :, 1] - gt_rois[:, :, 0] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_lengths

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_lengths
        targets_dl = torch.log(gt_lengths / ex_lengths.view(1,-1).expand_as(gt_lengths))

    elif ex_rois.dim() == 3:
        ex_lengths = ex_rois[:, :, 1] - ex_rois[:, :, 0] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_lengths

        gt_lengths = gt_rois[:, :, 1] - gt_rois[:, :, 0] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_lengths

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_lengths
        targets_dl = torch.log(gt_lengths / ex_lengths)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dl),2)

    return targets

# 计算重合程度，两个框之间的重合区域的面积/两个区域一共加起来的面
def twins_overlaps(anchors, gt_twins):
    """
    anchors: (N, 2) ndarray of float
    gt_twins: (K, 2) ndarray of float
    overlaps: (N, K) ndarray of overlap between twins and query_twins
    """
    N = anchors.size(0)
    K = gt_twins.size(0)

    # gt_twins_len = (gt_twins[:,1] - gt_twins[:,0] + 1).view(1, K)
    # gt_len_zero = (gt_twin_len == 1)
    # anchors_len = (anchors[:,1] - anchors[:,0] + 1).view(N, 1)
    # anchors_len_zero = (anchors_len_zero==1)
    gt_twins_len = (gt_twins[:, 1] - gt_twins[:, 0] + 1).view(1, K)
    gt_len_zero = (gt_twins_len == 1)
    anchors_len = (anchors[:, 1] - anchors[:, 0] + 1).view(N, 1)
    anchors_len_zero = (anchors_len == 1)

    twins = anchors.view(N, 1, 2).expand(N, K, 2)
    query_twins = gt_twins.view(1, K, 2).expand(N, K, 2)

    ilen = torch.min(twins[:,:,1], query_twins[:,:,1]) - torch.max(twins[:,:,0], query_twins[:,:,0]) + 1
    ilen[ilen < 0] = 0

    ua = anchors_len + gt_twins_len - ilen
    overlaps = ilen / ua

    # mask the overlap
    overlaps.mask_fill_(gt_len_zero.view(1, K).expand(N, K), 0)
    overlaps.mask_fill_(anchors_len_zero.view(N, 1).expand(N, K), -1)

    return overlaps

def twins_overlaps_batch(anchors, gt_twins):
    """
    anchors:
        For RPN: (N, 2) ndarray of float or (batch_size, N, 2) ndarray of float
        For TDCNN: (batch_size, N, 3) ndarray of float
    gt_twins: (batch_size, K, 3) ndarray of float, (x1, x2, class_id)
    overlaps: (batch_size, N, K) ndarray of overlap between twins and query_twins
    """
    batch_size = gt_twins.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_twins.size(1)

        anchors = anchors.view(1, N, 2).expand(batch_size, N, 2).contiguous()
        gt_twins = gt_twins[:,:,:2].contiguous()


        gt_twins_x = (gt_twins[:,:,1] - gt_twins[:,:,0] + 1)
        gt_twins_len = gt_twins_x.view(batch_size, 1, K)

        anchors_twins_x = (anchors[:,:,1] - anchors[:,:,0] + 1)
        anchors_len = anchors_twins_x.view(batch_size, N, 1)

        gt_len_zero = (gt_twins_x == 1)
        anchors_len_zero = (anchors_twins_x == 1)

        twins = anchors.view(batch_size, N, 1, 2).expand(batch_size, N, K, 2)
        query_twins = gt_twins.view(batch_size, 1, K, 2).expand(batch_size, N, K, 2)

        ilen = (torch.min(twins[:,:,:,1], query_twins[:,:,:,1]) -
            torch.max(twins[:,:,:,0], query_twins[:,:,:,0]) + 1)
        ilen[ilen < 0] = 0

        ua = anchors_len + gt_twins_len - ilen
        overlaps = ilen / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_len_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_len_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_twins.size(1)

        if anchors.size(2) == 2:
            anchors = anchors[:,:,:2].contiguous()
        # (video_idx, x1, x2)
        else:
            anchors = anchors[:,:,1:3].contiguous()

        gt_twins = gt_twins[:,:,:2].contiguous()

        gt_twins_x = (gt_twins[:,:,1] - gt_twins[:,:,0] + 1)
        gt_twins_len = gt_twins_x.view(batch_size, 1, K)

        anchors_twins_x = (anchors[:,:,1] - anchors[:,:,0] + 1)
        anchors_len = anchors_twins_x.view(batch_size, N, 1)

        gt_len_zero = (gt_twins_x == 1)
        anchors_len_zero = (anchors_twins_x == 1)
        twins = anchors.view(batch_size, N, 1, 2).expand(batch_size, N, K, 2)
        query_twins = gt_twins.view(batch_size, 1, K, 2).expand(batch_size, N, K, 2)

        ilen = (torch.min(twins[:,:,:,1], query_twins[:,:,:,1]) -
            torch.max(twins[:,:,:,0], query_twins[:,:,:,0]) + 1)
        ilen[ilen < 0] = 0

        ua = anchors_len + gt_twins_len - ilen
        overlaps = ilen / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_len_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_len_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps
