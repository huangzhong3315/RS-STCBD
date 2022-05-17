import numpy as np
import time
import pdb
from IPython import embed

# 根据anchor的参数生成anchors，也就是anchor框，基础的大小为base_size=8,即8X8
# base_size指定了最初的类似感受野的区域大小 也就是feature map上一点对应到原图的大小为8
# scales 输入的区域进行三种倍数，2^3=8，2^4=16，2^5=32倍的放大 8-> (8*8)、(8*16)、(8*32)
def generate_anchors(base_size=8, scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect 
    scales wrt a reference (0, 7) window.
    """
    #在时间维度上进行proposal，所以维数为1
    base_anchor = np.array([1, base_size]) - 1  # [0, 7] 表示最基本的一个大小为8的区域，2个值，分别代表这个区域的长度坐标
    anchors = _scale_enum(base_anchor, scales)  # 返回不通长度比的anchor
    return anchors

# 计算anchor的长度和中心点
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    l = anchor[1] - anchor[0] + 1     # anchor坐标中：x2 - x1 + 1 = 长度
    x_ctr = anchor[0] + 0.5 * (l - 1)  # 计算anchor的中心像素的坐标，x1+1/2(l-1)
    return l, x_ctr 

def _mkanchors(ls, x_ctr):
    """
    Given a vector of lengths (ls) around a center
    (x_ctr), output a set of anchors (windows).
    """
    # 给定一组计算好的ls以及中心点， 输出10个anchor框坐标，也就是anchors
    ls = ls[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ls - 1),
                         x_ctr + 0.5 * (ls - 1)))
    return anchors

# 大小的变化，没有比例的变化 计算不同长度尺度下的anchor坐标
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    l, x_ctr = _whctrs(anchor)  # 找到anchor的长度和中心点
    ls = l * scales  # 返回anchor的大小
    anchors = _mkanchors(ls, x_ctr) # 返回新的不同长度比的anchor
    return anchors

if __name__ == '__main__':
    t = time.time()
    a = generate_anchors(scales=np.array([2, 4, 5, 6, 8, 9, 10, 12, 14, 16]))  # 10个scale
    print(time.time() - t)
    print(a)
    embed()
