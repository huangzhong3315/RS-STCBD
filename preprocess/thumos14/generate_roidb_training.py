import os
import numpy as np
from util import *
import pickle
import copy

FPS = 25
ext = '.mp4'
LENGTH = 768  #
min_length = 3
overlap_thresh = 0.7
STEP = LENGTH / 4
WINS = [LENGTH * 1]

# FRAME_DIR = 'F:/TH14/frame'
FRAME_DIR = '/home/aqnu/tmy/frame'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')  # F:/TH14/frame/annotation_

print('Generate Training Segments')
train_segment = dataset_label_parser(META_DIR+'val', 'val', use_ambiguous=False)

def generate_roi(rois, video, start, end, stride, split):
  tmp = {}
  tmp['wins'] = (rois[:,:2] - start) / stride
  tmp['durations'] = tmp['wins'][:, 1] - tmp['wins'][:, 0]
  tmp['gt_classes'] = rois[:, 2]
  tmp['max_classes'] = rois[:, 2]
  tmp['max_overlaps'] = np.ones(len(rois))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
  tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
  if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg')):
    print(os.path.join(FRAME_DIR, split, video, 'image_'+str(end-1).zfill(5) + '.jpg'))
    raise
  return tmp



def generate_roidb(split, segment):
  VIDEO_PATH = os.path.join(FRAME_DIR, split)     # 'F:/TH14/frame\\val'
  video_list = set(os.listdir(VIDEO_PATH))     # 视频文件名
  duration = []
  roidb = []
  for vid in segment:
    if vid in video_list:
      length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))  # 视频切成帧的图片数量 video_validation_0000173 有2101张
      db = np.array(segment[vid])
      if len(db) == 0:
        continue
      db[:, :2] = db[:, :2] * FPS

      for win in WINS:
        # inner of windows
        stride = int(win / LENGTH)
        # Outer of windows
        step = int(stride * STEP)
        # Forward Direction
        for start in range(0, max(1, length - win + 1), step):
          end = min(start + win, length)
          assert end <= length
          rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:, 1] <= start))]   # 时间范围内数据

          # Remove duration less than min_length
          if len(rois) > 0:
            duration = rois[:, 1] - rois[:, 0]
            rois = rois[duration >= min_length]

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0])) * 1.0
            overlap = time_in_wins / (rois[:, 1] - rois[:, 0])
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois = rois[overlap >= overlap_thresh]

          # Append data
          if len(rois) > 0:
            rois[:, 0] = np.maximum(start, rois[:, 0])
            rois[:, 1] = np.minimum(end, rois[:, 1])
            tmp = generate_roi(rois, vid, start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
              flipped_tmp = copy.deepcopy(tmp)
              flipped_tmp['flipped'] = True
              roidb.append(flipped_tmp)

  return roidb


if __name__ == '__main__':
    USE_FLIPPED = True
    train_roidb = generate_roidb('val', train_segment)
    print(len(train_roidb))
    print("Save dictionary")
    pickle.dump(train_roidb, open('train_data_25fps_flipped.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

