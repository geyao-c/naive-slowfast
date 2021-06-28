import os
import torch
import logging
import numpy as np
import cv2


def randomflip(buffer):
    # 0.5的几率对视频进行水平翻转
    if np.random.random() < 0.5:
        for i, frame in enumerate(buffer):
            buffer[i] = cv2.flip(frame, 1)
    return buffer

def crop(buffer, clip_len, crop_size):
    # 对视频进行时间维和空间维的裁剪
    time_index = np.random.randint(buffer.shape[0] - clip_len)
    height_index = np.random.randint(buffer.shape[1] - crop_size)
    width_index = np.random.randint(buffer.shape[2] - crop_size)

    # 利用随机数在空间和时间上制造抖动
    buffer = buffer[time_index: time_index + clip_len, height_index: height_index + crop_size, \
             width_index: width_index + crop_size, :]
    return buffer


# 对视频进行标准化
def normalize(buffer):
    for i, frame in enumerate(buffer):
        buffer[i] = frame / 255
    return buffer

def to_tensor(buffer):
    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose((3, 0, 1, 2))

def loadvideo(filename, short_side=(128, 160), frame_sample_rate=1):
    if not os.path.exists(filename):
        logging.error('{} is not exist'.format(filename))
        exit(-1)

    # 读取视频
    cap = cv2.VideoCapture(filename)
    # 获取视频属性
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0 or frame_count == 0:
        logging.info(
            'frame_width: {}, frame_height: {}, frame_count: {}'.format(frame_width, frame_height, frame_count))
        logging.info('{} has some value error'.format(filename))
        return None

    # print('frame count: %d, frame_width: %d, frame height: %d' % (frame_count, frame_width, frame_height))
    # 调整视频大小
    if frame_height < frame_width:
        resized_height = np.random.randint(short_side[0], short_side[1])
        resized_width = int(resized_height / frame_height * frame_width)
    else:
        resized_width = np.random.randint(short_side[0], short_side[1])
        resized_height = int(resized_width / frame_width * frame_height)
    # 总采样帧数
    frame_count_sample = frame_count // frame_sample_rate
    buffer = np.empty((frame_count_sample, resized_height, resized_width, 3), dtype=np.float32)

    count, sample_count = 1, 0
    while (count <= frame_count):
        isok, frame = cap.read()
        if isok is False:
            break
        if (count % frame_sample_rate == 0):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resized_width, resized_height))
            buffer[sample_count] = frame
            sample_count += 1
        count += 1
    cap.release()
    return buffer

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0., 0., 0., 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def remove_file(foldername):
    filelist = os.listdir(foldername)
    for file in filelist:
        print(file)
        if file == 'jupyter_code' or file == 'UCF-101' or file == 'UCF101.rar':
            pass
        else:
            os.remove(file)
            print('%s has been removed' % (file))

def accuracy(predict, target, topk=(1,)):
    result = []
    maxk, batch = max(topk), predict.shape[0]
    _, pred = torch.topk(predict, maxk, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for k in topk:
        correct_k = correct[: k, :].contiguous().sum().float().item()
        result.append(correct_k / batch)

    return result

def get_class2idx(classInd):
    class2idx = {}
    idx2class = {}
    with open(classInd) as f:
        while(1):
            fileline = f.readline()
            if (fileline == ""):
                break
            fileline = fileline.split('\n')[0]
            fileline = fileline.split(' ')
            class2idx[fileline[1]] = int(fileline[0])
            idx2class[int(fileline[0])] = fileline[1]
    return class2idx, idx2class

if __name__ == '__main__':
    # remove_file('./')
    a = 1