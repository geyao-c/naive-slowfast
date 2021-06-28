import torch.utils.data as Data
import os
import cv2
import numpy as np
import utils
import logging

# 重写dataset都需要继承Dataset这个基类
class VideoDataset(Data.Dataset):
    def __init__(self, data_root, anno_path, classInd="", mode='train', frame_sample_rate=1, clip_len=8):
        super(VideoDataset, self).__init__()
        self.filenames, self.labels_indices = [], []
        self.short_side = [128, 160]
        self.mode = mode
        # 采样率，多少帧取一帧
        self.frame_sample_rate = frame_sample_rate
        # 每个视频使用多少帧作为输入(自认为这里的视频特指短视频)
        self.clip_len = clip_len
        # 裁剪图片大小
        self.crop_size = 112

        if (mode == 'validation'):
            if classInd == "":
                logging.error('please specify classInd')
                exit(-1)
            else:
                class2idx, _ = utils.get_class2idx(classInd)

        with open(anno_path) as f:
            fileline = f.readline()
            while (fileline != ""):
                # 去掉回车符
                fileline = fileline.split('\n')[0]
                if mode == 'train' or mode == 'training':
                    fileline = fileline.split(' ')
                    self.labels_indices.append(int(fileline[1]) - 1)
                    self.filenames.append(os.path.join(data_root, fileline[0]))
                else:
                    self.filenames.append(os.path.join(data_root, fileline))
                    fileline = fileline.split('/')
                    self.labels_indices.append(int(class2idx[fileline[0]]) - 1)
                fileline = f.readline()


    # 重写getitem函数
    def __getitem__(self, index):
        buffer = utils.loadvideo(self.filenames[index], self.short_side, self.frame_sample_rate)
        while (buffer is None or buffer.shape[0] <= self.clip_len):
            index = np.random.randint(self.__len__())
            buffer = utils.loadvideo(self.filenames[index])
        if (self.mode == 'train' or self.mode == 'training'):
            buffer = utils.randomflip(buffer)
        buffer = utils.crop(buffer, self.clip_len, self.crop_size)
        buffer = utils.normalize(buffer)
        buffer = utils.to_tensor(buffer)
        return buffer, self.labels_indices[index]

    #重写len函数
    def __len__(self):
        return self.filenames.__len__()


if __name__ == '__main__':
    data_root = 'E:/dataset/UCF-101'
    annotation_paths = './data/trainlist01.txt'
    video_dataset = VideoDataset(data_root, annotation_paths)
