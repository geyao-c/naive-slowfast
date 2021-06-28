import argparse
import utils
import logging
import torch
import slowfastnet
import os
import numpy as np

log_format = '%(asctime)s - %(module)s.%(funcName)s.%(lineno)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='evaluate given video')
    parser.add_argument('--videopath', type=str, default='/content/drive/MyDrive/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g04_c04.avi',help='plase give video path')
    parser.add_argument('--pretrain_model_path', default='./model/UCF_101.pt',type=str, help='pretrained model')
    parser.add_argument('--classInd', default='./annotation/classInd.txt', help='class label to index path')
    parser.add_argument('--clip_len', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=112)
    parser.add_argument('--class_num', type=int, default=101)
    return parser.parse_args()

def video2tensor(videopath, clip_len=64, crop_size=112):
    buffer = utils.loadvideo(videopath)
    if buffer is None:
        logging.error('the video can be read')
        exit(-1)
    buffer = utils.crop(buffer, clip_len, crop_size)
    buffer = utils.normalize(buffer)
    buffer = utils.to_tensor(buffer)
    buffer = torch.tensor([buffer.tolist()])
    return buffer

if __name__ == '__main__':
    args = get_args()

    # 将视频转换为能够作为网络输入的格式
    logging.info('transfer video to tensor')
    buffer = video2tensor(args.videopath).to(device)

    # 加载模型
    logging.info('load model')
    model = slowfastnet.resnet50(class_num=args.class_num)
    if not os.path.exists(args.pretrain_model_path):
        logging.error('the model is not exist')
        exit(-1)
    pretrained_dict = torch.load(args.pretrain_model_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    logging.info('begin analyzing')
    model.to(device).eval()
    predict = np.exp(model(buffer).detach())
    p_sum = predict.sum().item()

    predict = predict.contiguous().view(-1)
    values, indices = torch.topk(predict, k=20)
    _, idx2class = utils.get_class2idx(args.classInd)
    idx = 0
    for i in indices.tolist():
        logging.info('{}: {:.2f}%'.format(idx2class[i + 1], values.tolist()[idx] / p_sum * 100))
        idx += 1




