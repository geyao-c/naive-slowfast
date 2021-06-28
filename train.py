import torch
import logging
import argparse
import dataset
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import slowfastnet
from torch import nn, optim
import utils
import time
import os
from tensorboardX import SummaryWriter

log_format = "%(asctime)s - %(module)s.%(funcName)s.%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='model train')
    parser.add_argument('--data_root', default='/content/drive/MyDrive/UCF-101',\
                        help='UCF-101 dataset root path')
    parser.add_argument('--train_anno_path', default='./annotation/trainlist01.txt',\
                        help='train dataset annotation')
    parser.add_argument('--test_anno_path', default='./annotation/testlist01.txt',\
                        help='test dataset annotation')
    parser.add_argument('--classInd', default='./annotation/classInd.txt',\
                        help='class label to index path')
    parser.add_argument('--clip_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=101)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--model_save_dir', default='./model')
    parser.add_argument('--log_save_dir', default='./log')
    parser.add_argument('--num_epochs', type=int, default=60)
    return parser.parse_args()

def train_or_valid(model, mode, train_dataloader, epoch, loss, optimizer, device):
    logging.info('{} in {}'.format(mode, device))

    if mode == 'train':
        model.to(device).train()
    elif mode == 'validation':
        model.to(device).eval()
    else:
        logging.error('the model mode only can be train and validation')
        exit(-1)

    sum_loss, top1, top5 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    start = time.time()
    for step, (X, Y) in enumerate(train_dataloader):
        X, Y = X.to(device), Y.to(device)
        Y_hat = model(X)
        l = loss(Y_hat, Y)
        if mode == 'train':
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        top1_accu, top5_accu = utils.accuracy(Y_hat, Y, (1, 5))
        batch = Y.shape[0]
        sum_loss.update(l.item(), batch)
        top1.update(top1_accu, batch)
        top5.update(top5_accu, batch)

        if (step + 1) % 10 == 0:
            end = time.time()
            logging.info('------------------------------------------------')
            logging.info('{} epoch: [{}][{}/{}]'.format(mode, epoch + 1, step, len(train_dataloader)))
            logging.info("loss: {:.5f}".format(sum_loss.avg))
            logging.info('top1 accuracy is {:.2f}%, top5 acuuracy is {:.2f}%'.format(
                top1.avg * 100, top5.avg * 100
            ))
            logging.info("time cost {:.3f}".format(end - start))
            start = time.time()
    return sum_loss.avg, top1.avg, top5.avg

if __name__ == '__main__':

    args = get_args()

    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join(args.log_save_dir, curr_time)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # 加载数据
    logging.info('data load')
    train_dataset = dataset.VideoDataset(args.data_root, args.train_anno_path, mode='train', \
                                         frame_sample_rate=1, clip_len=args.clip_len)
    train_dataloader = Data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataset = dataset.VideoDataset(args.data_root, args.test_anno_path, args.classInd, mode='validation', \
                                        frame_sample_rate=1, clip_len=args.clip_len)
    test_dataloader = Data.DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers)

    # 加载模型
    logging.info('load model')
    model = slowfastnet.resnet50(class_num=101)
    if args.pretrained is not None:
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        logging.info('load pretrain model')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    # 开始训练
    for epoch in range(args.num_epochs):
        train_loss, top1_accu, top5_accu = train_or_valid(model, 'train', train_dataloader, epoch, loss, optimizer, device)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_top1_accu', top1_accu, epoch)
        writer.add_scalar('train_top5_accu', top5_accu, epoch)
        logging.info('train epoch {}, train_loss is {}, top1 accuracy is {}, top5 accuracy is {}'\
                     .format(epoch, train_loss, top1_accu, top5_accu))

        if epoch % 2 == 0:
            valid_loss, valid_top1_accu, valid_top5_accu = train_or_valid(model, 'validation', test_dataloader, epoch, loss,
                                                              optimizer, device)
            writer.add_scalar('valid_loss', valid_loss, epoch)
            writer.add_scalar('valid_top1_accu', valid_top1_accu, epoch)
            writer.add_scalar('valid_top5_accu', valid_top5_accu, epoch)
            logging.info('validation epoch {}, train_loss is {}, top1 accuracy is {}, top5 accuracy is {}' \
                         .format(epoch, valid_loss, valid_top1_accu, valid_top5_accu))

        # 存储模型
        if epoch % 1 == 0:
            curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            checkpoint = os.path.join(args.model_save_dir, curr_time + '.pt')
            torch.save(model.state_dict(), checkpoint)

        scheduler.step()



