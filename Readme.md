This is the pytorch implementation of slowfast. We can use this code to train the slowfast network on the UCF-101 dataset, and use  trained model to perform action recognition on a single input video . However, it is probably that the model trained with this code can't achieve the performance of the original paper. Because, the purpose of  this code is learning, such as how to use pytorch , how to writer video dataloader, etc.

#### environment

1. pytorch 1.0 or higher
2. python 3.6

#### train

```
python train.py --data_root '/content/drive/MyDrive/UCF-101' \
				--train_anno_path './annotation/trainlist01.txt' \
				--test_anno_path './annotation/testlist01.txt' \
				--classInd './annotation/classInd.txt'
```

#### evaluate

```
python evaluate --videopath 'videoname' \
				--pretrain_model_path './model/UCF-101.model'
				--classInd './annotation/classInd.txt'
```

