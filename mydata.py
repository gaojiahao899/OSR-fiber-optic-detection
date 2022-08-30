
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset
import pandas as pd

import torchvision
import numpy as np
import torchvision.transforms as transforms

import os
import argparse

import backbones.net as models

from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax
from Modelbuilder import Network
from Plotter import plot_feature

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='MyNet', choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--train_class_num', default=2, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=3, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')
parser.add_argument('--embed_dim', default=2, type=int, help='embedding feature dimension嵌入特征参数')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')

#Parameters for weibull distribution fitting.
parser.add_argument('--weibull_tail', default=20, type=int, help='Classes used in testing')
parser.add_argument('--weibull_alpha', default=2, type=int, help='Classes used in testing')
parser.add_argument('--weibull_threshold', default=0.7, type=float, help='Classes used in testing')


# Parameters for stage plotting
parser.add_argument('--plot', default=True, action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.每个类中最多要绘制的示例,0表示所有。')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')

args = parser.parse_args()

#准备数据集
class MyDataset(Dataset):
   # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
   # 能够通过index得到数据集的数据，能够通过len得到数据集大小

   def __init__(self, data_tensor, target_tensor):
       self.data_tensor = data_tensor
       self.target_tensor = target_tensor

   def __getitem__(self, index):
       return self.data_tensor[index], self.target_tensor[index]

   def __len__(self):
       return self.data_tensor.size(0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # checkpoint
    args.checkpoint = 't_n_fequence_feature0328/checkpoints_5_6_1/' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # folder to save figures
    args.plotfolder = 't_n_fequence_feature0328/checkpoints_5_6_1/' + args.arch + '/plotter'
    if not os.path.isdir(args.plotfolder):
        mkdir_p(args.plotfolder)

    # Data
    print('==> Preparing data..')

    train_url = "E:\exercise_4\Kmeans/t_n_fequence_feature0328\datasets/n_smote_fequence_feature_ -label.csv" 
    names = ['feature0','feature1','feature2','feature3','feature4','feature5',
    'feature6','feature7','feature8','feature9','feature10','feature11',
    'feature12','feature13','feature14','feature15',
    'feature16','feature17','class'] 
    train_dataset = pd.read_csv(train_url, names=names)
    train_array = train_dataset.values
    x_train = train_array[:,0:18]
    y_train = train_array[:,18]
 

    test_url = "E:\exercise_4\Kmeans/t_n_fequence_feature0328\datasets\s_n_smote_fequence_feature_-label.csv" 
    names = ['feature0','feature1','feature2','feature3','feature4','feature5',
    'feature6','feature7','feature8','feature9','feature10','feature11',
    'feature12','feature13','feature14','feature15',
    'feature16','feature17','class'] 
    test_dataset = pd.read_csv(test_url, names=names)
    test_array = test_dataset.values
    x_test = test_array[:,0:18]
    y_test = test_array[:,18]

    # 生成数据
    # 训练数据
    x_train_all = np.array(x_train).astype(np.float32)
    train_dataset_X = torch.from_numpy(x_train_all)
    train_dataset_Y = np.array(y_train).astype(np.int64)

    trainset = MyDataset(train_dataset_X, train_dataset_Y)  # 将数据封装成Dataset
    # print('tensor_data[0]: ', trainset[0])
    # print('len(tensor_data): ', len(trainset))

    # 测试数据
    x_test_all = np.array(x_test).astype(np.float32)
    test_dataset_X = torch.from_numpy(x_test_all)
    test_dataset_Y = torch.tensor(y_test.astype(float)).long()

    testset = MyDataset(test_dataset_X, test_dataset_Y)  # 将数据封装成Dataset
    # print('tensor_data[0]: ', testset[0])
    # print('len(tensor_data): ', len(testset))

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    # for data, target in testloader: 
    #     ad = data
    #     ta = target
    # print(ad)
    # print(ta)

    # Model
    net = Network(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim)
    fea_dim = net.classifier.in_features
    net = net.to(device)
   
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss','Train Acc.', 'Test Loss', 'Test Acc.'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)    

    epoch=0
    if not args.evaluate:
        for epoch in range(start_epoch, args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
            adjust_learning_rate(optimizer, epoch, args.lr, step=40)
            train_loss, train_acc = train(net,trainloader,optimizer,criterion,device)
            save_model(net, None, epoch, os.path.join(args.checkpoint,'last_model.pth'))
            test_loss, test_acc = 0, 0
            logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc])
            plot_feature(net, trainloader, device, args.plotfolder, epoch=epoch,
                         plot_class_num=args.train_class_num, maximum=args.plot_max, plot_quality=args.plot_quality)
            test(epoch, net, trainloader, testloader, criterion, device)
    
    test(99999, net, trainloader, testloader, criterion, device)
    plot_feature(net, testloader, device, args.plotfolder, epoch="test",
                 plot_class_num=args.train_class_num+1, maximum=args.plot_max, plot_quality=args.plot_quality)
    logger.close()


# Training
def train(net,trainloader,optimizer,criterion,device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.reshape(1,1,1,18)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total

def test(epoch, net,trainloader,  testloader,criterion, device):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs = inputs.reshape(1,1,1,18)
            
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            scores.append(outputs)
            labels.append(targets)

            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader))

    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)

    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

    pred_softmax = []
    pred_softmax_threshold = []
    pred_openmax = []

    score_softmax = []
    score_openmax = []

    #从测试结果中拿到每一个测试样本的liner层特征向量
    for score in scores:
        so, ss = openmax(weibull_model, categories, score, 0.5, args.weibull_alpha, "euclidean")
        print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
        #ss  [0.56673586 0.43326417]
        #so  [0.41054204 0.29086916 0.29858881] 
        
        #weibull_threshold=0._ 门槛 大于这个值才确定它属于哪个类
        pred_softmax.append(np.argmax(ss))#[0]
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)#[2]
        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)#[2]
        # print(pred_softmax)
        # print(pred_softmax_threshold)
        # print(pred_openmax)
        
        
        score_softmax.append(ss)
        score_openmax.append(so)
        # print("score_softmax",score_softmax)
        # print("score_openmax",score_openmax)


    print("Evaluation...")

    #softmax
    #实例化
    eval_softmax = Evaluation(pred_softmax, labels, score_softmax)
    torch.save(eval_softmax, os.path.join(args.checkpoint, 'eval_softmax.pkl'))

    #F1得分(F1 Score)：精确度与找召回率的加权平均
    print(f"Softmax accuracy is %.3f" % (eval_softmax.accuracy))
    print(f"Softmax F1 is %.3f" % (eval_softmax.f1_measure))
    print(f"Softmax f1_macro is %.3f" % (eval_softmax.f1_macro))
    print(f"Softmax f1_macro_weighted is %.3f" % (eval_softmax.f1_macro_weighted))
    print(f"Softmax area_under_roc is %.3f" % (eval_softmax.area_under_roc))
    print(f"_________________________________________")

    #softmax_threshold
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)
    torch.save(eval_softmax_threshold, os.path.join(args.checkpoint, 'eval_softmax_threshold.pkl'))    
    print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
    print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
    print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
    print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
    print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
    print(f"_________________________________________")
    
    #openmax
    eval_openmax = Evaluation(pred_openmax, labels, score_openmax)
    torch.save(eval_openmax, os.path.join(args.checkpoint, 'eval_openmax.pkl'))
    print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
    print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
    print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
    print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
    print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
    print(f"_________________________________________")



def save_model(net, acc, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':
    main()










