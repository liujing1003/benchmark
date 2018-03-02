import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet as RN
import Pyramid_SE_2 as PYRM

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

# used for compute confusion matrix
from torchnet import meter
# used for plot confusion matrix
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', '--batchsize', dest='batchsize', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--alpha', default=48, type=int,
                    help='number of new channel increases per depth (default: 12)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock (default: bottleneck)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--expname', default='PyramidNet', type=str,
                    help='name of experiment')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--dataset', dest='dataset', default='plankton', type=str,
                    help='dataset (options: cifar10, cifar100, WHOI and plankton)')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(verbose=True)

best_err1 = 100
numberofclass = 103

def main():
    global args, best_err1, numberofclass
    args = parser.parse_args()
    if args.tensorboard: configure("runs_WHOI/%s"%(args.expname))
    
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3]],
                                     # std=[x/255.0 for x in [63.0]])
    
    transform_test = transforms.Compose([
        transforms.Scale(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.dataset == 'cifar100':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, transform=transform_test),
            batch_size=args.batchsize, shuffle=True, **kwargs)   
        numberofclass = 100     
    elif args.dataset == 'cifar10':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.batchsize, shuffle=True, **kwargs)
        numberofclass = 10
    elif args.dataset == 'WHOI':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/2014', transform=transform_test),
            batch_size=args.batchsize, shuffle=True, **kwargs)
	numberofclass = 103
    elif args.dataset == 'plankton':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/DuAngAng/oceans-2018/codes/plankton-set/test', transform=transform_test),
            batch_size=args.batchsize, shuffle=True, **kwargs)
	numberofclass = 121
    else: 
        raise Exception ('unknown dataset: {}'.format(args.dataset)) 

    print('Training PyramidNet-{} on {} dataset:'.format(args.depth, args.dataset.upper()))
    
    # create model
    # model = RN.ResNet(args.depth, numberofclass, bottleneck=args.bottleneck) # for ResNet
    model = PYRM.PyramidNet(args.depth, args.alpha, numberofclass, bottleneck=args.bottleneck) # for PyramidNet

    # get the number of model parameters
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, the best err1)"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print(model)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # evaluate on validation set
    err1, cm = validate(val_loader, model, criterion, numberofclass)

    # plot confusion matrix and save the fig
    plot_and_save_confusion_matrix(cm, numberofclass, normalize=True)

    # compute average precision, recall and F1 score
    average_precision, average_recall, average_f1_score = compute_precision_recall_f1_score(cm)

    print("Accuracy (error): {}".format(err1))
    print("Precision: {}".format(average_precision))
    print("Recall: {}".format(average_recall))
    print("F1-score: {}".format(average_f1_score))

def validate(val_loader, model, criterion, numberofclass):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    confusion_matrix = meter.ConfusionMeter(numberofclass)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        confusion_matrix.add(output.data.squeeze(), target.type(torch.LongTensor))

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1,5))
        
        losses.update(loss.data[0], input.size(0))
        top1.update(err1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top 1-err {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    cm_value = confusion_matrix.value()

    print('Final result:\n')
    print('Top 1-err {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, cm_value

def plot_and_save_confusion_matrix(cm, number_of_class,
                                   normalize=False,
                                   title='Confusion matrix',
                                   cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float')
        non_zero_indices = (cm.sum(axis=1) > 0)
        cm[non_zero_indices] = cm[non_zero_indices] / cm[non_zero_indices].sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0, number_of_class, 10)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)

    # thresh = cm.max() / 2.

    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, '',
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    plt.savefig("runs_WHOI/%s/"%(args.expname) + "confusion_matrix.png")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the errision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))
    return res

def compute_precision_recall_f1_score(confusion_matrix):
    """Compute the average precision, recall and F1-score"""
    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0
    for i in range(confusion_matrix.shape[0]):
        true_pos = confusion_matrix[i, i]
        true_and_false_pos = np.sum(confusion_matrix[i, :])
        true_pos_and_false_neg = np.sum(confusion_matrix[:, [i]])

        if true_pos == 0:
            if true_and_false_pos == 0 and true_pos_and_false_neg == 0:
                precision = 1
                recall = 1
                f1_score = 1
            else:
                if true_pos_and_false_neg == 0:
                    precision = 1
                    recall = 0
                elif true_and_false_pos == 0:
                    precision = 0
                    recall = 1
                else:
                    precision = 0
                    recall = 0
                    
                f1_score = 0
        else:
            precision = float(true_pos) / true_and_false_pos
            recall = float(true_pos) / true_pos_and_false_neg
            f1_score = float(2 * precision * recall) / (precision + recall)

        sum_precision += precision
        sum_recall += recall
        sum_f1_score += f1_score

    average_precision = float(sum_precision) / confusion_matrix.shape[0]
    average_recall = float(sum_recall) / confusion_matrix.shape[0]
    average_f1_score = float(sum_f1_score) / confusion_matrix.shape[0]

    return average_precision, average_recall, average_f1_score


if __name__ == '__main__':
    main()
