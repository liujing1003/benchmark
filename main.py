from __future__ import print_function
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
from torch.autograd import Variable

# Import models
from models.alexnet import alexnet
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.inception import inception_v3
from models.resnet_v1 import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mpn_cov_resnet_v1 import mpn_cov_resnet18, mpn_cov_resnet34, mpn_cov_resnet50, mpn_cov_resnet101, mpn_cov_resnet152

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='DCNNs Training')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10, cifar100 and cub)')
parser.add_argument('--train_image_size', dest='train_image_size', default=224, type=int,
                    help='image size for training (default: 224)')
parser.add_argument('--test_image_size', dest='test_image_size', default=256, type=int,
                    help='image size for testing (default: 256)')
parser.add_argument('--test_crop_image_size', dest='test_crop_image_size', default=224, type=int,
                    help='image size for testing after cropping (default: 224)')
parser.add_argument('--model', dest='model', default='AlexNet', type=str,
                    help='model type (options: AlexNet')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--alpha', default=32, type=int,
                    help='number of new channel increase per depth (default: 12)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basic block (default: bottleneck)')
parser.add_argument('--epochs', default=90, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual start epoch number (useful on restarts)')
parser.add_argument('--b', '--batchsize', dest='batchsize', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='to use basicblock (default: bottleneck)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--expname', default='PyramidNet', type=str,
                    help='name of experiment')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--gpu_ids', default=0, help='gpu ids: e.g. 0  0,1,2, 0,2.')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(verbose=True)

best_err1 = 100


def main():
    global args, best_err1
    args = parser.parse_args()

    # TensorBoard configure
    if args.tensorboard:
        configure('%s_checkpoints/%s'%(args.dataset, args.expname))

    # CUDA
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
    if torch.cuda.is_available():
        cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        kwargs = {'num_workers': 2, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 2}

    # Data loading code
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2634, 0.2528, 0.2719])
    elif args.dataset == 'cub':
        normalize = transforms.Normalize(mean=[0.4862, 0.4973, 0.4293],
                                         std=[0.2230, 0.2185, 0.2472])
    elif args.dataset == 'webvision':
        normalize = transforms.Normalize(mean=[0.49274242, 0.46481857, 0.41779366],
                                         std=[0.26831809, 0.26145372, 0.27042758])
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))

    # Transforms
    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.train_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.train_image_size),
            transforms.ToTensor(),
            normalize,
        ])
    val_transform = transforms.Compose([
        transforms.Resize(args.test_image_size),
        transforms.CenterCrop(args.test_crop_image_size),
        transforms.ToTensor(),
        normalize
    ])

    # Datasets
    num_classes = 10    # default 10 classes
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data/', train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10('./data/', train=False, download=True, transform=val_transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data/', train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=val_transform)
        num_classes = 100
    elif args.dataset == 'cub':
        train_dataset = datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/DuAngAng/datasets/CUB-200-2011/train/',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/DuAngAng/datasets/CUB-200-2011/test/',
                                           transform=val_transform)
        num_classes = 200
    elif args.dataset == 'webvision':
        train_dataset = datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/WebVision/info/train',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/WebVision/info/val',
                                           transform=val_transform)
        num_classes = 1000
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    # Create model
    if args.model == 'AlexNet':
        model = alexnet(pretrained=False, num_classes=num_classes)
    elif args.model == 'VGG':
        use_batch_normalization = True  # default use Batch Normalization
        if use_batch_normalization:
            if args.depth == 11:
                model = vgg11_bn(pretrained=False, num_classes=num_classes)
            elif args.depth == 13:
                model = vgg13_bn(pretrained=False, num_classes=num_classes)
            elif args.depth == 16:
                model = vgg16_bn(pretrained=False, num_classes=num_classes)
            elif args.depth == 19:
                model = vgg19_bn(pretrained=False, num_classes=num_classes)
            else:
                raise Exception('Unsupport VGG detph: {}, optional depths: 11, 13, 16 or 19'.format(args.depth))
        else:
            if args.depth == 11:
                model = vgg11(pretrained=False, num_classes=num_classes)
            elif args.depth == 13:
                model = vgg13(pretrained=False, num_classes=num_classes)
            elif args.depth == 16:
                model = vgg16(pretrained=False, num_classes=num_classes)
            elif args.depth == 19:
                model = vgg19(pretrained=False, num_classes=num_classes)
            else:
                raise Exception('Unsupport VGG detph: {}, optional depths: 11, 13, 16 or 19'.format(args.depth))
    elif args.model == 'Inception':
        model = inception_v3(pretrained=False, num_classes=num_classes)
    elif args.model == 'ResNet':
        if args.depth == 18:
            model = resnet18(pretrained=False, num_classes=num_classes)
        elif args.depth == 34:
            model = resnet34(pretrained=False, num_classes=num_classes)
        elif args.depth == 50:
            model = resnet50(pretrained=False, num_classes=num_classes)
        elif args.depth == 101:
            model = resnet101(pretrained=False, num_classes=num_classes)
        elif args.depth == 152:
            model = resnet152(pretrained=False, num_classes=num_classes)
        else:
            raise Exception('Unsupport ResNet detph: {}, optional depths: 18, 34, 50, 101 or 152'.format(args.depth))
    elif args.model == 'MPN-COV-ResNet':
        if args.depth == 18:
            model = mpn_cov_resnet18(pretrained=False, num_classes=num_classes)
        elif args.depth == 34:
            model = mpn_cov_resnet34(pretrained=False, num_classes=num_classes)
        elif args.depth == 50:
            model = mpn_cov_resnet50(pretrained=False, num_classes=num_classes)
        elif args.depth == 101:
            model = mpn_cov_resnet101(pretrained=False, num_classes=num_classes)
        elif args.depth == 152:
            model = mpn_cov_resnet152(pretrained=False, num_classes=num_classes)
        else:
            raise Exception('Unsupport MPN-COV-ResNet detph: {}, optional depths: 18, 34, 50, 101 or 152'.format(args.depth))
    else:
        raise Exception('Unsupport model'.format(args.model))

    # Get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model = model.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err1 = checkpoint['best_err1']
            model.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    print(model)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        err1 = validate(val_loader, model, criterion, epoch)

        # Remember best err1 and save checkpoint
        is_best = (err1 <= best_err1)
        best_err1 = min(err1, best_err1)
        print("Current best accuracy (error):", best_err1)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
        }, is_best)

    print("Best accuracy (error):", best_err1)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Train for one epoch on the training set
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs_var = to_var(inputs)
        targets_var = to_var(targets)

        # Compute output
        outputs = model(inputs_var)
        # For Inception-v3, the output may be a tuple
        if type(outputs) is tuple:
            outputs = outputs[0]
        loss = criterion(outputs, targets_var)

        # Measure accuracy and record loss
        err1, err5 = accuracy(outputs.data.cpu(), targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(err1[0], inputs.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top 1-err {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch+1, i, len(train_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

    # Log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_error', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """
    Perform validation on the validation set
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs_var = to_var(inputs)
        targets_var = to_var(targets)

        # Compute output
        outputs = model(inputs_var)
        loss = criterion(outputs, targets_var)

        # Measure accuracy and record loss
        err1, err5 = accuracy(outputs.data.cpu(), targets, topk=(1,5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(err1[0], inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top 1-err {top1.val:.3f} ({top1.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

    print('Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}'
          .format(epoch+1, args.epochs, top1=top1))
    # Log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)

    return top1.avg


def to_var(x):
    """
    Convert tensor x to autograd Variable
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save checkpoint to disk
    """
    directory = "%s_checkpoints/%s/" % (args.dataset, args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "%s_checkpoints/%s/" % (args.dataset, args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """
    Compute and store the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def adjust_learning_rate(optimizer, epoch):
    """
    Adjust the learning rate
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr
    
    # Log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    Computes the error@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0/batch_size))

    return res

if __name__ == '__main__':
    main()
