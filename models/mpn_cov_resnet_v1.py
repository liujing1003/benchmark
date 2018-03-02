import torch
import torch.nn as nn
import math
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Compute Covariance Matrix
def covariance(A):
    batchSize = A.size(0)
    dim = A.size(1)
    N = A.size(2)  # N features
    P = Variable(torch.FloatTensor(batchSize, dim, dim)).cuda()

    for i in range(0, batchSize):
        I = Variable(torch.eye(N)).cuda()
        ones_vec = Variable(torch.ones(N, 1)).cuda()
        _I = 1.0 / N * (I - 1.0 / N * torch.matmul(ones_vec, torch.t(ones_vec)))
        P[i] = torch.matmul(torch.matmul(A[i], _I), torch.t(A[i]))

    return P


# Get the upper triangular mask of matirx
def triu_mask(value):
    n = value.size(-1)
    coords = value.data.new(n)
    torch.arange(float(0), float(n), out=coords)
    return coords >= coords.view(n, 1)


# Get the upper triangular matrix of a given matrix
def upper_triangular(A):
    batchSize = A.size(0)
    dim = A.size(1)
    N = A.size(2)   # N features
    U = Variable(torch.FloatTensor(batchSize, int(dim*(dim+1)/2))).cuda()
    for i in range(0, batchSize):
        U[i] = A[i][triu_mask(A[i])]

    return U


# Compute error
def compute_error(A, sA):
    normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1), dim=1))
    error = A - torch.bmm(sA, sA)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
    return torch.mean(error)


def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    error = compute_error(A, sA)
    return sA, error


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MpnCovResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MpnCovResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # For MPN-COV Pooling layer
        self.cov_in_dim = 256
        self.cov_out_dim = int(self.cov_in_dim * (self.cov_in_dim + 1) / 2.0)
        self.reduce_conv = nn.Conv2d(512 * block.expansion, self.cov_in_dim, kernel_size=1, stride=1)
        self.reduce_conv_bn = nn.BatchNorm2d(self.cov_in_dim)
        self.reduce_conv_relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(self.cov_out_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # For MPN-COV Pooling layer
        x = self.reduce_conv(x)
        x = self.reduce_conv_bn(x)
        x = self.reduce_conv_relu(x)

        x = x.view(x.size(0), self.cov_in_dim, -1)  # reshape
        x = covariance(x)
        x, _ = sqrt_newton_schulz_autograd(x, numIters=10, dtype=torch.FloatTensor)
        x = upper_triangular(x)
        x = self.fc(x)

        return x


def mpn_cov_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MpnCovResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def mpn_cov_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MpnCovResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def mpn_cov_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MpnCovResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def mpn_cov_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MpnCovResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def mpn_cov_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MpnCovResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
