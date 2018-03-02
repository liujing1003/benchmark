import torch
from torchvision import datasets, transforms

def get_mean_and_std(dataset, channel=3, batch_size=1, num_workers=2):
    """
    Compute the mean and std of dataset.
    :param dataset: a torchvision dataset
    :param channel: the channels of the dataset images, default 3
    :return: mean, std
    """

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    mean = torch.zeros(channel)
    std = torch.zeros(channel)
    print("Computing mean and std...")
    for inputs, targets in dataloader:
        for i in range(channel):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset)//batch_size)
    std.div_(len(dataset)//batch_size)
    return mean, std

# Replace the dataset with your own dataset
# dataset = datasets.CIFAR100('/Users/duang/Documents/research/incubator/imagenet/data/',
#                    train=True, download=True,
#                    transform=transforms.Compose([transforms.ToTensor()]))

dataset = datasets.ImageFolder('/media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/WebVision/info/train',
                               transform=transforms.Compose([
                                   transforms.RandomResizedCrop(224),
                                   transforms.ToTensor()
                               ]))

mean, std = get_mean_and_std(dataset, channel=3, batch_size=16, num_workers=2)
print("mean = {}\nstd = {}".format(mean.numpy(), std.numpy()))
