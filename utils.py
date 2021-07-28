import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

import sklearn.metrics

import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import PIL.Image as pil_image
from datasets import TrainDataset, EvalDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()


class AverageMeter(object):
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

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)

    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)



    return train_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):
    eval_dataset = EvalDataset("BLAH_BLAH/Set5_x4.h5")
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    model.eval()
    epoch_psnr = AverageMeter()



    for data in eval_dataloader:
        inputs,labels=data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs)

        preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
        labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

        preds = preds[4:-4, 4:-4]
        labels = labels[4:-4, 4:-4]

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))


    return epoch_psnr.avg

def img_cal_psnr(model,image_file,device):
    image = pil_image.open(image_file).convert('RGB')

    image_width = (image.width // 4) * 4
    image_height = (image.height // 4) * 4

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // 4, hr.height // 4), resample=pil_image.BICUBIC)

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr = torch.from_numpy(lr).to(device)
    hr = torch.from_numpy(hr).to(device)

    with torch.no_grad():
        preds = model(lr).squeeze(0)

    preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

    preds_y = preds_y[4:-4, 4:-4]
    hr_y = hr_y[4:-4, 4:-4]

    psnr = calc_psnr(hr_y, preds_y)
    return psnr

def evaluate_model_benchmark(model, device):

    model.eval()
    model.to(device)
    psnr_average = 0
    for i in range(1, 6):
        image_file = "data/Set5/x4/img_00{}_SRF_4_HR.png".format(i)
        psnr_average=psnr_average+img_cal_psnr(model,image_file,device)
    Set5_psnr_average = psnr_average/5
    psnr_average = 0
    for i in range(1, 15):
        if i <= 9:
            image_file = "data/Set14/x4/img_00{}_SRF_4_HR.png".format(i)
        elif i > 9 and i <= 14:
            image_file = "data/Set14/x4/img_0{}_SRF_4_HR.png".format(i)
        psnr_average=psnr_average+img_cal_psnr(model,image_file,device)
    Set14_psnr_average = psnr_average/14
    psnr_average = 0
    for i in range(1, 101):
        if i <= 9:
            image_file = "data/BSD100/x4/img_00{}_SRF_4_HR.png".format(i)
        elif i > 9 and i <= 99:
            image_file = "data/BSD100/x4/img_0{}_SRF_4_HR.png".format(i)
        elif i == 100:
            image_file = "data/BSD100/x4/img_100_SRF_4_HR.png"
        psnr_average = psnr_average + img_cal_psnr(model,image_file,device)
    BSD100_psnr_average = psnr_average / 100
    psnr_average = 0
    for i in range(1, 101):
        if i <= 9:
            image_file = "data/Urban100/x4/img_00{}_SRF_4_HR.png".format(i)
        elif i > 9 and i <= 99:
            image_file = "data/Urban100/x4/img_0{}_SRF_4_HR.png".format(i)
        elif i == 100:
            image_file = "data/Urban100/x4/img_100_SRF_4_HR.png"
        psnr_average = psnr_average + img_cal_psnr(model,image_file,device)
    Urban100_psnr_average = psnr_average / 100
    return Set5_psnr_average, Set14_psnr_average, BSD100_psnr_average, Urban100_psnr_average


def create_classification_report(model, device, test_loader):

    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            y_true += data[1].numpy().tolist()
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().numpy().tolist()

    classification_report = sklearn.metrics.classification_report(
        y_true=y_true, y_pred=y_pred)

    return classification_report


def train_model(model,
                train_dataloader,
                train_dataset,
                eval_dataloader,
                device,
                l1_regularization_strength=0,
                l2_regularization_strength=1e-4,
                learning_rate=1e-4,
                num_epochs=800):

    # The training configurations were not carefully selected.

    criterion = nn.L1Loss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    # Learning Rate decay ratio

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_psnr = evaluate_model(model=model,test_loader=eval_dataloader,device=device,criterion=criterion)
    print("Epoch: {:03d} Eval Acc: {:.2f}".format(0,eval_psnr))

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * (0.1 ** (epoch // int(800 * 0.8)))
        # Training
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % 16)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, 800 - 1))
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                preds = model(inputs)

                loss = criterion(preds, labels)
                ##########pruning
                l1_reg = torch.tensor(0.).to(device)
                for module in model.modules():
                    mask = None
                    weight = None
                    for name, buffer in module.named_buffers():
                        if name == "weight_mask":
                            mask = buffer
                    for name, param in module.named_parameters():
                        if name == "weight_orig":
                            weight = param
                    # We usually only want to introduce sparsity to weights and prune weights.
                    # Do the same for bias if necessary.
                    if mask is not None and weight is not None:
                        l1_reg += torch.norm(mask * weight, 1)

                loss += l1_regularization_strength * l1_reg
                epoch_losses.update(loss.item(), len(inputs))
                ##########pruning
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # statistics
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))



            # Evaluation

        eval_psnr = evaluate_model(model=model, test_loader=eval_dataloader, device=device, criterion=criterion)
        print('eval psnr: {:.2f}'.format(eval_psnr))



    return model


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def create_model(num_classes=10, model_func=torchvision.models.resnet18):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = model_func(num_classes=num_classes, pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model
