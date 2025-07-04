import argparse
import copy
from datetime import datetime
import os
import sys

import numpy as np
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as lrscheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_11k import HandsDataset, AttributesDataset, mean, std
from utils.get_loss import get_loss
from utils.lr_scheduler import LRScheduler
from model.models import IGAE_ResNet50, IGAE_DenseNet121, IGAE_ConvNext_t, IGAE_vit_b_16, IGAE_swin_t, IGAE_maxvit_t
from test import calculate_metrics, validate, visualize_grid


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch, best=False):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    if best:
        f = os.path.join(name, 'checkpoint-{:06d}-{:s}.pth'.format(epoch, 'best'))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use. You can set the seed to any fixed value.
    Read more on:
        https://docs.pytorch.org/docs/stable/notes/randomness.html
        https://medium.com/@heyamit10/pytorch-reproducibility-a-practical-guide-d6f573cba679
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True  # Disable CuDNN's non-deterministic optimizations.
    torch.backends.cudnn.benchmark = False  # If True, it causes cuDNN to benchmark multiple convolution algorithms and
    # select the fastest. For PyTorch reproducibility, you need to set to False at cost of slightly lower run-time
    # performance but easy for experimentation.

    # Avoiding nondeterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Set seed to for os.environ, which is a mapping object that represents the user’s OS environmental variables.
    os.environ['PYTHONHASHSEED'] = str(seed) # pythonhashseed is randomly generated when you create a variable in Python

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    parser = argparse.ArgumentParser(description='Training pipeline for IGAE-Net models.')
    parser.add_argument('--attributes_file', type=str, default='./11k/sub_dataset/dorsal_dr.csv',
                        help="Path to the csv file with attributes")
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use: sgd or adam')
    parser.add_argument('--lr', default=0.0008, type=float,
                        help='learning rate for new parameters, try 0.015, 0.02, 0.05, 0.001 (adam default)'
                             'For pretrained parameters, it is 10 times smaller than this')
    parser.add_argument('--warmup', default=True, action='store_true', help='Use warmup learning strategy.')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size')  # 10, 20, 32, etc
    parser.add_argument('--N_epochs', default=50, type=int, help='Number of epochs for training.')  # 50, 60,
    parser.add_argument('--num_workers', default=0, type=int, help='Number of processes to handle dataset loading')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For reproducibility
    if args.is_repr:
        seed = 42  # You can set the seed to any fixed value.
        set_seed(seed)
    else:
        cudnn.benchmark = True  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the
        # fastest. For PyTorch reproducibility, you need to set to False at cost of slightly lower run-time performance
        # but easy for experimentation.

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # Specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        # transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
        #                         shear=None, resample=False, fillcolor=(255, 255, 255)),  # It is not helpful!
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # transforms.RandomErasing()    # It is not helpful!
    ])

    # During validation, we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    sub_dataset = os.path.split(args.attributes_file)[1].split('.')[0]
    train_dataset = HandsDataset(os.path.join('./11k/trainval', sub_dataset, 'train.csv'), attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    val_dataset = HandsDataset(os.path.join('./11k/trainval', sub_dataset, 'val.csv'), attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    model = IGAE_swin_t(n_identity_classes=attributes.num_identities,
                        n_gender_classes=attributes.num_genders,
                        n_age_classes=attributes.num_ages)

    # Send model to GPU; perform multi-GPU training if more than one GPU is available. It is recommended to use
    # DistributedDataParallel instead of DataParallel though.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Data parallelism
    model = model.to(device)

    # Optimizer
    if hasattr(model, 'backbone'):
        base_param_ids = set(map(id, model.backbone.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': 0.1*args.lr},
            {'params': filter(lambda p: p.requires_grad, new_params), 'lr': args.lr}]
    else:
        param_groups = model.parameters()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups,
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4,
                                    nesterov=True
                                    )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=5e-4
        )
    else:
        raise ValueError('Set the optimizer to either sgd or adam')

    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    # print("\nAll identity labels:\n", attributes.identity_labels)
    # print("\nAll gender labels:\n", attributes.gender_labels)
    # print("\nAll age labels:\n", attributes.age_labels)

    print(" ----- Starting training ------- ")

    best_accuracy = 0
    best_epoch = 0
    average_accuracy = 0
    each_at_best_accuracy = []
    best_model_wts = copy.deepcopy(model.state_dict())

    if args.warmup:
        # Learning rate scheduler. Make sure warmup_begin_lr is smaller than base_lr.
        lr_scheduler_warmup = LRScheduler(base_lr=args.lr, step=[25, 40],
                                          factor=0.5, warmup_epoch=10,
                                          warmup_begin_lr=0.000008)
    else:
        # Decay LR by a factor of 0.1 every some (e.g. 30) epochs
        # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        lr_scheduler = lrscheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    for epoch in range(1, args.N_epochs + 1):
        total_loss = 0
        accuracy_identity = 0
        accuracy_gender = 0
        accuracy_age = 0

        if args.warmup:
            lr = lr_scheduler_warmup.update(epoch)  # Warmup strategy is included.
            optimizer.param_groups[0]['lr'] = 0.1*lr  # For pretrained layers
            optimizer.param_groups[1]['lr'] = lr  # For new layers

        for batch in train_dataloader:
            optimizer.zero_grad()  # Zero the parameter gradients

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_identity, batch_accuracy_gender, batch_accuracy_age = \
                calculate_metrics(output, target_labels)

            accuracy_identity += batch_accuracy_identity
            accuracy_gender += batch_accuracy_gender
            accuracy_age += batch_accuracy_age

            loss_train.backward()
            optimizer.step()

        if not args.warmup:
            lr_scheduler.step()  # Use this in case warm-up learning strategy is not used!

        # print('lr: ', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])

        print("epoch {:4d}, loss: {:.4f}, identity: {:.4f}, gender: {:.4f}, age: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_identity / n_train_samples,
            accuracy_gender / n_train_samples,
            accuracy_age / n_train_samples))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            # validate(model, val_dataloader, logger, epoch, device)
            accuracy_val_identity, accuracy_val_gender, accuracy_val_age = \
                validate(model, val_dataloader, logger, epoch, device)

            average_accuracy = (accuracy_val_identity + accuracy_val_gender + accuracy_val_age) / 3  # Average accuracy

            if average_accuracy > best_accuracy:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_accuracy = average_accuracy
                each_at_best_accuracy = [accuracy_val_identity, accuracy_val_gender, accuracy_val_age]

        if epoch % 10 == 0:
            checkpoint_save(model, savedir, epoch)

    if best_epoch >= 5:
        # Print best validation accuracy
        print('Best validation average Accuracy: {:4f}, identity: {:.4f}, gender: {:.4f}, age: {:.4f}'.format(
            best_accuracy, each_at_best_accuracy[0], each_at_best_accuracy[1], each_at_best_accuracy[2]))

        # Load best model weights
        model.load_state_dict(best_model_wts)
        checkpoint_save(model, savedir, best_epoch, True)

    logger.close()

    print('Training is finished!')


# Execute from the interpreter
if __name__ == "__main__":
    main()
