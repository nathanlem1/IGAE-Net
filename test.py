import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset_11k import HandsDataset, AttributesDataset, mean, std
from model.models import IGAE_ResNet50, IGAE_DenseNet121, IGAE_ConvNext_t, IGAE_vit_b_16, IGAE_swin_t, IGAE_maxvit_t
from get_loss import get_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_identity = 0
        accuracy_gender = 0
        accuracy_age = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_identity, batch_accuracy_gender, batch_accuracy_age = \
                calculate_metrics(output, target_labels)

            accuracy_identity += batch_accuracy_identity
            accuracy_gender += batch_accuracy_gender
            accuracy_age += batch_accuracy_age

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_identity /= n_samples
    accuracy_gender /= n_samples
    accuracy_age /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, identity: {:.4f}, gender: {:.4f}, age: {:.4f}\n".format(
        avg_loss, accuracy_identity, accuracy_gender, accuracy_age))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_identity', accuracy_identity, iteration)
    logger.add_scalar('val_accuracy_gender', accuracy_gender, iteration)
    logger.add_scalar('val_accuracy_age', accuracy_age, iteration)

    model.train()

    return accuracy_identity, accuracy_gender, accuracy_age


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_identity_all = []
    gt_gender_all = []
    gt_age_all = []
    predicted_identity_all = []
    predicted_gender_all = []
    predicted_age_all = []

    accuracy_identity = 0
    accuracy_gender = 0
    accuracy_age = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_identities = batch['labels']['identity_labels']
            gt_genders = batch['labels']['gender_labels']
            gt_ages = batch['labels']['age_labels']
            output = model(img.to(device))

            batch_accuracy_identity, batch_accuracy_gender, batch_accuracy_age = \
                calculate_metrics(output, batch['labels'])
            accuracy_identity += batch_accuracy_identity
            accuracy_gender += batch_accuracy_gender
            accuracy_age += batch_accuracy_age

            # get the most confident prediction for each image
            _, predicted_identities = output['identity'].cpu().max(1)
            _, predicted_genders = output['gender'].cpu().max(1)
            _, predicted_ages = output['age'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_identity = attributes.identity_id_to_name[predicted_identities[i].item()]
                predicted_gender = attributes.gender_id_to_name[predicted_genders[i].item()]
                predicted_age = attributes.age_id_to_name[predicted_ages[i].item()]

                gt_identity = attributes.identity_id_to_name[gt_identities[i].item()]
                gt_gender = attributes.gender_id_to_name[gt_genders[i].item()]
                gt_age = attributes.age_id_to_name[gt_ages[i].item()]

                gt_identity_all.append(gt_identity)
                gt_gender_all.append(gt_gender)
                gt_age_all.append(gt_age)

                predicted_identity_all.append(predicted_identity)
                predicted_gender_all.append(predicted_gender)
                predicted_age_all.append(predicted_age)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_identity, predicted_gender, predicted_age))
                gt_labels.append("{}\n{}\n{}".format(gt_identity, gt_gender, gt_age))

    # Show classification accuracy and save it
    n_samples = len(dataloader)
    print("\nAccuracy:\nidentity: {:.4f}, gender: {:.4f}, age: {:.4f}".format(
        accuracy_identity / n_samples,
        accuracy_gender / n_samples,
        accuracy_age / n_samples))
    result = os.path.join('/'.join(checkpoint.split('/')[:-1]), checkpoint.split('/')[-1][:-4] + '_accuracy.txt')
    res = open(result, 'w')
    res.write('identity:%.4f gender:%.4f age:%.4f' % (accuracy_identity / n_samples, accuracy_gender / n_samples,
                                                      accuracy_age / n_samples))

    # Show classification report
    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in classification report
        warnings.simplefilter("ignore")
        print('\nIdentity classification report: \n', classification_report(gt_identity_all, predicted_identity_all))
        print('Gender classification report: \n', classification_report(gt_gender_all, predicted_gender_all))
        print('Age classification report: \n', classification_report(gt_age_all, predicted_age_all))

    # Draw confusion matrices
    if show_cn_matrices:
        # Identity
        cn_matrix = confusion_matrix(
            y_true=gt_identity_all,
            y_pred=predicted_identity_all,
            labels=attributes.identity_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.identity_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Identity")
        plt.tight_layout()
        plt.savefig(os.path.join('/'.join(checkpoint.split('/')[:-1]), 'Identity_cm.png'))
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_gender_all,
            y_pred=predicted_gender_all,
            labels=attributes.gender_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.gender_labels).plot(
            xticks_rotation='horizontal')
        plt.title("Gender")
        plt.tight_layout()
        plt.savefig(os.path.join('/'.join(checkpoint.split('/')[:-1]), 'Gender_cm.png'))
        plt.show()

        # Uncomment code below to see the age confusion matrix (it may be too big to display)
        cn_matrix = confusion_matrix(
            y_true=gt_age_all,
            y_pred=predicted_age_all,
            labels=attributes.age_labels,
            normalize='true')
        # plt.rcParams.update({'font.size': 1.8})
        # plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(cn_matrix, attributes.age_labels).plot(
            include_values=True, xticks_rotation='vertical')
        # plt.rcParams.update({'figure.dpi': 100})
        # plt.rcParams.update({'font.size': 5})
        plt.title("Age")
        plt.savefig(os.path.join('/'.join(checkpoint.split('/')[:-1]), 'Age_cm.png'))
        plt.show()

    if show_images:
        title = "Ground truth labels vs Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, gt_label, pr_label in zip(imgs, axs, gt_labels, labels):
            ax.set_xlabel('GT: ' + gt_label + "\n " + 'PR: ' + pr_label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join('/'.join(checkpoint.split('/')[:-1]), 'sample_result.png'))
        plt.show()

    model.train()


def calculate_metrics(output, target):
    _, predicted_identity = output['identity'].cpu().max(1)
    gt_identity = target['identity_labels'].cpu()

    _, predicted_gender = output['gender'].cpu().max(1)
    gt_gender = target['gender_labels'].cpu()

    _, predicted_age = output['age'].cpu().max(1)
    gt_age = target['age_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_identity = balanced_accuracy_score(y_true=gt_identity.numpy(), y_pred=predicted_identity.numpy())
        accuracy_gender = balanced_accuracy_score(y_true=gt_gender.numpy(), y_pred=predicted_gender.numpy())
        accuracy_age = balanced_accuracy_score(y_true=gt_age.numpy(), y_pred=predicted_age.numpy())

    return accuracy_identity, accuracy_gender, accuracy_age


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--attributes_file', type=str, default='./11k/sub_dataset/dorsal_dr.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/2023-06-21_14-21/checkpoint-000030-best.pth',
                        help="Path to the checkpoint")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attributes = AttributesDataset(args.attributes_file)  # attributes variable contains labels for the categories in
    # the dataset and mapping between string names and IDs.

    # During validation, we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    sub_dataset = os.path.split(args.attributes_file)[1].split('.')[0]
    test_dataset = HandsDataset(os.path.join('./11k/trainval', sub_dataset, 'val.csv'), attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=8)

    model = IGAE_swin_t(n_identity_classes=attributes.num_identities, n_gender_classes=attributes.num_genders,
                        n_age_classes=attributes.num_ages)

    # Send model to GPU; perform multi-GPU training if more than one GPU is available. It is recommended to use
    # DistributedDataParallel instead of DataParallel though.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Data parallelism
    model = model.to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)

    print('Testing is finished!')
