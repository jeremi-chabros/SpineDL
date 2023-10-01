# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import torchmetrics

from data_loader import ScanDataset
from model import NIMA
from gradcam.utils import visualize_cam
from gradcam import GradCAM


def overlay_images(original, cam_output, alpha=0.5):
    return (alpha * original) + ((1 - alpha) * cam_output)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.ToTensor()])

    base_model = models.resnet50(weights="DEFAULT", progress=False)
    model = NIMA(base_model)

    # Load pretrained model if required
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(
            config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' %
              config.warm_start_epoch)

    model = model.to(device)

    if config.test:
        print('Testing')

        # Load model
        model_path = os.path.join(
            config.ckpt_path, f'epoch-{config.warm_start_epoch}.pkl')
        model.load_state_dict(torch.load(model_path))

        # Set target layer
        target_layer = model.features

        # Prepare test dataset and loader
        testset = ScanDataset(csv_file=config.test_csv_file,
                              root_dir=config.test_img_path,
                              transform=val_transform)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        model.eval()
        gradcam = GradCAM(model, target_layer)

        ypreds = []
        ylabels = []
        im_ids = []
        count = 0

        for data in test_loader:
            im_id = data['img_id']
            im_name = os.path.split(im_id[0])
            base_name = os.path.splitext(im_name[1])
            image = data['image'].to(device)
            labels = data['annotations'].to(device).long()

            output = model(image)
            output = output.view(-1, 2)
            bpred = output.to(torch.device("cpu"))
            cpred = bpred.data.numpy()
            blabel = labels.to(torch.device("cpu"))
            clabel = blabel.data.numpy()

            # pred_rd = cpred[0]
            # label_rd = clabel[0]
            
            prediction = [cpred[0][0], cpred[0][1]]
            labl = [clabel[0][0][0],clabel[0][1][0]]

            pred_dd = np.argmax(prediction, axis=0)
            label_dd = np.argmax(labl, axis=0)

            if label_dd == pred_dd:
                addon = "TP" if label_dd == 1 else "TN"
            else:
                addon = "FP" if label_dd == 1 else "FN"

            # GradCAM visualization
            # mask, _ = gradcam(image, success)
            # heatmap, cam_result = visualize_cam(mask, image)
            # overlaid_image = overlay_images(image.squeeze(0).cpu(), cam_result.squeeze(0).cpu())
            # im = transforms.ToPILImage()(overlaid_image)

            # UNCOMMENT
            if config.generate_heatmaps:
                mask, _ = gradcam(image)
                heatmap, result = visualize_cam(mask, image)
                im = transforms.ToPILImage()(result)
                im.save(f"/Users/jjc/Research/CNOC_Spine/All/heatmaps/{base_name[0]}_{addon}.jpg")

            ypreds.append(cpred)
            ylabels.append(clabel)
            im_name = os.path.split(im_id[0])
            im_ids.append(im_name[1])
            count = count+1

    # Convert predictions and labels to tensors
    apreds = torch.Tensor(ypreds).squeeze()
    alabels = torch.Tensor(ylabels).squeeze()

    # Set values below the threshold to a small number, so they won't be chosen by argmax
    # threshold = 0.23
    threshold=0.25
    apreds[apreds[:, 1] >= threshold, 1] = float('inf')
    apreds[apreds[:, 1] < threshold, 0] = float('inf')
    pred_labels = torch.argmax(apreds, dim=1)
    true_labels = torch.argmax(alabels, dim=1)  # For true labels, you probably don't need thresholding.

    # pred_labels = torch.argmax(apreds, dim=1)
    # true_labels = torch.argmax(alabels, dim=1)

    TP = ((pred_labels == 1) & (true_labels == 1)).float().sum().item()
    TN = ((pred_labels == 0) & (true_labels == 0)).float().sum().item()
    FP = ((pred_labels == 1) & (true_labels == 0)).float().sum().item()
    FN = ((pred_labels == 0) & (true_labels == 1)).float().sum().item()

    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    tasktype = "binary"
    acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
    spec = torchmetrics.classification.Specificity(task=tasktype)
    f1 = torchmetrics.classification.F1Score(task=tasktype)
    bprecision = torchmetrics.classification.Precision(task=tasktype)
    brec = torchmetrics.classification.Recall(task=tasktype)
    auroc = torchmetrics.classification.AUROC(task=tasktype)
    broc = torchmetrics.classification.ROC(task=tasktype)
    cm = torchmetrics.ConfusionMatrix(task=tasktype)
    prcurve = torchmetrics.classification.BinaryPrecisionRecallCurve()


    # Compute metrics
    print(f'Accuracy: {acc(pred_labels, true_labels):.4f}')
    print(f'Specificity: {spec(pred_labels, true_labels):.4f}')
    print(f'F1 Score: {f1(pred_labels, true_labels):.4f}')
    print(f'Precision: {bprecision(pred_labels, true_labels):.4f}')
    print(f'Recall: {brec(pred_labels, true_labels):.4f}')
    print(f'AUC-ROC: {auroc(apreds, alabels):.4f}')

    # Plot ROC
    broc.update(apreds, alabels.long())
    fig, ax = broc.plot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    fig.savefig("/Users/jjc/Research/CNOC_Spine/All/ROC.png",
                dpi=300, bbox_inches='tight')

    # Plot Confusion Matrix
    cm.update(pred_labels, true_labels)
    fig, ax = cm.plot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    fig.savefig("/Users/jjc/Research/CNOC_Spine/All/confusion_matrix.png",
                dpi=300, bbox_inches='tight')
    
    prcurve.update(apreds, alabels.long())
    fig, ax = prcurve.plot(score=True)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    fig.savefig("/Users/jjc/Research/CNOC_Spine/All/PR_curve.png",
                dpi=300, bbox_inches='tight')

    # Save results
    np.savez('test_results.npz', Label=ylabels, Predict=ypreds)
    df = pd.DataFrame(data={'Label': ylabels, "Predict": ypreds})
    print(df.dtypes)
    df.to_pickle("./test_results.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train_img_path', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/imgs/train26weeks')
    parser.add_argument('--val_img_path', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/imgs/val26weeks')
    parser.add_argument('--test_img_path', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/imgs/test26weeks')
    parser.add_argument('--train_csv_file', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/imgs/train26weeks.csv')
    parser.add_argument('--val_csv_file', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/imgs/val26weeks.csv')
    parser.add_argument('--test_csv_file', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/imgs/test26weeks.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=.001)
    parser.add_argument('--dense_lr', type=float, default=.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/ckpts/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=216)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=True)
    parser.add_argument('--generate_heatmaps', type=bool, default=False)

    config, unknown = parser.parse_known_args()
    main(config)
