import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import confusion_matrix

from datasets.generic import SimpleImageDataset
from models.tsn_resnet_model_singular import Network
from utils import misc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--static", action="store_true")
# parser.add_argument("--root_path", type=str, default=R"G:\Datasets\CTBCYogie\cropped_frames")
# parser.add_argument("--root_path", type=str, default=R"G:\Datasets\CTBCYogie\cropped_frames_dynamic")
parser.add_argument("--root_path", type=str, default=R"G:\Datasets\CTBCYogie2\cropped_frames_dynamic")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--log_path", type=str, default="logs")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--comment", type=str, default="")


def setup_logger(name, filename, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(filename)
    # handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def main(args):
    # checkpoint_name = "tsn_sparse_{}_ctbc_16seg_split{:d}_bs16_wd0.001".format(args.arch, args.split)
    checkpoint_name = "tsn_sparse_{}_ctbc2_16seg_split{:d}_bs16_wd0.001_{}_allcam".format(args.arch, args.split, "static" if args.static else "dynamic")
    save_name = checkpoint_name
    if args.comment != '':
        save_name += "_{}".format(args.comment)
    
    args.checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_name)
    # args.log_path = os.path.join(args.log_path, save_name)
    args.result_path = os.path.join(args.result_path, save_name)
    os.makedirs(args.result_path, exist_ok=True)
    default_log = setup_logger('default_log', filename=os.path.join(args.result_path, 'results.log'))
    outputs_log = setup_logger('outputs_log', filename=os.path.join(args.result_path, 'outputs.csv'))

    net = Network(backbone=args.arch, n_class=7).to(device)

    try:
        checkpoint = torch.load("%s/checkpoint_best.pth" % args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded best checkpoint from epoch %s with accuracy %s' % (checkpoint['epoch'] + 1, checkpoint['accuracy']))
        visualize(net, args, default_log, outputs_log)
    except FileNotFoundError:
        print('Best checkpoint not found')


@torch.no_grad()
def visualize(model, args, default_log, outputs_log):
    test_tfs = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inverse_norm = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

    class_dict_decode, class_dict_encode = get_class_dict(args)

    model.eval()

    subject_dict = {
        1: [2, 10],
        2: [5, 9],
        3: [1, 4],
        4: [7, 8],
        5: [3, 6],
    }

    for emotion in os.listdir(args.root_path):
        path_with_emotion = os.path.join(args.root_path, emotion)
        labels_idx = class_dict_encode[emotion]

        for clip_name in os.listdir(path_with_emotion):
            path_with_clip = os.path.join(path_with_emotion, clip_name)

            subject, cam, clip_no = clip_name.split('_')
            cam = int(cam.replace('cam', ''))

            if int(subject) not in subject_dict[args.split]:
                continue

            dataset = SimpleImageDataset(path_with_clip, transform=test_tfs)
            loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=50, num_workers=2)

            top1 = misc.AverageMeter()
            top5 = misc.AverageMeter()

            clip_outputs = []

            for i, inputs in enumerate(loader):
                inputs = inputs.to(device)

                n_batch = inputs.size(0)
                labels = torch.LongTensor([labels_idx]).repeat(n_batch).to(device)

                # compute output
                outputs = model(inputs, False)
                clip_outputs.append(outputs.argmax(1).cpu())
                
                original_inputs = torch.stack([inverse_norm(row) for row in inputs], dim=0).cpu()
                original_inputs = F.interpolate(original_inputs, (256, 256))

                results = original_inputs * 0.7
                results = put_text(results, outputs, labels, class_dict_decode)

                torchvision.utils.save_image(results, os.path.join(args.result_path, "%s_%s_%04d.jpg" % (emotion, clip_name, i)), normalize=False, nrow=10)

                # measure accuracy and record loss
                prec1, prec5 = misc.accuracy(outputs, labels, topk=(1, 5))
                top1.update(prec1[0], n_batch)
                top5.update(prec5[0], n_batch)

            clip_msg = '{emotion} {clip_name} | Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(emotion=emotion, clip_name=clip_name, top1=top1, top5=top5)
            print(clip_msg)
            default_log.info(clip_msg)

            clip_outputs = torch.cat(clip_outputs, dim=0).numpy()
            outputs_log.info("{clip_name},{subject},{cam},{clip_no},{labels_idx},{emotion},{outputs}".format(
                clip_name=clip_name, subject=subject, cam=cam, clip_no=clip_no, labels_idx=labels_idx, emotion=emotion,
                outputs=",".join(map(str, clip_outputs))
            ))


def put_text(imgs, logits, targets, emotion_dict):
    """
    Args:
        imgs: tensor of float images in [0, 1] value with shape (N, C, H, W).
        logits: tensor of logits with shape (N, n_class)
    
    Output:
        Tensor of float images in [0, 1] value with shape (N, C, H, W).
    """
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255)
    color_correct = (159, 253, 50)

    # imgs = normalize_to_img(imgs).byte()
    imgs = (255 * imgs).byte()

    topk = len(emotion_dict)
    confidences, class_idxs = torch.softmax(logits, 1).topk(topk, dim=1, largest=True, sorted=True)

    results = []
    for i in range(imgs.shape[0]):
        result = np.zeros(imgs.shape[-2:] + (3,), dtype=np.uint8)
        result[:, :, :] = imgs[i, :, :, :].permute(1, 2, 0)

        class_idx = targets[i].item()
        text = emotion_dict[class_idx]
        text_position = (0, 10)
        cv2.putText(result, "gt: {}".format(text), text_position, font_face, font_scale, color, thickness)

        for j in range(topk):
            confidence = confidences[i, j].item()
            class_idx = class_idxs[i, j].item()
            text = emotion_dict[class_idx]
            text_position = (0, 30 + j * 10)
            cv2.putText(result, "{}: {} ({:.3f})".format(j+1, text, confidence), text_position, font_face, font_scale, color_correct if class_idx == targets[i].item() else color, thickness)

        result = torch.from_numpy(np.array(result)).permute(2, 0, 1)
        result = result.float() / 255
        results.append(result)
    results = torch.stack(results, dim=0)
    
    return results


def save_model(model, optimizer, epoch, epoch_accuracy, best_accuracy, checkpoint_path):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': epoch_accuracy,
        'best_accuracy': best_accuracy,
        'epoch': epoch,
    }

    torch.save(checkpoint, '%s/checkpoint_last.pth' % (checkpoint_path))

    if epoch_accuracy >= best_accuracy:
        torch.save(checkpoint, '%s/checkpoint_best.pth' % (checkpoint_path))


def get_class_dict(args):
    class_dict_decode = {}
    class_dict_encode = {}
    
    class_desc_file = "datasets/lists/ctbc/ctbc_classInd.txt"
    class_info = pd.read_csv(class_desc_file, sep=' ', header=None)
    for _, row in class_info.iterrows():
        class_idx, class_name = row
        class_dict_decode[class_idx] = class_name
        class_dict_encode[class_name] = class_idx

    return class_dict_decode, class_dict_encode

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
