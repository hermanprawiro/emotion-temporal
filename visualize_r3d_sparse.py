import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from datasets.enterface import Enterface_Sparse
from datasets.ctbc import CTBC_Sparse
from models.r3d_model import Network
from utils import misc
from utils import transforms_video as tfv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, default="r3d")
parser.add_argument("--dataset", type=str, default="enterface")
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=112)
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--n_segments", type=int, default=16)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--log_path", type=str, default="logs")
parser.add_argument("--comment", type=str, default="")


def main(args):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    save_name = "{}_{}_{:d}seg_split{:d}_bs{}_wd{}".format(args.arch, args.dataset, args.n_segments, args.split, args.batch_size, args.weight_decay)
    if args.comment != '':
        save_name += "_{}".format(args.comment)
    
    args.checkpoint_path = os.path.join(args.checkpoint_path, save_name)
    args.log_path = os.path.join(args.log_path, save_name)
    args.result_path = os.path.join(args.result_path, save_name)
    os.makedirs(args.result_path, exist_ok=True)
    # os.makedirs(args.checkpoint_path, exist_ok=True)
    # os.makedirs(args.log_path, exist_ok=True)
    # writer = SummaryWriter(args.log_path)
    # logging.basicConfig(filename=os.path.join(args.log_path, 'results.log'), level=logging.INFO)

    test_tfs = transforms.Compose([
        tfv.ToTensorVideo(),
        tfv.ResizeVideo((args.image_res, args.image_res)),
        tfv.NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if args.dataset == 'enterface':
        dset = Enterface_Sparse
        n_class = 6
    elif args.dataset == 'ctbc':
        dset = CTBC_Sparse
        n_class = 7
    else:
        raise NotImplementedError('Dataset not yet implemented!')

    trainset = dset(train=True, num_segments=args.n_segments, which_split=args.split, transform=test_tfs)
    testset = dset(train=False, num_segments=args.n_segments, which_split=args.split, transform=test_tfs)

    net = Network(arch=args.arch, n_class=n_class).to(device)

    try:
        checkpoint = torch.load("%s/checkpoint_best.pth" % args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded best checkpoint from epoch %s with accuracy %s' % (checkpoint['epoch'] + 1, checkpoint['accuracy']))
        validate(testset, net, args, "val")
    except FileNotFoundError:
        print('Best checkpoint not found')


def validate(dataset, model, args, visualization_prefix="val"):
    batch_time = misc.AverageMeter()
    data_time = misc.AverageMeter()
    top1 = misc.AverageMeter()
    top5 = misc.AverageMeter()

    model.eval()
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=4, num_workers=args.n_workers)
    inverse_norm = tfv.NormalizeVideo([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            end = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            n_batch = inputs.size(0)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(inputs)
            
            original_inputs = torch.stack([inverse_norm(row) for row in inputs], dim=0).cpu()

            results = original_inputs * 0.7
            results[:, :, 0] = put_text(results[:, :, 0], outputs, labels, dataset.class_dict_decode)

            results = results.permute(0, 2, 1, 3, 4).reshape(n_batch * args.n_segments, args.image_ch, args.image_res, args.image_res)

            torchvision.utils.save_image(results, os.path.join(args.result_path, "%s_%04d.jpg" % (visualization_prefix, i)), normalize=False, nrow=16)

            # measure accuracy and record loss
            prec1, prec5 = misc.accuracy(outputs, labels, topk=(1, 5))
            top1.update(prec1[0], n_batch)
            top5.update(prec5[0], n_batch)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % 5 == 4:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i+1, len(loader), batch_time=batch_time, data_time=data_time, 
                        top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
