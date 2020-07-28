import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from datasets.enterface import Enterface_Dense
from models.tsn_resnet_model import Network
from utils import misc
from utils import transforms_video as tfv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
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
parser.add_argument("--backbone", type=str, default="resnet50")
parser.add_argument("--pooling", type=str, default="avg")
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--n_segments", type=int, default=16)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--log_path", type=str, default="logs")
parser.add_argument("--comment", type=str, default="")


def main(args):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    save_name = "tsn_dense_{}_{}_{:d}seg_split{:d}_bs{}_wd{}".format(args.backbone, args.dataset, args.n_segments, args.split, args.batch_size, args.weight_decay)
    if args.comment != '':
        save_name += "_{}".format(args.comment)
    
    args.checkpoint_path = os.path.join(args.checkpoint_path, save_name)
    args.log_path = os.path.join(args.log_path, save_name)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    writer = SummaryWriter(args.log_path)
    logging.basicConfig(filename=os.path.join(args.log_path, 'results.log'), level=logging.INFO)

    train_tfs = transforms.Compose([
        tfv.ToTensorVideo(),
        tfv.RandomResizedCropVideo((args.image_res, args.image_res), scale=(0.8, 1.0)),
        tfv.RandomHorizontalFlipVideo(),
        tfv.NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_tfs = transforms.Compose([
        tfv.ToTensorVideo(),
        tfv.ResizeVideo((args.image_res, args.image_res)),
        tfv.NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if args.dataset == 'enterface':
        dset = Enterface_Dense
    else:
        raise NotImplementedError('Dataset not yet implemented!')

    trainset = dset(train=True, num_segments=1, num_frames_per_clip=args.n_segments, which_split=args.split, transform=train_tfs)
    testset = dset(train=False, num_segments=10, num_frames_per_clip=args.n_segments, sample_uniform=True, which_split=args.split, transform=test_tfs)

    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1, num_workers=args.n_workers)

    net = Network(backbone=args.backbone, pooling=args.pooling).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # resume checkpoint
    try:
        checkpoint = torch.load("%s/checkpoint_last.pth" % args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('Loaded checkpoint from epoch %s with accuracy %s and best accuracy so far %s' % (start_epoch, checkpoint['accuracy'], best_accuracy))
    except FileNotFoundError:
        best_accuracy = 0.0
        start_epoch = 0
        print('Checkpoint not found')

    net = nn.DataParallel(net)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, last_epoch=(start_epoch-1))

    for epoch in range(start_epoch, args.n_epochs):
        train(train_loader, net, criterion, optimizer, epoch, writer=writer)
        acc = validate(test_loader, net, criterion, epoch, writer=writer)

        if acc > best_accuracy:
            best_accuracy = acc
        save_model(net.module, optimizer, epoch, acc, best_accuracy, args.checkpoint_path)
        scheduler.step()

    try:
        checkpoint = torch.load("%s/checkpoint_best.pth" % args.checkpoint_path, map_location='cpu')
        net.module.load_state_dict(checkpoint['state_dict'])
        print('Loaded best checkpoint from epoch %s with accuracy %s' % (checkpoint['epoch'] + 1, checkpoint['accuracy']))
        validate(test_loader, net, criterion, show_confusion=True)
    except FileNotFoundError:
        print('Best checkpoint not found')


def train(loader, model, criterion, optimizer, epoch, writer=None):
    batch_time = misc.AverageMeter()
    data_time = misc.AverageMeter()
    losses = misc.AverageMeter()
    top1 = misc.AverageMeter()
    top5 = misc.AverageMeter()

    model.train()
    for i, (inputs, labels) in enumerate(loader):
        end = time.time()
        inputs, labels = inputs.to(device), labels.to(device)

        n_batch = inputs.size(0)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        optimizer.zero_grad()
        outputs = model(inputs.squeeze(1))
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = misc.accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), n_batch)
        top1.update(prec1[0], n_batch)
        top5.update(prec5[0], n_batch)

        # backpropagation
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % 5 == 4:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch+1, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    if writer is not None:
        writer.add_scalar('Loss/train', losses.avg, epoch)
        writer.add_scalar('Accuracy@1/train', top1.avg, epoch)
        writer.add_scalar('Accuracy@5/train', top5.avg, epoch)
        writer.close()

    logging.info('Epoch: [{0}]\t'
                'Loss {loss.avg:.4f}\t'
                'Prec@1 {top1.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f}'.format(
                epoch+1, loss=losses, top1=top1, top5=top5))


def validate(loader, model, criterion, epoch=0, writer=None, show_confusion=False):
    batch_time = misc.AverageMeter()
    data_time = misc.AverageMeter()
    losses = misc.AverageMeter()
    top1 = misc.AverageMeter()
    top5 = misc.AverageMeter()

    model.eval()

    y_preds = []
    y_trues = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            end = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            n_batch, n_clip, n_channel, n_frame, height, width = inputs.shape
            inputs = inputs.view(n_batch * n_clip, n_channel, n_frame, height, width)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(inputs)
            outputs = outputs.view(n_batch, n_clip, -1).mean(1)
            loss = criterion(outputs, labels)

            if show_confusion:
                y_preds.append(outputs.argmax(dim=1).cpu().numpy())
                y_trues.append(labels.cpu().numpy())

            # measure accuracy and record loss
            prec1, prec5 = misc.accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), n_batch)
            top1.update(prec1[0], n_batch)
            top5.update(prec5[0], n_batch)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % 5 == 4:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i+1, len(loader), batch_time=batch_time, data_time=data_time, 
                        loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if show_confusion:
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
        print(confusion_matrix(y_trues, y_preds))

        logging.info(confusion_matrix(y_trues, y_preds))
        # np.save(os.path.join(writer.log_dir, 'y_trues.npy'), y_trues)
        # np.save(os.path.join(writer.log_dir, 'y_preds.npy'), y_preds)

    if writer is not None:
        writer.add_scalar('Loss/val', losses.avg, epoch)
        writer.add_scalar('Accuracy@1/val', top1.avg, epoch)
        writer.add_scalar('Accuracy@5/val', top5.avg, epoch)
        writer.close()

    logging.info('Loss {loss.avg:.4f}\t'
                'Prec@1 {top1.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f}'.format(
                loss=losses, top1=top1, top5=top5))

    return top1.avg


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
