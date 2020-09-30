import logging
import os
import time

import hydra
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tfs
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import define_dataset
from models.networks import define_net
from utils import misc, transforms_video as tfv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = True

@hydra.main(config_path="options", config_name="config")
def main(args):
    print(args)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Init save and logging
    save_name = "{}_{}_{:d}seg_split{:d}_bs{}_wd{}_{}_cam{}".format(
        args.model.arch, args.data.dataset, args.data.n_segments, 
        args.data.split, args.data.batch_size, args.experiment.weight_decay,
        'static' if args.data.static else 'dynamic', args.data.camera    
    )
    if args.experiment.comment != '':
        save_name += "_{}".format(args.experiment.comment)

    checkpoint_path = os.path.join('checkpoints', save_name)
    log_path = os.path.join('logs', save_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    # Init dataset transformations
    train_tfs = tfs.Compose([
        tfv.ToTensorVideo(),
        tfv.RandomResizedCropVideo((args.data.image_res, args.data.image_res), scale=(0.8, 1.0)),
        tfv.RandomHorizontalFlipVideo(),
        tfv.NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_tfs = tfs.Compose([
        tfv.ToTensorVideo(),
        tfv.ResizeVideo((args.data.image_res, args.data.image_res)),
        tfv.NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Datasets
    dset = define_dataset(dataset=args.data.dataset, sampling=args.data.sampling)

    dataset_config = {
        'static': args.data.static,
        'num_segments': args.data.n_segments,
        'which_cam': args.data.camera,
        'which_split': args.data.split
    }
    trainset = dset(train=True, transform=train_tfs, **dataset_config)
    testset = dset(train=False, transform=test_tfs, **dataset_config)

    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.data.batch_size, num_workers=args.experiment.n_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=args.data.batch_size, num_workers=args.experiment.n_workers)

    # Networks
    net = define_net(**args.model, n_class=args.data.n_class).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.experiment.learning_rate, betas=(args.experiment.beta1, args.experiment.beta2), weight_decay=args.experiment.weight_decay)
    scaler = amp.GradScaler(enabled=(USE_AMP))

    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint
    try:
        checkpoint = torch.load("%s/checkpoint_last.pth" % checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('Loaded checkpoint from epoch %s with accuracy %s and best accuracy so far %s' % (start_epoch, checkpoint['accuracy'], best_accuracy))
    except FileNotFoundError:
        best_accuracy = 0.0
        start_epoch = 0
        print('Checkpoint not found')

    # Train
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.experiment.n_epochs, last_epoch=(start_epoch-1))
    
    for epoch in range(start_epoch, args.experiment.n_epochs):
        logging.info('Epoch {}'.format(epoch + 1))
        train(args, train_loader, net, scaler, criterion, optimizer, epoch, writer=writer)
        acc = validate(args, test_loader, net, criterion, epoch, writer=writer)

        if acc > best_accuracy:
            best_accuracy = acc
        save_model(net, optimizer, scaler, epoch, acc, best_accuracy, checkpoint_path)
        scheduler.step()

    # Final validation
    try:
        checkpoint = torch.load("%s/checkpoint_best.pth" % checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded best checkpoint from epoch %s with accuracy %s' % (checkpoint['epoch'] + 1, checkpoint['accuracy']))
        validate(args, test_loader, net, criterion, show_confusion=True)
    except FileNotFoundError:
        print('Best checkpoint not found')


def train(args, loader, model, scaler, criterion, optimizer, epoch, writer=None):
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
        with amp.autocast(enabled=USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        prec1, prec5 = misc.accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), n_batch)
        top1.update(prec1[0], n_batch)
        top5.update(prec5[0], n_batch)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % 10 == 9:
            message = ('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch+1, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(message)
            logging.info(message)

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


def validate(args, loader, model, criterion, epoch=0, writer=None, show_confusion=False):
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

            n_batch = inputs.size(0)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if show_confusion:
                y_preds.append(outputs.argmax(dim=1).cpu().numpy())
                y_trues.append(labels.cpu().numpy())

            # measure accuracy and record loss
            # val_step += 1
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

        message = ('* Loss {loss.avg:.4f}\t'
                'Prec@1 {top1.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f}').format(loss=losses, top1=top1, top5=top5)
        print(message)
        logging.info(message)

    if show_confusion:
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
        confusion = confusion_matrix(y_trues, y_preds)
        print(confusion)
        logging.info("\n{}".format(confusion))

    if writer is not None:
        writer.add_scalar('Loss/val', losses.avg, epoch)
        writer.add_scalar('Accuracy@1/val', top1.avg, epoch)
        writer.add_scalar('Accuracy@5/val', top5.avg, epoch)
        writer.close()

    return top1.avg


def save_model(model, optimizer, scaler, epoch, epoch_accuracy, best_accuracy, checkpoint_path):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'accuracy': epoch_accuracy,
        'best_accuracy': best_accuracy,
        'epoch': epoch,
    }

    torch.save(checkpoint, '%s/checkpoint_last.pth' % (checkpoint_path))

    if epoch_accuracy >= best_accuracy:
        torch.save(checkpoint, '%s/checkpoint_best.pth' % (checkpoint_path))


if __name__ == "__main__":
    main()
