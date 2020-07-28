import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class NT_Xent(nn.Module):
    def __init__(self, temperature, batch_size, device):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# https://github.com/Kurumi233/Mutual-Channel-Loss
class MCLoss(nn.Module):
    def __init__(self, num_classes=200, cnums=[10, 11], cgroups=[152, 48], p=0.4, lambda_=10):
        super().__init__()
        if isinstance(cnums, int): cnums = [cnums]
        elif isinstance(cnums, tuple): cnums = list(cnums)
        assert isinstance(cnums, list), print("Error: cnums should be int or a list of int, not {}".format(type(cnums)))
        assert sum(cgroups) == num_classes, print("Error: num_classes != cgroups.")

        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, feat, targets):
        n, c, h, w = feat.size()
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[:i + 1]))
        # L_div branch
        feature = feat
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(F.softmax(feature[:, sp[i - 1]:sp[i]].view(n, -1, h * w), dim=2).view(n, -1, h, w)) # Softmax

        l_div = 0.
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = F.max_pool2d(features.view(n, -1, h * w), kernel_size=(self.cnums[i], 1), stride=(self.cnums[i], 1))
            l_div = l_div + (1.0 - torch.mean(torch.sum(features, dim=2)) / (self.cnums[i] * 1.0))

        # L_dis branch
        mask = self._gen_mask(self.cnums, self.cgroups, self.p).expand_as(feat)
        if feat.is_cuda: mask = mask.cuda()

        feature = mask * feat  # CWA
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(feature[:, sp[i - 1]:sp[i]])

        dis_branch = []
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = F.max_pool2d(features.view(n, -1, h * w), kernel_size=(self.cnums[i], 1), stride=(self.cnums[i], 1))
            dis_branch.append(features)

        dis_branch = torch.cat(dis_branch, dim=1).view(n, -1, h, w)  # CCMP
        dis_branch = self.avgpool(dis_branch).view(n, -1)  # GAP

        l_dis = self.celoss(dis_branch, targets)

        return l_dis + self.lambda_ * l_div

    def _gen_mask(self, cnums, cgroups, p):
        """
        :param cnums:
        :param cgroups:
        :param p: float, probability of random deactivation
        """
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(-1,)
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
            drop_idx = np.stack(drop_idx, axis=0).reshape(-1,)
            foo[drop_idx] = 0.
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar