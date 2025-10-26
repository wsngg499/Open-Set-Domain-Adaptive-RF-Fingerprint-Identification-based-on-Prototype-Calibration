import torch
import torch.nn as nn
import torch.nn.functional as F
from Dist import Dist

class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(ARPLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['ndz'])
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1)

    def forward(self, x, labels=None,prototypes=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p-dist_dot_p

        if labels is None: return logits, 0
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r=torch.abs(_dis_known-self.radius).mean(0)
        #loss_r = self.margin_loss(self.radius, _dis_known, target)  #max(0,-y*(x1-x2)+margin)   (x1,x2,y)
        if prototypes is None:
            loss = loss + self.weight_pl * loss_r
        else:
            dis = (prototypes - self.points).pow(2).mean(1)
            # loss3=self.margin_loss(self.radius,dis,torch.ones(dis.size()).cuda())
            loss3 = torch.abs(dis - self.radius).mean(0)
            loss = loss + self.weight_pl * loss_r + 0.01 * loss3
        return logits, loss

    def fake_loss(self, x):
        self.Dist.eval()
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()
        return loss
