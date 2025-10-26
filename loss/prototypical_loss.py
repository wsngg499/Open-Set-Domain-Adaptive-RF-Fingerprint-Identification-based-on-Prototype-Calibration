import torch
from torch.nn import functional as F
import torch.nn as nn
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

class PrototypicalLoss(nn.CrossEntropyLoss):
    def __init__(self, options):
        super(PrototypicalLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(0))
        self.CroEn_loss = nn.CrossEntropyLoss()
        self.num_classes = options['num_classes']
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def euclidean_dist(self,x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        if d != y.size(1):
            raise Exception
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        l_2=torch.sqrt(torch.pow(x-y, 2).sum(2))
        cos=torch.bmm(F.normalize(y), F.normalize(torch.transpose(x,2,1))).mean(2)
        return l_2

    def forward(self, input, target, prototypes=None):
        classes = torch.LongTensor([i for i in range(self.num_classes)])
        def supp_idxs(c):
            return target.eq(c).nonzero().squeeze(1)

        support_idxs = list(map(supp_idxs, classes))
        if prototypes is None:
            prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
        else:
            prototypes=prototypes
        query_samples = input
        dists = self.euclidean_dist(query_samples, prototypes)
        s = torch.exp(-dists).sum(1).unsqueeze(1).expand(len(dists), self.num_classes)
        #loss_c=F.mse_loss(torch.exp(-dists) / s,F.one_hot(target).float())
        log_p_y = F.log_softmax(torch.exp(-dists) / s, dim=1)

        # loss_c = self.CroEn_loss(log_p_y, target)
        center = prototypes[target, :]
        #loss_c = F.mse_loss(log_p_y, center)
        #loss_r =F.mse_loss(input,center)-0.1*torch.bmm(F.normalize(input.unsqueeze(1)), F.normalize(center.unsqueeze(2))).mean((0,1,2))
        loss_r=F.mse_loss(input,center)
        return loss_r, log_p_y, dists, prototypes









