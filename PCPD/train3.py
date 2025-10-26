import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from util import  AverageMeter,fft,save_networks,one_hot
from gene_model import *
from Dist import Dist
import math

class AAC(nn.Module):

    def forward(self, sim_mat, prob_u, prob_us):

        P = prob_u.matmul(prob_us.t())
        # print("P.min():", P.min().item(), "P.max():", P.max().item())

        loss = -(
            sim_mat * torch.log(P + 1e-7) +
            (1.-sim_mat) * torch.log(1. - P + 1e-7)
        )
        return loss.mean()

def freq_augmentation(data, freq):
    # cfo_det1 = (torch.rand(data.size(0), 1)-0.5)*2*2000
    cfo_det = (torch.rand(data.size(0), 1) * freq).cuda()
    t = torch.arange(320).cuda() / 20e6
    print("data shape:", data.shape)
    print("t shape:", t.shape)
    data1 = data * torch.exp(1j * 2 * torch.pi * cfo_det * t)
    return data1

def phase_augmentation(data, phase):
    phase_det = (torch.rand(data.size(0), 1) * phase).cuda()
    data1 = data * torch.exp(1j * (-phase_det))
    return data1

def process_rff(data):
    sts1 = data[:, 16:16 * 5].requires_grad_(True)
    sts2 = data[:, 16 * 5:16 * 9].requires_grad_(True)
    lts1 = data[:, 192:192 + 64].requires_grad_(True)
    lts2 = data[:, 192 + 64:192 + 64 * 2].requires_grad_(True)
    # print("sts1 shape:", sts1.shape)
    # print("lts1 shape:", lts1.shape)
    rff1 = torch.log(torch.fft.fft(sts1)) - torch.log(torch.fft.fft(lts1))
    rff2 = torch.log(torch.fft.fft(sts2)) - torch.log(torch.fft.fft(lts2))
    rff3 = torch.log(torch.fft.fft(data[:, 16 * 0:16 * 4])) - torch.log(torch.fft.fft(data[:, 160 - 32:160 + 32]))
    # rff = torch.cat((rff1, rff2), dim=-1)
    rff = torch.cat((rff1, rff2, rff3), dim=-1)
    # rff = data_norm(rff)
    data1 = torch.cat((rff.real.unsqueeze(1).float(), rff.imag.unsqueeze(1).float()), dim=1)
    return data1.requires_grad_(True)

def data_norm(data_tensor):
    data_norm = torch.zeros_like(data_tensor, dtype=torch.cfloat)
    for i in range(data_tensor.shape[0]):
        sig_amplitude = torch.abs(data_tensor[i])
        rms = torch.sqrt(torch.mean(sig_amplitude ** 2))
        data_norm[i] = data_tensor[i] / rms
    return data_norm


def sample_gaussian(m, v):
    sample = torch.randn(m.shape).cuda()
    z = m + (v**0.5)*sample
    return z
def kl_normal(qm, qv, pm, pv, yh):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl
def zerograd(net,criterion):
    net.Q.zero_grad()
    net.P.zero_grad()
    net.C.zero_grad()
    net.C2.zero_grad()

def pre_train(net, criterion, optimizer, trainloader, model_path, pre_file_name, **options):
    net.train()
    best_acc = 0
    for epoch in range(300):
        ACC_pre = AverageMeter()
        Loss = AverageMeter()
        for step, (data, labels) in enumerate(trainloader):
            X = Variable(data).cuda()
            Y = Variable(labels).cuda()

            f1 = net.Q(X)
            logits1, loss1 = criterion["mse_dis"](f1, Y)

            optimizer["class_solver"].zero_grad()
            loss1.backward()
            optimizer["class_solver"].step()

            acc = torch.eq(torch.argmax(logits1, dim=1), Y).sum() / data.size(0)
            ACC_pre.update(acc.item())
            Loss.update(loss1.item())

        print("==> Epoch {}/{}".format(epoch + 1, 300))
        print('Pre Train_acc (%): {:.5f} \t Pre Train_loss (%): {:.5f} \t '.format(ACC_pre.avg,Loss.avg))
        if ACC_pre.avg > best_acc and ACC_pre.avg > 0.96:
            best_acc = ACC_pre.avg
            save_networks(net, model_path, pre_file_name,criterion=criterion['mse_dis'])
        if ACC_pre.avg > 0.96 and epoch > 2:
            break
    print("Finished Pre Train!")


def train_cs(netG,netD,net, criterion, optimizer, trainloader,epoch,results,**options):
    torch.cuda.empty_cache()
    net.train()
    netD.train()
    netG.train()
    ACC = AverageMeter()
    ACC2 = AverageMeter()
    ACC3 = AverageMeter()
    ACC4 = AverageMeter()
    LOSS2 = AverageMeter()
    lossr2 = AverageMeter()
    Class_loss = AverageMeter()
    len_dataloader = len(trainloader)

    real_label, fake_label = 1, 0
    lossesG, lossesD = AverageMeter(), AverageMeter()
    for step,(data,labels) in enumerate(trainloader):
        p = float(step + epoch * len_dataloader) / options['max_epoch'] / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        X1 = Variable(data).cuda()
        Y = Variable(labels).cuda()
        source_data = freq_augmentation(X1, 2000)
        source_data = phase_augmentation(source_data, 2)
        X1 = process_rff(source_data)

        gan_target = torch.FloatTensor(Y.size()).fill_(0).cuda()
        noise = torch.FloatTensor(X1.size(0), options['nz'], 1).normal_(0, 1).cuda()
        noise = Variable(noise)
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizer['optimizerD'].zero_grad()
        output = netD(X1)
        errD_real = criterion["criterion_D"](output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion["criterion_D"](output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizer['optimizerD'].step()
        ###########################
        # (2) Update G network    #
        ###########################
        optimizer['optimizerG'].zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterion["criterion_D"](output, targetv)

        # minimize the true distribution
        net.eval()
        x, domain_output1 = net.Q(fake,alpha = alpha)
        errG_F = criterion["mse_dis"].fake_loss(x).mean()
        generator_loss = errG + 0.1 * errG_F
        generator_loss.backward()
        optimizer['optimizerG'].step()

        lossesG.update(generator_loss.item())
        lossesD.update(errD.item())

        # KL divergence
        noise = torch.FloatTensor(X1.size(0), options['nz'], 1).normal_(0, 1).cuda()
        noise = Variable(noise)
        X_1 = netG(noise)
        f_new1, domain_output2  = net.Q(X_1, alpha = alpha)
        #f_new2 = net.M(X_1)

        net.train()
        f1 ,doamin_output3 = net.Q(X1 ,alpha = alpha)
        loss_c, log_p_y, dists, prototypes1 = criterion['class_C'](f1, Y)
        logits1, loss1 = criterion["mse_dis"](f1, Y)
        F_loss_fake = criterion["mse_dis"].fake_loss(f_new1).mean()
        loss = loss_c + loss1+0.1*F_loss_fake

        optimizer["class_solver"].zero_grad()
        loss.backward()
        optimizer["class_solver"].step()

        f2 = net.M(X1)
        loss_c2, log_p_y2, dists2, prototypes2 = criterion['class_C'](f2, Y, prototypes1.detach())
        logits2, loss2 = criterion["mse_dis"](f2, Y, prototypes1.detach())
        #F_loss_fake2 = criterion["mse_dis"].fake_loss(f_new2.detach()).mean()
        loss = loss_c2 + loss2

        optimizer["class_solver2"].zero_grad()
        loss.backward()
        optimizer["class_solver2"].step()

        acc = torch.eq(torch.argmax(logits1, dim=1), Y).sum() / Y.size(0)
        acc2 = torch.eq(torch.argmax(logits2, dim=1), Y).sum() / Y.size(0)
        ACC.update(acc.item())
        ACC2.update(acc2.item())
        Class_loss.update(loss_c.item())
        LOSS2.update(loss1.item())
        lossr2.update(F_loss_fake.item())

    return print(
        "Epoch:{:.1f}\t LOSS2: {:.5f}\t lossr2: {:.5f}\t loss1: {:.5f}\tACC: {:.5f}\tACC2: {:.5f}\tACC3: {:.5f}\tACC4: {:.5f}\tlossesG: {:.5f}\tlossesD: {:.5f}\t"
        .format(epoch + 1, LOSS2.avg, lossr2.avg, Class_loss.avg, ACC.avg, ACC2.avg, ACC3.avg, ACC4.avg,lossesG.avg, lossesD.avg))


def train(net, criterion, optimizer, trainloader, target_train_loader, epoch, results, **options):
    torch.cuda.empty_cache()
    net.train()
    ACC = AverageMeter()
    ACC2 = AverageMeter()
    ACC3 = AverageMeter()
    ACC4 = AverageMeter()
    LOSS1 = AverageMeter()
    lossr2 = AverageMeter()
    LOSS = AverageMeter()
    PL_LOSS = AverageMeter()
    AAC_LOSS = AverageMeter()
    AAC_LOSS3 = AverageMeter()
    CONS_LOSS = AverageMeter()
    Class_loss = AverageMeter()
    len_dataloader = len(trainloader)
    target_iter = iter(target_train_loader)
    F11,F22,y,F33,F44=[],[],[],[],[]
    pseudo_label_threshold = options.get('pseudo_threshold', 0.9)  # 新增：默认阈值0.9，可自定义
    # pseudo_label_threshold = 0.9 + 0.05/(220-epoch)

    for i, (data, labels) in enumerate(trainloader):
        # print(f"Source data shape: {data.shape}")
        p = float(i + epoch * len_dataloader) / options['max_epoch'] / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        X = Variable(data).cuda()
        Y = Variable(labels).cuda()
        # source_data = freq_augmentation(X, 2000)
        # source_data = phase_augmentation(source_data, 2)
        source_data = process_rff(X)
        try:
            target_data, _ = next(target_iter)
        except:
            target_iter = iter(target_train_loader)
            target_data, _ = next(target_iter)
        target_data = Variable(target_data).cuda()
        target_data = process_rff(target_data)
        # print(source_data.shape)
        f1, source_domain_output, _ = net.Q(source_data, alpha=alpha)
        f2, target_domain_output, _ = net.Q(target_data, alpha=alpha)

        source_domain_loss = criterion['loss_domain'](source_domain_output, torch.zeros(source_domain_output.size(0), dtype=torch.long).cuda())
        target_domain_loss = criterion['loss_domain'](target_domain_output, torch.ones(target_domain_output.size(0), dtype=torch.long).cuda())

        loss_c, log_p_y, dists, prototypes1 = criterion['class_C'](f1, Y)
        logits1, loss1 = criterion["mse_dis"](f1, Y)

        # pseudo_logits, _ = criterion["mse_dis"](f2, None)  # 仅拿 logits，不用真标签
        # pseudo_probs = torch.softmax(pseudo_logits, dim=1)
        # pseudo_confidences, pseudo_preds = torch.max(pseudo_probs, dim=1)
        #
        # mask = pseudo_confidences > pseudo_label_threshold
        #
        # prototypes_new0 = prototypes1.clone()
        #
        # if mask.sum() > 0:
        #     pseudo_selected_logits = pseudo_logits[mask]
        #     pseudo_selected_features = f2[mask]
        #     pseudo_selected_preds = pseudo_preds[mask]
        #     assert len(pseudo_selected_features) == len(pseudo_selected_preds), \
        #         f"特征与预测长度不匹配！特征: {len(pseudo_selected_features)}, 预测: {len(pseudo_selected_preds)}"
        #     unique_classes = torch.unique(pseudo_selected_preds)
        #     pseudo_class_prototypes = {}

        #     for cls in unique_classes:
        #         cls_mask = (pseudo_selected_preds == cls)
        #         if cls_mask.sum() > 0:
        #             class_feats = pseudo_selected_features[cls_mask]
        #             class_proto = class_feats.mean(dim=0)  # 平均作为原型
        #             pseudo_class_prototypes[cls.item()] = class_proto
        #
        #     for cls_id, tgt_proto in pseudo_class_prototypes.items():
        #         src_proto = prototypes1[cls_id]
        #         fused_proto = 0.99 * src_proto + 0.01 * tgt_proto  # 可改为其他比例
        #         prototypes_new0[cls_id] = fused_proto
        #
        #     proto2, _, _, _ = criterion['class_C'](pseudo_selected_features, pseudo_selected_preds, prototypes=prototypes_new0)
        #     pl_loss = criterion['loss_class'](pseudo_selected_logits, pseudo_selected_preds)
        # else:
        #     pl_loss = torch.tensor(0.0).cuda()
        #     proto2 = torch.tensor(0.0).cuda()

        pseudo_logits, _ = criterion["mse_dis"](f2, None)  # 仅拿 logits，不用真标签
        # print('ps:',pseudo_logits.shape)
        # print(pseudo_logits)
        pseudo_probs = torch.softmax(pseudo_logits, dim=1)
        # torch.set_printoptions(threshold=1000)
        # torch.set_printoptions(linewidth=10000)
        # print(pseudo_probs)
        pseudo_confidences, pseudo_preds = torch.max(pseudo_probs, dim=1)

        mask = pseudo_confidences > pseudo_label_threshold

        if mask.sum() > 0:
            pseudo_selected_features = f2[mask]
            pseudo_selected_preds = pseudo_preds[mask]

            # _, pl_loss = criterion["mse_dis"](pseudo_selected_features, pseudo_selected_preds)
            pl_loss = criterion['loss_class'](pseudo_selected_features, pseudo_selected_preds)
        else:
            pl_loss = torch.tensor(0.0).cuda()

        # loss = loss_c + source_domain_loss + target_domain_loss + loss1 + 0.1*pl_loss + 0.1*proto2
        # loss = loss_c + source_domain_loss + target_domain_loss + loss1
        # loss = loss_c + loss1 + 0.1*pl_loss
        # loss = loss_c + loss1
        loss = loss_c + source_domain_loss + target_domain_loss + loss1 + 0.1*pl_loss

        optimizer["class_solver"].zero_grad()
        loss.backward()
        optimizer["class_solver"].step()

        acc = torch.eq(torch.argmax(logits1, dim=1), Y).sum() / Y.size(0)
        ACC.update(acc.item())

        Class_loss.update(loss_c.item())
        LOSS1.update(loss1.item())
        LOSS.update(loss.item())
        PL_LOSS.update(pl_loss.item())
        #lossr2.update(loss_1.item())

    # return print("Epoch:{:.1f}\t LOSS2: {:.5f}\t lossr2: {:.5f}\t loss1: {:.5f}\tACC: {:.5f}\tACC2: {:.5f}\tACC3: {:.5f}\tACC4: {:.5f}\t"
    #     .format(epoch+1,LOSS2.avg,lossr2.avg,Class_loss.avg,ACC.avg,ACC2.avg,ACC3.avg,ACC4.avg))

    # return print("Epoch:{:.1f}\t LOSS: {:.5f}\t aac_loss3: {:.12f}\t loss1: {:.5f}\t cons_loss: {:.5f}\tACC: {:.5f}\tACC2: {:.5f}\tACC3: {:.5f}\tACC4: {:.5f}\t"
    #     .format(epoch+1,LOSS.avg,AAC_LOSS3.avg,LOSS1.avg,CONS_LOSS.avg,ACC.avg,ACC2.avg,ACC3.avg,ACC4.avg))

    # 添加训练性能分析（每10个epoch执行一次）
    if epoch % 2 == 0 and options.get('profile_training', False):
        print(f"\n===== 训练性能分析 (Epoch {epoch}) =====")
        # 获取一个批次的训练数据
        data_iter = iter(trainloader)
        data_sample, labels_sample = next(data_iter)
        data_sample = process_rff(data_sample).cuda()
        labels_sample = labels_sample.cuda()

        with torch.no_grad():
            start_time = time.time()
            f1, _, _ = net.Q(data_sample, alpha=0)
            loss_c, _, _, _ = criterion['class_C'](f1, labels_sample)
            forward_time = (time.time() - start_time) * 1000

        # 分析反向传播性能
        start_time = time.time()
        f1, _, _ = net.Q(data_sample, alpha=0)
        loss_c, _, _, _ = criterion['class_C'](f1, labels_sample)
        loss_c.backward()
        backward_time = (time.time() - start_time) * 1000

        print(f"前向传播时间: {forward_time:.4f} ms")
        print(f"反向传播时间: {backward_time:.4f} ms")
        print(f"单批次总时间: {forward_time + backward_time:.4f} ms")

    return print("Epoch:{:.1f}\t LOSS: {:.5f}\t loss1: {:.5f}\t  pl_loss: {:.5f}\t ACC: {:.5f}\tACC2: {:.5f}\tACC3: {:.5f}\tACC4: {:.5f}\t"
        .format(epoch+1,LOSS.avg,LOSS1.avg,PL_LOSS.avg,ACC.avg,ACC2.avg,ACC3.avg,ACC4.avg))