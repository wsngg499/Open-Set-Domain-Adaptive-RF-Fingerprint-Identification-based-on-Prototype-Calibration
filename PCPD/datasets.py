import os
import torch
import numpy as np
import torch.utils.data as Data
import pickle as pickle
from util import one_hot

def Filter(known,x,y,size,SNR):
    datas,snr, targets = x,y[0],y[1]

    new_datas, new_targets,new_targets2= [],[],[]
    for i in range(len(datas)):
        if targets[i] in known and snr[i] in SNR:
            new_datas.append(datas[i])
            new_targets.append(known.index(targets[i]))

            if targets[i]==0 or targets[i]==1 or targets[i]==2 or targets[i]==3:
                new_targets2.append(0)
            elif targets[i]==4 or targets[i]==5 or targets[i]==6:
                new_targets2.append(1)
            else:
                new_targets2.append(2)

    datas, targets, targets2=np.array(new_datas), np.array(new_targets), np.array(new_targets2)

    datas3 = torch.sqrt(torch.Tensor(datas[:, :, 0]) ** 2 + torch.Tensor(datas[:, :, 1]) ** 2).resize(len(datas), 1, size)
    datas_fft = torch.complex(torch.Tensor(datas[:, :, 0]), torch.Tensor(datas[:, :, 1]))
    #datas_fft = torch.stft(datas_fft,size)
    datas_fft = torch.fft.fft(datas_fft)
    datas_fft = torch.cat((torch.sort(datas_fft.real,dim=1)[0], torch.sort(datas_fft.imag,dim=1)[0]), dim=1).resize(len(datas), 2, size)
    #datas_fft = (torch.sort(torch.sqrt(datas_fft.real**2+datas_fft.imag**2), dim=1)[0]).resize(len(datas), 1, size)
    #datas_fft = torch.cat((datas_fft.real,datas_fft.imag),dim=1).resize(
    #    len(datas), 2, size)
    datas31 = (torch.sort(datas3, dim=2)[0]).resize(len(datas), 1, size)
    datas1 = torch.bmm(torch.Tensor(
        [[torch.cos(torch.Tensor([30 * np.pi / 180])), torch.sin(torch.Tensor([30 * np.pi / 180]))],
         [-torch.sin(torch.Tensor([30 * np.pi / 180])), torch.cos(torch.Tensor([30 * np.pi / 180]))]]).unsqueeze(
        0).expand(len(datas), 2, 2), torch.Tensor(datas).resize(len(datas), 2,size))
    datas2 = torch.bmm(torch.Tensor([[1, 0], [0, -1]]).unsqueeze(0).expand(len(datas), 2, 2),
                       torch.Tensor(datas).resize(len(datas), 2, size))
    #datas3 = (torch.sqrt(torch.ones(torch.Tensor(datas[:, :, 0]).size())-torch.Tensor(datas[:, :, 0]))\
             #* torch.Tensor(datas[:, :, 1])\
             #+torch.sqrt(torch.ones(torch.Tensor(datas[:, :, 0]).size())-torch.Tensor(datas[:, :, 1]))* torch.Tensor(datas[:, :, 0]))\
             #.resize(len(datas), 1, 512)

    datas4  = torch.bmm(torch.Tensor([[1,torch.tan(torch.Tensor([45 * np.pi / 180])) ], [0, 1]]).unsqueeze(0).expand(len(datas), 2, 2),
                       torch.Tensor(datas).resize(len(datas), 2,size))
    datas0=torch.atan(torch.Tensor(datas[:, :, 1])/torch.Tensor(datas[:, :, 0])).resize(len(datas), 1,size)
    datas6 = torch.Tensor(datas).permute(0,2,1)
    datas5 = torch.bmm(torch.Tensor(
        [[torch.cos(torch.Tensor([60 * np.pi / 180])), torch.sin(torch.Tensor([60* np.pi / 180]))],
         [-torch.sin(torch.Tensor([60* np.pi / 180])), torch.cos(torch.Tensor([60* np.pi / 180]))]]).unsqueeze(
        0).expand(len(datas), 2, 2), torch.Tensor(datas).resize(len(datas), 2, size))

    datas =torch.cat((datas6,datas_fft,datas0,datas3),dim=1)
    #datas = torch.cat((torch.Tensor(datas[:, :, 0]),torch.Tensor(datas[:, :, 1]), -1*torch.Tensor(datas[:, :, 0]),-1*torch.Tensor(datas[:, :, 1])), dim=1)
    #targets=torch.cat((torch.LongTensor(targets).unsqueeze(1),torch.LongTensor(targets2).unsqueeze(1)),dim=1)
    dataset = Data.TensorDataset(torch.Tensor(datas), torch.LongTensor(targets))
    #dataset = Data.TensorDataset(torch.Tensor(datas).resize(len(datas1),size,4).permute(0,2,1), targets)
    return dataset

def Filter1(known,x,y,size):
    datas,targets = x,y
    new_datas, new_targets= [],[]

    for i in range(len(datas)):
        if targets[i] in known:
            new_datas.append(datas[i,:,:,0:size])
            new_targets.append(known.index(targets[i]))

    datas, targets=torch.Tensor(np.array(new_datas)).squeeze(1), torch.LongTensor(np.array(new_targets))
    datas_fft = torch.complex(datas[:, 0,:], datas[:,1,: ])
    datas_fft = torch.fft.fft(datas_fft)
    datas_fft = torch.cat((datas_fft.real, datas_fft.imag), dim=1).resize(
        len(datas), 2, size)
    datas3 = (torch.sqrt(datas[:, 0,:] ** 2 + datas[:, 1,:] ** 2)).resize(
        len(datas), 1, size)
    datas0=torch.atan(datas[:, 1,:]/(datas[:, 0,:]+1)).resize(len(datas), 1,size)
    datas_ = torch.cat((datas,datas_fft,datas0,datas3), dim=1)
    dataset = Data.TensorDataset(datas_, targets)
    return dataset


def Filter2(known,x,y,size):
    datas,targets = x,y
    new_datas, new_targets= [],[]

    for i in range(len(datas)):
        if targets[i] in known:
            new_datas.append(datas[i,0:size,:])
            new_targets.append(known.index(targets[i]))

    datas, targets=torch.Tensor(np.array(new_datas)).squeeze(1), torch.LongTensor(np.array(new_targets))
    datas_fft = torch.complex(datas[:,:,0], datas[:,:,1])
    datas_fft = torch.fft.fft(datas_fft)
    datas_fft = torch.cat((datas_fft.real, datas_fft.imag), dim=1).resize(
        len(datas), 2, size)
    datas3 = (torch.sqrt(datas[:,:,0] ** 2 + datas[:,:,1] ** 2)).resize(
        len(datas),1, size)
    datas0=torch.atan(datas[:,:,1]/(datas[:,:,0]+1)).resize(
        len(datas), 1, size)
    datas6 =datas.permute(0, 2, 1)
    datas_ = torch.cat((datas6,datas_fft,datas0,datas3), dim=1)
    dataset = Data.TensorDataset(datas_, targets)
    return dataset
class RF_2019(object):
    def __init__(self, known, unknown,dataroot='/home/hz/LTT/Radio/Task-02', use_gpu=True,
                 num_workers=48, batch_size=256, img_size=512,SNR=0):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        trainset_x = np.load(os.path.join(dataroot, 'radio12CNormTrainX.npy'))
        trainser_y=np.load(os.path.join(dataroot, 'radio12CNormTrainSnrY.npy'))
        print('All Train Data:', len(trainset_x))
        trainset=Filter(self.known,trainset_x,trainser_y,img_size,SNR)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valset_x =np.load(os.path.join(dataroot, 'radio12CNormValX.npy'))
        valset_y = np.load(os.path.join(dataroot, 'radio12CNormValSnrY.npy'))
        print('All Val Data:', len(valset_x))
        valset=Filter(self.known,valset_x,valset_y,img_size,SNR)

        self.val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory,
        )

        valoutset_x = np.load(os.path.join(dataroot, 'radio12CNormValX.npy'))
        valoutset_y = np.load(os.path.join(dataroot, 'radio12CNormValSnrY.npy'))
        valoutset=Filter(self.unknown,valoutset_x,valoutset_y,img_size,SNR)
      
        self.val_out_loader = torch.utils.data.DataLoader(
            valoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset_x = np.load(os.path.join(dataroot, 'radio12CNormTestX.npy'))
        testset_y = np.load(os.path.join(dataroot, 'radio12CNormTestSnrY.npy'))
        print('All Test Data:', len(testset_x))
        testset = Filter(self.known, testset_x, testset_y,img_size,SNR)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testoutset_x = np.load(os.path.join(dataroot, 'radio12CNormTestX.npy'))
        testoutset_y = np.load(os.path.join(dataroot, 'radio12CNormTestSnrY.npy'))
        testoutset = Filter(self.unknown, testoutset_x, testoutset_y,img_size,SNR)

        self.test_out_loader = torch.utils.data.DataLoader(
            testoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        print('Train: ', len(trainset), 'Val: ', len(valset), 'Val_Out: ', len(valoutset))
        print('All Test: ', (len(testset) + len(testoutset)))


class RF_2018(object):
    def __init__(self, known, unknown, dataroot='/home/hz/LTT/Radio/Task-00', use_gpu=True,
                 num_workers=48, batch_size=256, img_size=1024, SNR=0):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        trainset_x1 = np.load(os.path.join(dataroot, 'train_x1.npy'))
        trainset_y1 = np.load(os.path.join(dataroot, 'train_y1.npy'))
        trainset_x2 = np.load(os.path.join(dataroot, 'train_x2.npy'))
        trainset_y2 = np.load(os.path.join(dataroot, 'train_y2.npy'))
        trainset_x=np.concatenate((trainset_x1,trainset_x2),axis=0)
        trainset_y = np.concatenate((trainset_y1, trainset_y2), axis=1)
        print('All Train Data:', len(trainset_x))
        trainset = Filter(self.known, trainset_x, trainset_y, img_size, SNR)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset_x1 = np.load(os.path.join(dataroot, 'test_x1.npy'))
        testset_y1 = np.load(os.path.join(dataroot, 'test_y1.npy'))
        testset_x2 = np.load(os.path.join(dataroot, 'test_x2.npy'))
        testset_y2 = np.load(os.path.join(dataroot, 'test_y2.npy'))
        testset_x = np.concatenate((testset_x1, testset_x2), axis=0)
        testset_y = np.concatenate((testset_y1, testset_y2), axis=1)
        print('All Test Data:', len(testset_x))
        testset = Filter(self.known, testset_x, testset_y, img_size, SNR)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testoutset = Filter(self.unknown, testset_x, testset_y, img_size, SNR)

        self.test_out_loader = torch.utils.data.DataLoader(
            testoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset))
        print('All Test: ', (len(testset) + len(testoutset)))

class RF_2016(object):
    def __init__(self, known, unknown,dataroot='/home/hz/LTT/Radio/Task-01', use_gpu=True,
                 num_workers=48, batch_size=256, img_size=128,SNR=0):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown

        print('Selected Labels: ', known)
        print('Selected Unknown Labels: ', unknown)
        pin_memory = True if use_gpu else False

        trainset_x = np.load(os.path.join(dataroot, 'X_train.npy'))
        trainser_y = np.load(os.path.join(dataroot, 'Y_train.npy'))
        print('All Train Data:', len(trainset_x))
        trainset = Filter(self.known, trainset_x, trainser_y, img_size,SNR)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valset_x = np.load(os.path.join(dataroot, 'X_test.npy'))
        valset_y = np.load(os.path.join(dataroot, 'Y_test.npy'))
        print('All Val Data:', len(valset_x))
        valset = Filter(self.known, valset_x, valset_y, img_size,SNR)
        print(valset_y)
        self.val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valoutset_x = np.load(os.path.join(dataroot, 'X_test.npy'))
        valoutset_y = np.load(os.path.join(dataroot, 'Y_test.npy'))
        valoutset = Filter(self.unknown, valoutset_x, valoutset_y, img_size,SNR)

        self.val_out_loader = torch.utils.data.DataLoader(
            valoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset_x = np.load(os.path.join(dataroot, 'X_test.npy'))
        testset_y = np.load(os.path.join(dataroot, 'Y_test.npy'))
        print('All Test Data:', len(testset_x))
        testset = Filter(self.known, testset_x, testset_y, img_size,SNR)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testoutset_x = np.load(os.path.join(dataroot, 'X_test.npy'))
        testoutset_y = np.load(os.path.join(dataroot, 'Y_test.npy'))
        testoutset = Filter(self.unknown, testoutset_x, testoutset_y, img_size,SNR)

        self.test_out_loader = torch.utils.data.DataLoader(
            testoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Val: ', len(valset), 'Val_Out: ', len(valoutset))
        print('All Test: ', (len(testset) + len(testoutset)))
class RF_2021(object):
    def __init__(self, known, unknown,dataroot='/home/hz/LTT/Radio/Task-02', use_gpu=True,
                 num_workers=48, batch_size=256, img_size=500):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown

        print('Selected Labels: ', known)
        print('Selected Unknown Labels: ', unknown)
        pin_memory = True if use_gpu else False

        trainset_x = np.load(os.path.join(dataroot, 'ForTrain/X_train.npy'))
        trainser_y = np.load(os.path.join(dataroot, 'ForTrain/YTrainData.npy'))

        print('All Train Data:', len(trainset_x))
        trainset = Filter1(self.known, trainset_x, trainser_y, img_size)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset_x = np.load(os.path.join(dataroot, 'ForTrain/X_test.npy'))
        testset_y = np.load(os.path.join(dataroot, 'ForTrain/YTestData.npy'))
        print('All Test Data:', len(testset_x))
        testset = Filter1(self.known, testset_x, testset_y, img_size)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testoutset = Filter1(self.unknown, testset_x, testset_y, img_size)

        self.test_out_loader = torch.utils.data.DataLoader(
            testoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Test_Out: ', len(testoutset))
        print('All Test: ', (len(testset) + len(testoutset)))

class RF_2022(object):
    def __init__(self, known, unknown,dataroot='/home/hz/LTT/Radio/Task-04', use_gpu=True,
                 num_workers=48, batch_size=256, img_size=500):
        self.num_classes = len(known)
        self.known = known
        self.unknown = unknown

        print('Selected Labels: ', known)
        print('Selected Unknown Labels: ', unknown)
        pin_memory = True if use_gpu else False

        trainset_x = np.load(os.path.join(dataroot, '3040/ZGDTrainX_106.npy'))
        trainser_y = np.load(os.path.join(dataroot, '3040/ZGDTrainY_106.npy'))

        print('All Train Data:', len(trainset_x))
        trainset = Filter2(self.known, trainset_x, trainser_y, img_size)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset_x = np.load(os.path.join(dataroot, '3040/ZGDTestX_106.npy'))
        testset_y = np.load(os.path.join(dataroot, '3040/ZGDTestY_106.npy'))
        print('All Test Data:', len(testset_x))
        testset = Filter2(self.known, testset_x, testset_y, img_size)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testoutset = Filter2(self.unknown, testset_x, testset_y, img_size)

        self.test_out_loader = torch.utils.data.DataLoader(
            testoutset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Test_Out: ', len(testoutset))
        print('All Test: ', (len(testset) + len(testoutset)))


if __name__=='__main__':
    path=r'/home/hz/LTT/Radio/Task-01/RML2016.10a_dict.pkl'
    with open(path, 'rb') as p_f:
        Xd = pickle.load(p_f, encoding="latin-1")
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []

    for mod in range(len(mods)):
        for snr in snrs:
            X.append(Xd[(mods[mod], snr)])
            for i in range(Xd[(mods[mod], snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)
    # %%
    print(snrs)
    print(mods)
    np.random.seed(2016)
    n_examples = X.shape[0]
    Y = lbl
    X_Train,Y_Train,X_Test,Y_Test=[],[],[],[]
    for i in set(Y):
        id_y= [j for (j, v) in enumerate(Y) if v == i]
        n_train =int(len(id_y) * 0.7)
        train_idx = np.random.choice(range(0, len(id_y)), size=int(n_train), replace=False)
        test_idx = list(set(range(0, len(id_y))) - set(train_idx))  # label
        X_Train.append(X[np.array(id_y)[train_idx]])
        X_Test.append(X[np.array(id_y)[test_idx]])
        print(i)
        Y_Train.append([i]*n_train)
        Y_Test.append([i]*(len(id_y)-n_train))
    X_Train = np.vstack(X_Train)
    X_Test = np.vstack(X_Test)
    X_Train=X_Train.transpose(0,2,1)
    X_Test = X_Test.transpose(0,2,1)
    Y_Train = np.vstack(Y_Train).T
    Y_Test =np.vstack(Y_Test).T
    Y_Train = np.vstack((Y_Train[1],Y_Train[0]))
    Y_Test = np.vstack((Y_Test[1],Y_Test[0]))
    print(Y_Train)
    print(len(Y_Test))

    #np.save('/home/hz/LTT/Radio/Task-01/X_train', X_Train)
    #np.save('/home/hz/LTT/Radio/Task-01/X_test', X_Test)
    #np.save('/home/hz/LTT/Radio/Task-01/Y_train', Y_Train)
    #np.save('/home/hz/LTT/Radio/Task-01/Y_test', Y_Test)

    '''
    n_train = n_examples * 0.7
    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    X_train = X[train_idx]
    X_test = X[test_idx]
    trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_train = np.array(trainy)
    Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    np.save('/home/hz/LTT/Radio/Task-01/X_train', X_train)
    np.save('/home/hz/LTT/Radio/Task-01/X_test', X_test)
    np.save('/home/hz/LTT/Radio/Task-01/Y_train', Y_train)
    np.save('/home/hz/LTT/Radio/Task-01/Y_test', Y_test)
    '''


