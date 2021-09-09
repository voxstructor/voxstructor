#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader,loadWAV

from torch.cuda.amp import autocast, GradScaler

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__();

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs);

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs);

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data    = data.reshape(-1,data.size()[-1]).cuda() 
        outp    = self.__S__.forward(data)

        if label == None:
            return outp

        else:

            outp    = outp.reshape(self.nPerSpeaker,-1,outp.size()[-1]).transpose(1,0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp,label)

            return nloss, prec1


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__  = speaker_model

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        tstart = time.time()
        
        for data, data_label in loader:

            data    = data.transpose(1,0)

            self.__model__.zero_grad();

            label   = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward();
                self.scaler.step(self.__optimizer__);
                self.scaler.update();       
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward();
                self.__optimizer__.step();


            loss    += nloss.detach().cpu();
            top1    += prec1.detach().cpu()
            counter += 1;
            index   += stepsize;

        

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing (%d) "%(index));
                sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def voiceprint(self, test_path,print_interval=100, num_eval=10, **kwargs):
        self.__model__.eval();
        #读取测试集路径，将数值写入文件中
        se_path = open(os.path.join('humman_dvector.txt'), mode='a')
        for line in open('humman_trials.txt').readlines():
            filename = line.strip().split()[0]
            # in_fpath = os.path.join(test_path, filename[-5:] + '.wav')
            in_fpath = os.path.join(test_path, filename[:7], filename[8:-6], filename[-5:] + '.wav')
            audio = loadWAV(in_fpath, 400)
            audio = torch.FloatTensor(audio)
            inp1 = audio[0].cuda()
            feat = self.__model__(inp1).detach().cpu()
            feat=feat[0].numpy()
            print(filename)
            a = filename + '  ' + str(feat.tolist()).replace(',', '').replace('[', '[ ').replace(']', ' ]') + '\n'
            se_path.write(a)
        se_path.close()

        # # 测试数据
        # se_path = open(os.path.join('dp-id10270.txt'), mode='a')
        # for line in open('dp_trials.txt').readlines():
        #     filename = line.strip().split()[0]
        #     in_fpath = os.path.join(test_path, filename+ '.wav')
        #     audio = loadWAV(in_fpath, 400)
        #     audio = torch.FloatTensor(audio)
        #     inp1 = audio[0].cuda()
        #     feat = self.__model__(inp1).detach().cpu()
        #     feat = feat[0].numpy()
        #     print(filename)
        #     a = filename + '  ' + str(feat.tolist()).replace(',', '').replace('[', '[ ').replace(']', ' ]') + '\n'
        #     se_path.write(a)
        # se_path.close()


        # return feat
        # #去读测试集的特征向量
        # total_path=r'E:\个人\李祁\Voxceleb1\wav'
        # se_dir = r'RES_vector'
        # for i in range(80):
        #     pathstr = 'dpdir' + str(i + 1) + '.txt'
        #     dirs = os.path.join(test_path, pathstr)
        #     se_path = open(os.path.join(se_dir, pathstr), mode='a')
        #     for line in open(dirs).readlines():
        #         filename = line.strip().split()[0]
        #         in_fpath = os.path.join(total_path, filename[:7], filename[8:-6], filename[-5:] + '.wav')
        #         audio = loadWAV(in_fpath, 400)
        #         audio = torch.FloatTensor(audio)
        #         inp1 = audio[0].cuda()
        #         feat = self.__model__(inp1).detach().cpu()
        #         feat = feat[0].numpy()
        #         a = filename + '  ' + str(feat.tolist()).replace(',', '').replace('[', '[ ').replace(']', ' ]') + '\n'
        #         se_path.write(a)
        #     se_path.close()
        #     print(i)




    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, print_interval=100, num_eval=10, **kwargs):
        
        self.__model__.eval();
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()


        ## Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__model__(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__model__.module.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();

            score = -1 * numpy.mean(dist);

            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('')

        return (all_scores, all_labels, all_trials);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.module.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu);
        # loaded_state = torch.load(path, map_location=torch.device('cpu'));
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

