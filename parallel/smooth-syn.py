import torch
from torch import nn, optim
import os
import torch.nn.functional as F
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import concurrent.futures
import threading
import multiprocessing


# def embedEX(in_fpath):
#     filepath=r'D:\voiceset\dataset\test_wav'
# #     filepath=r'D:\voiceset\dataset\wav'
#     in_fpath=os.path.join(filepath,in_fpath[:7],in_fpath[8:-6],in_fpath[-5:]+'.wav')
#     reprocessed_wav = encoder.preprocess_wav(in_fpath)
#     original_wav, sampling_rate = librosa.load(in_fpath)
#     preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#     embed,_ = encoder.embed_utterance(preprocessed_wav)
#     return embed

# class I2E_Net2(torch.nn.Module):
#     def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
#         super(I2E_Net2, self).__init__()
#         self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim, n_hidden_1), torch.nn.ReLU(True))
#         self.layer2 = torch.nn.Sequential(torch.nn.Linear(n_hidden_1, n_hidden_2), torch.nn.ReLU(True))
#         self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2, out_dim))
#     def forward(self, x):
#         x = x / torch.norm(x, p=2)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x / torch.norm(x, p=2)
#         return x

# class X2E_Net3Smooth(torch.nn.Module):
#     def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
#         super(X2E_Net3Smooth, self).__init__()
#         self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim, n_hidden_1), torch.nn.ReLU(True))
#         self.layer2 = torch.nn.Sequential(torch.nn.Linear(n_hidden_1, n_hidden_2), torch.nn.ReLU(True))
#         self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2, out_dim))
#
#     def forward(self, x):
#         x = x / torch.norm(x, p=2)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x / torch.norm(x, p=2)
#         return x
class D2E_NetSMOOTH(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(D2E_NetSMOOTH, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_dim, n_hidden_1), torch.nn.ReLU(True))
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(n_hidden_1, n_hidden_2), torch.nn.ReLU(True))
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = x / torch.norm(x, p=2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x / torch.norm(x, p=2)
        return x
def save_syn(lines):
    encoder_weights = Path("../encoder/saved_models/pretrained.pt")
    encoder.load_model(encoder_weights)
    vocoder_weights = Path("../vocoder/saved_models/pretrained/pretrained.pt")
    syn_dir = Path("../synthesizer/saved_models/logs-pretrained/taco_pretrained")
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)
    model=D2E_NetSMOOTH(512,400,300,256)
    checkpoint = torch.load(r'..\checkpointD2E\modelsmooth_epoch_148600.ckpt')
    model.load_state_dict(checkpoint['model'])
    # file = open('..\I2E\ivector270-283.txt')
    # lines = file.readlines()
    text = "This is being said in my own voice. The computer has learned to do an impression of me."
    # filesynVS = r'D:\voiceset\dataset\testVS_wav'
    filesynI2E = r'D:\voiceset\dataset\SPK_testD2ESmooth_wav'
    if not os.path.exists(filesynI2E):
        os.makedirs(filesynI2E)
    #SE????????????
    # se_vectors={}
    # for line in open('..\se_vectortest.txt').readlines():
    #     se=line.strip().replace('[', '').replace(']', '').split()
    #     se_vectors[se[0]]=list(map(float, se[1:]))
    for line in lines:
        data = line.strip().replace('[', '').replace(']', '').split()
        filename = data[0]
        # print("filename:", filename)
        # # ??????VS??????in_fpath=os.path.join(filepath,in_fpath[:7],in_fpath[8:-6],in_fpath[-5:]+'.wav')
        # pathvs = os.path.join(filesynVS, filename[:7], filename[8:-6])
        # if not os.path.exists(pathvs):
        #     os.makedirs(pathvs)
        # ??????i2??????
        # pathi2e = os.path.join(filesynI2E, filename[:7], filename[8:-6])
        # if not os.path.exists(pathi2e):
        #     os.makedirs(pathi2e)
        data = data[1:]
        out = list(map(float, data))
        out = torch.tensor(out)
        pred = model(out)
        pred = pred.tolist()
        #     print("pred",pred)
        print("Synthesizing ????????????...",filename)
        specs = synthesizer.synthesize_spectrograms([text], [pred])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        sf.write(os.path.join(filesynI2E, filename+ '.wav'), generated_wav, synthesizer.sample_rate)
        # ??????????????????
        # label = np.array(se_vectors[filename])
        # print("Synthesizing ????????????...",filename)
        # print(filename,'?????????', np.linalg.norm(pred - label))
        # specs = synthesizer.synthesize_spectrograms([text], [label])
        # generated_wav = vocoder.infer_waveform(specs[0])
        # generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        # sf.write(os.path.join(pathvs, filename[-5:] + '.wav'), generated_wav, synthesizer.sample_rate)

#?????????????????????
def QS():
    file = open(r'..\I2E\ivectortest.txt')
    # file2 = open(r'C:\Users\pedestrian\Desktop\re-voiceprint\VS\vs\vs_ivector.txt')
    lines = file.readlines()
    filesynI2E = r'C:\Users\pedestrian\Desktop\re-voiceprint\VS\testVS_wav'
    qslist=[]
    for i,line in enumerate(lines):
        data=data = line.strip().replace('[', '').replace(']', '').split()
        filename = data[0]
        pathi2e = os.path.join(filesynI2E, filename[:7], filename[8:-6],filename[-5:]+'.wav')
        # print(filename,pathi2e)
        if not os.path.isfile(pathi2e):
            qslist.append(line)
    return qslist

if __name__=='__main__':
    file = open('..\D2E\spk_p_dvector.txt')
    lines = file.readlines()
    line1 = lines[:10]
    line2 = lines[10:20]
    line3 = lines[20:30]
    line4 = lines[30:]
    # line5=lines[40:]
    # save_syn(line1)
    p1 = multiprocessing.Process(target=save_syn, name='p1', args=(line1,))
    p2 = multiprocessing.Process(target=save_syn, name='p2', args=(line2,))
    p3 = multiprocessing.Process(target=save_syn, name='p3', args=(line3,))
    p4 = multiprocessing.Process(target=save_syn, name='p4', args=(line4,))
    # p5 = multiprocessing.Process(target=save_syn, name='p5',args=(line5,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()