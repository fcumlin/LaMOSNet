import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
    

class MeanNet(nn.Module):
    
    def __init__(self):
        super(MeanNet, self).__init__()
        self.mean_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.mean_rnn = nn.LSTM(input_size = 512,
                                hidden_size = 128,
                                num_layers = 1,
                                batch_first = True,
                                bidirectional = True)
        
        self.mean_MLP = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )
        
    def forward(self, speech_spectrum):
        batch = speech_spectrum.shape[0]
        time = speech_spectrum.shape[2]
        speech_spectrum = self.mean_conv(speech_spectrum)
        speech_spectrum = speech_spectrum.view((batch, time, 512))
        speech_spectrum, (h, c) = self.mean_rnn(speech_spectrum)
        mos = self.mean_MLP(speech_spectrum) 
        return mos


class BiasNet(nn.Module):
    
    def __init__(self, num_judges):
        super(BiasNet, self).__init__()
        self.bias_increase_spectrum = nn.Conv2d(1, 16, (3,3), padding = (1,1), stride = (1,3))
        self.bias_conv = nn.Sequential(
            nn.Conv2d(18, 32, (3,3), padding = (1,1), stride=(1,3)),
            
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, (3,3), padding = (1,1), stride = (1,3)),
            nn.Conv2d(32, 32, (3,3), padding = (1,1), stride = (1,3)),
            
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.bias_rnn = nn.LSTM(input_size = 128,
                                hidden_size = 64,
                                num_layers = 1,
                                batch_first = True,
                                bidirectional = True)
        
        self.bias_MLP = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,1)
        )
        
        self.embedding = nn.Embedding(num_embeddings = num_judges,
                                      embedding_dim = 86)
        
    def forward(self, speech_spectrum, judge_id, mos):
        batch = speech_spectrum.shape[0]
        time = speech_spectrum.shape[2]
        
        speech_spectrum = self.bias_increase_spectrum(speech_spectrum)
        judge = self.embedding(judge_id)
        judge = judge.unsqueeze(1)
        judge = torch.stack([judge for i in range(time)], dim = 2)
        speech_spectrum = torch.cat([speech_spectrum, judge], dim = 1)
        
        mos = mos / 5 
        mos = torch.stack([mos for i in range(1)], dim = 1)
        mos = torch.stack([mos for i in range(86)], dim = 3)[:,:,:,:,0]
        speech_spectrum = torch.cat([speech_spectrum, mos], dim = 1)
        
        speech_spectrum = self.bias_conv(speech_spectrum)
        speech_spectrum = speech_spectrum.view((batch, time, 128))
        speech_spectrum, (h, c) = self.bias_rnn(speech_spectrum)
        ld_score = self.bias_MLP(speech_spectrum)
        return ld_score


class LaMOSNet(nn.Module):
    
    def __init__(self, num_judges):
        super(LaMOSNet, self).__init__()
        self.MeanNet = MeanNet()
        self.BiasNet = BiasNet(num_judges)
        
    def forward(self, speech_spectrum , judge_id):
        #spectrum should have shape (batch, 1, time, 257)
        mos = self.MeanNet(copy.deepcopy(speech_spectrum))
        ld_score = self.BiasNet(speech_spectrum, judge_id, mos)
        return mos, ld_score
        
    def listener_inference(self, speech_spectrum, judge_id, mos):
        bias_score = self.BiasNet(speech_spectrum, judge_id, mos)
        return bias_score
    
    def mos_inference(self, speech_spectrum):
        mos = self.MeanNet(speech_spectrum)
        return mos
    
    def all_listener(self, speech_spectrum, device, judge_ids = None):
        mos = self.MeanNet(copy.deepcopy(speech_spectrum))
        bias_scores = []
        if not judge_ids:
            judge_ids = [i for i in range(270)]

        for judge_id in judge_ids:
            judge_id = torch.tensor([judge_id]).to(device)
            bias_score = self.BiasNet(copy.deepcopy(speech_spectrum), judge_id, mos)
            bias_score.squeeze(-1)
            bias = torch.mean(bias_score, dim = -1)
            bias_scores.append(bias)
        avg_mos = sum(bias_scores) / len(bias_scores)
        return avg_mos
            
            
        