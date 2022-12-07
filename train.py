import argparse
import copy
import math
import os
import random
import typing

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import get_dataloader, get_dataset
from EMA import WeightExponentialMovingAverage
from model import LaMOSNet


parser = argparse.ArgumentParser(description='Training LaMOSmodel.')
parser.add_argument('--num_epochs', type=int, help='Number of epochs.')
parser.add_argument('--log_valid', type=int, help='Logging valid score each log_valid epochs.')
parser.add_argument('--log_epoch', type=int, help='Logging training during a global run.')
parser.add_argument('--data_path', type = str, help='Path to data.')
parser.add_argument('--id_table', type = str, help='Path to ID of judges.')
parser.add_argument('--save_path', type = str, help='Path to save the model.')
args = parser.parse_args()


def valid(model, 
          dataloader, 
          systems,
          steps,
          prefix,
          device,
          MSE_list,
          LCC_list,
          SRCC_list):
    model.eval()

    mos_predictions = []
    mos_targets = []
    mos_predictions_sys = {system:[] for system in systems}
    true_sys_mean_scores = {system:[] for system in systems}

    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        wav, filename, _, mos, _ = batch
        sys_names = list(set([name.split("_")[0] for name in filename]))
        wav = wav.to(device)
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            try:
                mos_prediction = model.mos_inference(speech_spectrum = wav)
                mos_prediction = mos_prediction.squeeze(-1)
                mos_prediction = torch.mean(mos_prediction, dim = -1)

                mos_prediction = mos_prediction.cpu().detach().numpy()
                mos_predictions.extend(mos_prediction.tolist())
                mos_targets.extend(mos.tolist())
                for j, sys_name in enumerate(sys_names):
                    mos_predictions_sys[sys_name].append(mos_prediction[j])
                    true_sys_mean_scores[sys_name].append(mos_targets[j])
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[Runner] - CUDA out of memory at step {global_step}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    mos_predictions = np.array(mos_predictions)
    mos_targets = np.array(mos_targets)
    mos_predictions_sys = np.array([np.mean(scores) for scores in mos_predictions_sys.values()])
    true_sys_mean_scores = np.array([np.mean(scores) for scores in true_sys_mean_scores.values()])
    
    utt_MSE=np.mean((mos_targets-mos_predictions)**2)
    utt_LCC=np.corrcoef(mos_targets, mos_predictions)[0][1]
    utt_SRCC=scipy.stats.spearmanr(mos_targets, mos_predictions)[0]
    
    sys_MSE=np.mean((true_sys_mean_scores-mos_predictions_sys)**2)
    sys_LCC=np.corrcoef(true_sys_mean_scores, mos_predictions_sys)[0][1]
    sys_SRCC=scipy.stats.spearmanr(true_sys_mean_scores, mos_predictions_sys)[0]
    
    MSE_list.append(utt_MSE)
    LCC_list.append(utt_LCC) 
    SRCC_list.append(utt_SRCC)
    print(
        f"\n[{prefix}][{steps}][UTT][ MSE = {utt_MSE:.4f} | LCC = {utt_LCC:.4f} | SRCC = {utt_SRCC:.4f} ] [SYS][ MSE = {sys_MSE:.4f} | LCC = {sys_LCC:.4f} | SRCC = {sys_SRCC:.4f} ]\n"
    )

    model.train()
    return MSE_list, LCC_list, SRCC_list, sys_SRCC


def train(num_epochs,
          log_valid,
          log_epoch,
          train_set,
          valid_set,
          test_set,
          train_loader,
          valid_loader,
          test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaMOSNet(num_judges = 5000).to(device)
    momentum_model = LaMOSNet(num_judges = 5000).to(device) 
    for param in momentum_model.parameters(): 
        param.detach_()
    momentum_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_momentum = WeightExponentialMovingAverage(model, momentum_model) 
    optimizer.zero_grad()
    criterion = F.mse_loss

    backward_steps = 0
    all_loss = []
    mean_losses = []
    bias_losses = []
    lamb = 4
    lamb_c = 1
    best_LCC = -1
    best_LCC_teacher = -1
    best_sys_SRCC = -1
    
    MSE_list = []
    LCC_list = []
    SRCC_list = []
    train_loss = []
    MSE_teacher, LCC_teacher, SRCC_teacher = [], [], []
    
    model.train()
    epoch = 0
    while epoch <= num_epochs:
        if epoch == 5:
            optimizer_momentum.set_alpha(alpha = 0.999) 
        
        for i, batch in enumerate(tqdm(train_loader, ncols=0, desc="Train", unit=" step")):
            try:
                wavs, filename, judge_ids, means, scores = batch
                wavs = wavs.to(device)
                wavs = wavs.unsqueeze(1)
                judge_ids = judge_ids.to(device)
                means = means.to(device)
                scores = scores.to(device)

                # Stochastic Gradient Noise (SGN)
                label_noise = torch.randn(means.size(), device=device)
                means += 0.1*label_noise
                label_noise_scores = torch.randn(scores.size(), device=device)
                scores += 0.1*label_noise_scores

                # Forward
                mean_scores, bias_scores = model(speech_spectrum=wavs, judge_id=judge_ids)
                mean_mom, bias_mom = momentum_model(speech_spectrum=wavs, judge_id=judge_ids)
                mean_scores = mean_scores.squeeze()
                bias_scores = bias_scores.squeeze()
                mean_mom = mean_mom.squeeze() 
                bias_mom = bias_mom.squeeze() 
                seq_len = mean_scores.shape[1]
                bsz = mean_scores.shape[0]
                means = means.unsqueeze(1).repeat(1, seq_len)
                scores = scores.unsqueeze(1).repeat(1, seq_len)

                # Loss
                mean_loss = criterion(mean_scores, means)
                bias_loss = criterion(bias_scores, scores)
                cost_mean = criterion(mean_scores, mean_mom)
                cost_bias = criterion(bias_scores, bias_mom)
                loss = mean_loss + lamb*bias_loss + lamb_c*(cost_mean+cost_bias)
                
                # Backwards
                loss.backward()

                all_loss.append(loss.item())
                mean_losses.append(mean_loss.item())
                bias_losses.append(bias_loss.item())
                del loss

                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                optimizer_momentum.step() 
                optimizer.zero_grad()
    
            except Exception as e:
                print(e)

        if epoch % log_epoch == 0:
            average_loss = torch.FloatTensor(all_loss).mean().item()
            mean_losses = torch.FloatTensor(mean_losses).mean().item()
            bias_losses = torch.FloatTensor(bias_losses).mean().item()
            train_loss.append(average_loss)
            print(f"Average loss={average_loss}")
            print(f"Mean loss={mean_loss}")
            print(f"Bias loss={bias_loss}")
            all_loss = []
            mean_losses = []
            bias_losses = []

        if epoch % log_valid == 0:
            MSE_teacher, LCC_teacher, SRCC_teacher, sys_SRCC = valid(momentum_model,
                                                                     valid_loader,
                                                                     valid_set.systems,
                                                                     epoch,
                                                                     'Valid',
                                                                     device,
                                                                     MSE_teacher,
                                                                     LCC_teacher,
                                                                     SRCC_teacher)

            if LCC_teacher[-1] > best_LCC_teacher:
                best_LCC_teacher = LCC_teacher[-1]
                best_model = copy.deepcopy(momentum_model)

        epoch += 1

    print('Best model performance test:')
    _, _, _, _ = valid(best_model, test_loader, test_set.systems, epoch, 'Test', device, MSE_teacher, LCC_teacher, SRCC_teacher)
    return best_model, train_loss, MSE_list, LCC_list, SRCC_list, LCC_teacher

def main():
    data_path = args.data_path
    id_table = args.id_table
    train_set = get_dataset(data_path,
                            "training_data.csv",
                            vcc18=True,
                            idtable=os.path.join(id_table, 'idtable.pkl'))
    valid_set = get_dataset(data_path,
                            "valid_data.csv",
                            vcc18=True,
                            valid=True,
                            idtable=os.path.join(id_table, 'idtable.pkl'))
    test_set = get_dataset(data_path,
                           "testing_data.csv",
                           vcc18=True,
                           valid=True,
                           idtable=os.path.join(id_table, 'idtable.pkl'))
    train_loader = get_dataloader(train_set, batch_size=64, num_workers=1)
    valid_loader = get_dataloader(valid_set, batch_size=1, num_workers=1)
    test_loader = get_dataloader(test_set, batch_size=1, num_workers=1)
    
    model, train_loss, MSE_list, LCC_list, SRCC_list, LCC_teacher = train(
        args.num_epochs, args.log_valid, args.log_epoch, train_set,
        valid_set, test_set, train_loader, valid_loader, test_loader)

    torch.save(model, args.save_path)

main()
