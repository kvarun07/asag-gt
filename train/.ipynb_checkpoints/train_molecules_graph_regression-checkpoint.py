"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE

def train_epoch(model, optimizer, device, data_loader_list, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0

    batch_dict = {'batch_graphs':[], 'batch_x': [], 'batch_e':[], 'batch_targets':[], 'batch_lap_pos_enc':[], 'batch_wl_pos_enc':[], 'batch_sim_scores':[]}
    all_subgraphs = []      # corresponding to 8 dataloaders

    # Iterating over 8 dataloaders (s_arg0, s_arg1, ....)
    for data_loader in data_loader_list:
        temp_batch_dict = {'batch_graphs':[], 'batch_x': [], 'batch_e':[], 'batch_targets':[], 'batch_lap_pos_enc':[], 'batch_wl_pos_enc':[], 'batch_sim_scores':[]}

        for iter, (batch_graphs, batch_sim_scores, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_sim_scores = batch_sim_scores.to(device)
            optimizer.zero_grad()
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
            except:
                batch_lap_pos_enc = None
                
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None

            temp_batch_dict['batch_graphs'].append(batch_graphs)
            temp_batch_dict['batch_x'].append(batch_x)
            temp_batch_dict['batch_e'].append(batch_e)
            temp_batch_dict['batch_targets'].append(batch_targets)
            temp_batch_dict['batch_lap_pos_enc'].append(batch_lap_pos_enc)
            temp_batch_dict['batch_wl_pos_enc'].append(batch_wl_pos_enc)
            temp_batch_dict['batch_sim_scores'].append(batch_sim_scores)

        all_subgraphs.append(temp_batch_dict)
    
    # no. of batches iterations
    for iter, _ in enumerate(data_loader_list[0]):
        
        forward_batch_graphs = []
        forward_batch_x = []
        forward_batch_e = []
        forward_batch_targets = [] 
        forward_batch_lap_pos_enc = []
        forward_batch_wl_pos_enc = []
        forward_batch_sim_scores = []

        for i in all_subgraphs:
            forward_batch_graphs.append(i['batch_graphs'][iter])
            forward_batch_x.append(i['batch_x'][iter])
            forward_batch_e.append(i['batch_e'][iter])
            forward_batch_targets.append(i['batch_targets'][iter])
            forward_batch_lap_pos_enc.append(i['batch_lap_pos_enc'][iter])
            forward_batch_wl_pos_enc.append(i['batch_wl_pos_enc'][iter])
            forward_batch_sim_scores.append(i['batch_sim_scores'][iter])

        batch_scores = model.forward(forward_batch_graphs, forward_batch_x, forward_batch_e, forward_batch_lap_pos_enc, forward_batch_wl_pos_enc, forward_batch_sim_scores[0])
        # print("Train epoch:", batch_scores.size())

        loss = model.loss(batch_scores, forward_batch_targets[0])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, forward_batch_targets[0])
        nb_data += forward_batch_targets[0].size(0)

    # print("NB_DATA = ", nb_data)
    print(" _._")
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network(model, device, data_loader_list, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0

    batch_dict = {'batch_graphs':[], 'batch_x': [], 'batch_e':[], 'batch_targets':[], 'batch_lap_pos_enc':[], 'batch_wl_pos_enc':[], 'batch_sim_scores':[]}
    all_subgraphs = []      # corresponding to 8 dataloaders

    with torch.no_grad():
        # Iterating over 8 dataloaders (s_arg0, s_arg1, ....)
        for data_loader in data_loader_list:
            temp_batch_dict = {'batch_graphs':[], 'batch_x': [], 'batch_e':[], 'batch_targets':[], 'batch_lap_pos_enc':[], 'batch_wl_pos_enc':[], 'batch_sim_scores':[]}

            for iter, (batch_graphs, batch_sim_scores, batch_targets) in enumerate(data_loader):
                batch_graphs = batch_graphs.to(device)
                batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
                batch_e = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                batch_sim_scores = batch_sim_scores.to(device)
                try:
                    batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                    sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                    sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                    batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
                except:
                    batch_lap_pos_enc = None
                    
                try:
                    batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
                except:
                    batch_wl_pos_enc = None

                temp_batch_dict['batch_graphs'].append(batch_graphs)
                temp_batch_dict['batch_x'].append(batch_x)
                temp_batch_dict['batch_e'].append(batch_e)
                temp_batch_dict['batch_targets'].append(batch_targets)
                temp_batch_dict['batch_lap_pos_enc'].append(batch_lap_pos_enc)
                temp_batch_dict['batch_wl_pos_enc'].append(batch_wl_pos_enc)
                temp_batch_dict['batch_sim_scores'].append(batch_sim_scores)

            all_subgraphs.append(temp_batch_dict)
            
        # no. of batches iterations
        for iter, _ in enumerate(data_loader_list[0]):
            
            forward_batch_graphs = []
            forward_batch_x = []
            forward_batch_e = []
            forward_batch_targets = [] 
            forward_batch_lap_pos_enc = []
            forward_batch_wl_pos_enc = []
            forward_batch_sim_scores = []

            for i in all_subgraphs:
                forward_batch_graphs.append(i['batch_graphs'][iter])
                forward_batch_x.append(i['batch_x'][iter])
                forward_batch_e.append(i['batch_e'][iter])
                forward_batch_targets.append(i['batch_targets'][iter])
                forward_batch_lap_pos_enc.append(i['batch_lap_pos_enc'][iter])
                forward_batch_wl_pos_enc.append(i['batch_wl_pos_enc'][iter])
                forward_batch_sim_scores.append(i['batch_sim_scores'][iter])

            batch_scores = model.forward(forward_batch_graphs, forward_batch_x, forward_batch_e, forward_batch_lap_pos_enc, forward_batch_wl_pos_enc, forward_batch_sim_scores[0])
            
            loss = model.loss(batch_scores, forward_batch_targets[0])
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, forward_batch_targets[0])
            nb_data += forward_batch_targets[0].size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae

