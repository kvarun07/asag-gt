"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
import math
from layers.mlp_readout_layer import MLPReadout
from coattention import *

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.molecules_graph_regression.load_net import gnn_model 
from data.data import LoadData 


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
ENSEMBLE CLASS
"""
class MyEnsemble(nn.Module):
    def __init__(self, net_params, model_list, MODEL_NAME):
        super(MyEnsemble, self).__init__()

        self.out_dim = net_params['out_dim']
        self.device = net_params['device']    
        self.l = 16 #number of perspectives
        self.dropout = nn.Dropout(0.2)
        
        self.GT1 = gnn_model(MODEL_NAME, net_params).to(self.device)
        self.GT2 = gnn_model(MODEL_NAME, net_params).to(self.device)
        self.GT3 = gnn_model(MODEL_NAME, net_params).to(self.device)
        self.GT4 = gnn_model(MODEL_NAME, net_params).to(self.device)
        
        #multi perspective matching weight initialization starts       
        
        self.mp_w = []
        for i in range(4): # 4 weights corrsponding to each subgraph
            self.mp_w.append(nn.Parameter(torch.rand(self.l, self.out_dim)).to(self.device))
                   
        #multi perspective matching weight initialization ends       
        
        self.MLP_layer_1 = MLPReadout(4*self.l, 1) # #subgraphs * #of perspective 
        
        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None, sim_scores=None):

        g_out = []
            
        iter = 0
        temp = self.GT1(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT2(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT3(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT4(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT1(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT2(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT3(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
        
        iter += 1
        temp = self.GT4(g[iter], h[iter], e[iter], h_lap_pos_enc[iter], h_wl_pos_enc[iter])
        temp = temp.view(temp.size(0), -1) # Linearize for the FC
        g_out.append(temp)
                
        
        g_student = torch.cat(tuple(g_out[:4]), dim=1)
        g_model = torch.cat(tuple(g_out[4:]), dim=1)
        
        # print(final.shape)

        # ----- Matching Layer -----
        def mp_matching_func(v1, v2, w):
            m = []
            for i in range(self.l):
                # v1: (batch, hidden_size)
                # v2: (batch, hidden_size)
                # w: (1, hidden_size)
                # -> (batch, 1)
                
                m.append(F.cosine_similarity(w[i].view(1, -1) * v1, w[i].view(1, -1) * v2, dim=1))
                
            # list of (batch, 1) -> (batch, l)           
            m = torch.stack(m, dim=1)   
            return m
           
        step = 4
        graph_list = []
        for i in range(len(g_out)//2):
            # v1 = v2 = (batch, hidden_size), w = (l, hidden_size)  -> (batch, l)           
            temp = mp_matching_func(g_out[i], g_out[i+step], self.mp_w[i])
            graph_list.append(temp)

        # list of (batch, l) -> (batch, l, 4)           
        graph_list = torch.stack(graph_list, dim=1)     
        # list of (batch, l, 4) -> (batch, l*4)
        graph_list = graph_list.flatten(start_dim=1)
        # (batch, l*4) -> batch
        final = self.MLP_layer_1(graph_list) 
        return final
        

    def loss(self, scores, targets):
        mse_loss = nn.MSELoss()(scores, targets)
        eps = 1e-6
        rmse_loss = torch.sqrt(mse_loss + eps)
        return rmse_loss

"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):

    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if net_params['full_graph']:
        st = time.time()
        print("[!] Converting the given graphs to full graphs..")
        dataset._make_full_graph()
        print('Time taken to convert to full graphs:',time.time()-st)

    if net_params['self_loop']:
        st = time.time()
        print("[!] Adding self loops to graphs..")
        dataset._add_self_loops()
        print('Time taken to add self loops:',time.time()-st)  
    
    if net_params['lap_pos_enc']:
        st = time.time()
        print("[!] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        print('Time LapPE:',time.time()-st)
        
    if net_params['wl_pos_enc']:
        st = time.time()
        print("[!] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()
        print('Time WL PE:',time.time()-st)  
        
    trainset, valset, testset = dataset.train_s_arg0, dataset.val_s_arg0, dataset.test_s_arg0
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model_list = [ gnn_model(MODEL_NAME, net_params) for i in range(8) ]
    # print("MODEL LIST = ", model_list)
    for i in model_list:
        i.to(device)
    # model = gnn_model(MODEL_NAME, net_params)
    # model = model.to(device)
    model = MyEnsemble(net_params, model_list, MODEL_NAME)
    model.to(device)
    print("Ensemble model PARAMS = ", model.parameters)

#     optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    optimizer = optim.RMSprop(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
        

    # import train and evaluate functions
    from train.train_molecules_graph_regression import train_epoch, evaluate_network

    # DATA LOADERS (STUDENT, MODEL)
    # Train Loader
    train_loader_s_arg0 = DataLoader(dataset.train_s_arg0, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    train_loader_s_arg1 = DataLoader(dataset.train_s_arg1, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    train_loader_s_def = DataLoader(dataset.train_s_def, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    train_loader_s_rest = DataLoader(dataset.train_s_rest, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    train_loader_m_arg0 = DataLoader(dataset.train_m_arg0, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    train_loader_m_arg1 = DataLoader(dataset.train_m_arg1, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    train_loader_m_def = DataLoader(dataset.train_m_def, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    train_loader_m_rest = DataLoader(dataset.train_m_rest, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
    train_loader_list = []
    train_loader_list.append(train_loader_s_arg0)
    train_loader_list.append(train_loader_s_arg1)
    train_loader_list.append(train_loader_s_def)
    train_loader_list.append(train_loader_s_rest)
    train_loader_list.append(train_loader_m_arg0)
    train_loader_list.append(train_loader_m_arg1)
    train_loader_list.append(train_loader_m_def)
    train_loader_list.append(train_loader_m_rest)

    # Val Loader
    val_loader_s_arg0 = DataLoader(dataset.val_s_arg0, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    val_loader_s_arg1 = DataLoader(dataset.val_s_arg1, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    val_loader_s_def = DataLoader(dataset.val_s_def, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    val_loader_s_rest = DataLoader(dataset.val_s_rest, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    val_loader_m_arg0 = DataLoader(dataset.val_m_arg0, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    val_loader_m_arg1 = DataLoader(dataset.val_m_arg1, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    val_loader_m_def = DataLoader(dataset.val_m_def, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    val_loader_m_rest = DataLoader(dataset.val_m_rest, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    val_loader_list = []
    val_loader_list.append(val_loader_s_arg0)
    val_loader_list.append(val_loader_s_arg1)
    val_loader_list.append(val_loader_s_def)
    val_loader_list.append(val_loader_s_rest)
    val_loader_list.append(val_loader_m_arg0)
    val_loader_list.append(val_loader_m_arg1)
    val_loader_list.append(val_loader_m_def)
    val_loader_list.append(val_loader_m_rest)

    # Test Loader
    test_loader_s_arg0 = DataLoader(dataset.test_s_arg0, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader_s_arg1 = DataLoader(dataset.test_s_arg1, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader_s_def = DataLoader(dataset.test_s_def, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader_s_rest = DataLoader(dataset.test_s_rest, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    test_loader_m_arg0 = DataLoader(dataset.test_m_arg0, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader_m_arg1 = DataLoader(dataset.test_m_arg1, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader_m_def = DataLoader(dataset.test_m_def, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader_m_rest = DataLoader(dataset.test_m_rest, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    test_loader_list = []
    test_loader_list.append(test_loader_s_arg0)
    test_loader_list.append(test_loader_s_arg1)
    test_loader_list.append(test_loader_s_def)
    test_loader_list.append(test_loader_s_rest)
    test_loader_list.append(test_loader_m_arg0)
    test_loader_list.append(test_loader_m_arg1)
    test_loader_list.append(test_loader_m_def)
    test_loader_list.append(test_loader_m_rest)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader_list, epoch)
                    
                epoch_val_loss, epoch_val_mae = evaluate_network(model, device, val_loader_list, epoch)
                epoch_test_loss, epoch_test_mae = evaluate_network(model, device, test_loader_list, epoch)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_test_losses.append(epoch_test_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('test/_loss', epoch_test_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                print(f'\ntest_MAE={epoch_test_mae:.3f}, test_loss={epoch_test_loss:.3f}, train_MAE={epoch_train_mae:.3f}, train_loss={epoch_train_loss:.3f}, val_MAE={epoch_val_mae:.3f}, val_loss={epoch_val_loss:.3f}')

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_mae = evaluate_network(model, device, test_loader_list, epoch)
    _, train_mae = evaluate_network(model, device, train_loader_list, epoch)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        

def main():    
    """
        USER CONTROLS
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--lap_pos_enc', help="Please give a value for lap_pos_enc")
    parser.add_argument('--wl_pos_enc', help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    print("DATASET_NAME", DATASET_NAME)
    dataset = LoadData(DATASET_NAME)
    print(dataset)

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
        print("args.edge_feat = ", args.edge_feat)
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.lap_pos_enc is not None:
        net_params['lap_pos_enc'] = True if args.lap_pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.wl_pos_enc is not None:
        net_params['wl_pos_enc'] = True if args.wl_pos_enc=='True' else False
        
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    print("\nTRAINING MODEL")
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


##### RUN #####    
main()    
















