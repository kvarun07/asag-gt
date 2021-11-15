import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CoAttention(nn.Module):
    def __init__(self, device, latent_dim = 200):
        super(CoAttention, self).__init__()

        self.latent_dim = latent_dim
        self.k = 30
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))

        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))

        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))

        #register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)


        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))

#     def forward(self, sentence_rep, comment_rep):
        
# #         sentence_rep = sentence_rep.view(sentence_rep.size(0), -1)
# #         comment_rep = comment_rep.view(comment_rep.size(0), -1)
        
#         # sentence and comment rep: 128 x 1
#         sentence_rep = sentence_rep.reshape(sentence_rep.shape[0], sentence_rep.shape[1], 1)
#         comment_rep = comment_rep.reshape(comment_rep.shape[0], comment_rep.shape[1], 1)
        
# #         print(sentence_rep.shape, comment_rep.shape)
        
#         sentence_rep_trans = sentence_rep.transpose(2, 1)
#         comment_rep_trans = comment_rep.transpose(2, 1)
        
# #         sentence_rep_trans = sentence_rep
# #         comment_rep_trans = comment_rep
# #         print(sentence_rep_trans.shape, comment_rep_trans.shape, self.Wl.shape)
        
#         L = torch.tanh(torch.matmul(sentence_rep_trans, torch.matmul(self.Wl, comment_rep)))
#         L_trans = L.transpose(2, 1)
# #         print(L.shape, L_trans.shape)     # L: 1 x 1
# #         L_trans = L
        
#         # Hs: k(80) x 1
#         Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep) + torch.matmul(torch.matmul(self.Wc, comment_rep), L))
        
#         # Hc: k(80) x 1
#         Hc = torch.tanh(torch.matmul(self.Wc, comment_rep)+ torch.matmul(torch.matmul(self.Ws, sentence_rep), L_trans))
        
#         # As, Ac: 1 x 1
#         As = F.softmax(torch.matmul(self.whs, Hs), dim = 2)

#         Ac = F.softmax(torch.matmul(self.whc, Hc), dim = 2)
        
#         As = As.transpose(2,1)
#         Ac = Ac.transpose(2,1)
        
#         # co_c & co_s : 128 x 1
#         co_s = torch.matmul(sentence_rep, As)

#         co_c = torch.matmul(comment_rep, Ac)
        
#         # co_sc: 256 x 1
#         co_sc = torch.cat([co_s, co_c], dim=1)
        
# #         print(co_sc.shape, torch.squeeze(co_sc, -1).shape)
# #         return co_sc.transpose(2,1)
#         return torch.squeeze(co_sc, -1)

    def forward(self, sentence_rep, comment_rep):
        
#         sentence_rep = sentence_rep.view(sentence_rep.size(0), -1)
#         comment_rep = comment_rep.view(comment_rep.size(0), -1)
        
        print('COATTENTION ORIGNAL SHAPE',sentence_rep.shape, comment_rep.shape)
        # sentence and comment rep: 128 x 1
        sentence_rep = sentence_rep.reshape(sentence_rep.shape[0], sentence_rep.shape[1]//4, 4)
        comment_rep = comment_rep.reshape(comment_rep.shape[0], comment_rep.shape[1]//4, 4)
        
        print('COATTENTION RESHAPE',sentence_rep.shape, comment_rep.shape)
        
        sentence_rep_trans = sentence_rep.transpose(2, 1)
        comment_rep_trans = comment_rep.transpose(2, 1)
        
#         sentence_rep_trans = sentence_rep
#         comment_rep_trans = comment_rep
#         print(sentence_rep_trans.shape, comment_rep_trans.shape, self.Wl.shape)
        
        L = torch.tanh(torch.matmul(sentence_rep_trans, torch.matmul(self.Wl, comment_rep)))
        L_trans = L.transpose(2, 1)
        print('L SHAHE and Transpose',L.shape, L_trans.shape)     # L: 1 x 1
        print('Weights',self.Wl.shape,self.Wc.shape,self.Ws.shape,self.whs.shape,self.whc.shape)
#         L_trans = L
        
        # Hs: k(80) x 1
        Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep) + torch.matmul(torch.matmul(self.Wc, comment_rep), L_trans))
        
        # Hc: k(80) x 1
        Hc = torch.tanh(torch.matmul(self.Wc, comment_rep)+ torch.matmul(torch.matmul(self.Ws, sentence_rep), L))
        
        # As, Ac: 1 x 1
        As = F.softmax(torch.matmul(self.whs, Hs), dim = 2)

        Ac = F.softmax(torch.matmul(self.whc, Hc), dim = 2)
        
        As = As.transpose(2,1)
        Ac = Ac.transpose(2,1)
        # print(As)
        
        # co_c & co_s : 128 x 1
        co_s = torch.matmul(sentence_rep, As)

        co_c = torch.matmul(comment_rep, Ac)
        
        # co_sc: 256 x 1
        co_sc = torch.cat([co_s, co_c], dim=1)
        
        print(co_sc.shape, torch.squeeze(co_sc, -1).shape)
#         return co_sc.transpose(2,1)
        return torch.squeeze(co_sc, -1)


