import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from transformers import *
import torch
import os
import re
import pickle
import time
import json
import sys
from utils import get_top_k_eval, l2norm
from losses import MarginRankingLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def print_debug(tensor_map):
    for key, value in tensor_map.items():
        print(key + ': ')
        print(value)
class NeuralNetwork(nn.Module):
    '''
    Neural network with custom hidden layers
    '''
    def __init__(self, input_dim, output_dim, hidden_units, hidden_activation='relu', output_activation='relu', use_dropout = False, use_batchnorm=False):
        super().__init__()
        self.network = nn.Sequential()
        hidden_units = [input_dim] + hidden_units
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        
        for i in range(len(hidden_units) - 1):
            self.network.add_module("dense_" + str(i), nn.Linear(hidden_units[i], hidden_units[i+1]))
            # Hidden activation
            if hidden_activation == 'relu':
              self.network.add_module("activation_" + str(i), nn.ReLU())
            elif hidden_activation == 'sigmoid':
              self.network.add_module("activation_" + str(i), nn.Sigmoid())
            elif hidden_activation == 'tanh':
              self.network.add_module("activation_" + str(i), nn.Tanh())
            elif hidden_activation == 'lrelu':
              self.network.add_module("activation_" + str(i), nn.LeakyReLU())
            elif hidden_activation == 'prelu':
              self.network.add_module("activation_" + str(i), nn.PReLU())
            # Batchnorm on hidden layers
            if self.use_batchnorm:
              self.network.add_module("batchnorm_" + str(i), nn.BatchNorm1d(hidden_units[i+1]))
        
        # Dropout with 20% probability
        if self.use_dropout:
          self.network.add_module("dropout", nn.Dropout(0.2))

        self.network.add_module("output", nn.Linear(hidden_units[-1], output_dim))
        # Output activation
        # if output_activation == 'relu':
        #   self.network.add_module("activation_out", nn.ReLU())
        # elif output_activation == 'sigmoid':
        #   self.network.add_module("activation_out", nn.Sigmoid())
        # elif output_activation == 'tanh':
        #   self.network.add_module("activation_out", nn.Tanh())

    def forward(self, x):
        return self.network(x)

class CustomSelfAttention(nn.Module):
    '''
    Custom self attention module (inspired from Multi-Head Self Attention)
    '''
    def __init__(self, embed_dim, bias = True, dropout = 0):
        super().__init__()
        self.bias = bias
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.output_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([embed_dim])
        self.init_weights()
    def forward(self, image_features, attention_mask):
        '''
        image_features  --region features (B, N, D)
        attention_mask  --mask of ones and zeros indicates which regions are attended, avoid attending to zeros padding regions (B, N)
        '''
        query = self.query_proj(image_features) # (B, N, D)
        key = self.key_proj(image_features)
        value = self.value_proj(image_features)
        scores = query.bmm(key.permute(0,2,1))
        attn_weights = F.softmax(scores, dim=-1) # (B, N, N)
        attn_weights = torch.mul(attn_weights, attention_mask.unsqueeze(1))
        attn_output = attn_weights.bmm(value)
        attn_output = self.output_proj(attn_output)
        attn_output = self.output_dropout(attn_output)
        residual = self.layer_norm(image_features + attn_output)
        #output = residual.mean(dim=0, keepdim=True)
        
        return residual
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.bias:
            nn.init.constant_(self.key_proj.bias, 0.0)
            nn.init.constant_(self.query_proj.bias, 0.0)
            nn.init.constant_(self.value_proj.bias, 0.0)
            nn.init.constant_(self.output_proj.bias, 0.0)

class MultiSelfAttention(nn.Module):
    """
    Multi layer self attention module
    """
    def __init__(self, embed_dim, num_layers=2, bias=True, dropout=0):
        super().__init__()
        blocks = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            blocks.append(CustomSelfAttention(embed_dim, bias, dropout))
        self.attn_modules = nn.ModuleList(blocks)
    def forward(self, x, attention_mask):
        eps = 1e-9
        for attn_module in self.attn_modules:
            x = attn_module(x, attention_mask)
        x = torch.mul(x, attention_mask.unsqueeze(-1))
        output = torch.div(x.sum(dim=1, keepdim=False), attention_mask.sum(dim=1, keepdim=True) + eps)
        return output

class BertFinetune(nn.Module):
    def __init__(self, bert_model, output_type='cls'):
        super().__init__()
        self.bert_model = bert_model
        self.output_type = output_type
        #self.dropout = nn.Dropout(0.2)
    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask = attention_mask)
        if self.output_type == 'mean':
            feature = (output[0] * attention_mask.unsqueeze(2)).sum(dim=1).div(attention_mask.sum(dim=1, keepdim=True))
        elif self.output_type == 'cls2':
            feature = torch.cat((output[2][-1][:,0,...], output[2][-2][:,0,...]), -1)
        elif self.output_type == 'cls4':
            feature = torch.cat((output[2][-1][:,0,...], output[2][-2][:,0,...], output[2][-3][:,0,...], output[2][-4][:,0,...]), -1)
        else:
            feature = output[2][-1][:,0,...]
        return feature

class SAJEM():
    '''
    Self-Attention based Joint Embedding Model
    Consist of 2 branches to encode image and text
    '''
    def __init__(self, image_encoder, text_encoder, image_mha, bert_model, optimizer = 'adam', lr = 1e-3, l2_regularization=1e-2, margin_loss = 1e-2,
               max_violation=True, cost_style='mean', use_lr_scheduler=False, grad_clip=0, num_training_steps = 30000, device='cuda'):
        self.image_mha = image_mha
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.bert_model = bert_model
        self.device = device

        self.use_lr_scheduler = use_lr_scheduler
        self.params = []
        self.params = list(self.image_mha.parameters())
        self.params += list(self.text_encoder.parameters())
        self.params += list(self.image_encoder.parameters())
        self.params += list(self.bert_model.parameters())
        self.grad_clip = grad_clip
        self.frozen = False
        if optimizer == 'adamW':
            self.optimizer = AdamW([{'params':list(self.bert_model.parameters()),'lr':3e-5},
                                {'params':list(self.image_encoder.parameters()) + list(self.text_encoder.parameters()) + list(self.image_mha.parameters()),'lr':1e-4}])
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam([{'params':list(self.bert_model.parameters()),'lr':3e-5},
                                {'params':list(self.image_encoder.parameters()) + list(self.text_encoder.parameters()) + list(self.image_mha.parameters()),'lr':1e-4}])
        
        if self.use_lr_scheduler:
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
        self.lr_scheduler_0 = get_constant_schedule(self.optimizer)
        # loss
        self.mrl_loss = MarginRankingLoss(margin=margin_loss, max_violation=max_violation, cost_style=cost_style, direction='bidir')
    
    def forward(self, image_feature, image_attention_mask, input_ids, attention_mask, epoch):
        if epoch > 1 and self.frozen:
            self.frozen = False
            del self.lr_scheduler_0
            torch.cuda.empty_cache()
    
        image_feature = l2norm(image_feature).detach()
        final_image_features = l2norm(self.image_mha(image_feature, image_attention_mask))
        text_feature = self.bert_model(input_ids, attention_mask=attention_mask)
        text_feature = l2norm(text_feature)
        if epoch == 1:
            text_feature = text_feature.detach()
            self.frozen = True
        image_to_common = self.image_encoder(final_image_features)
        text_to_common = self.text_encoder(text_feature)
        return image_to_common, text_to_common
    def save_network(self, folder):
        torch.save(self.image_mha.state_dict(), os.path.join(folder, 'image_mha.pth'))
        torch.save(self.text_encoder.state_dict(), os.path.join(folder, 'text_encoder.pth'))
        torch.save(self.image_encoder.state_dict(), os.path.join(folder, 'image_encoder.pth'))
        torch.save(self.bert_model.state_dict(), os.path.join(folder, 'bert_model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(folder, 'optimizer.pth'))
        if self.use_lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(folder, 'scheduler.pth'))
    def switch_to_train(self):
        self.image_mha.train()
        self.text_encoder.train()
        self.image_encoder.train()
        self.bert_model.train()
    def switch_to_eval(self):
        self.image_mha.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()
        self.bert_model.eval()

    def train(self, image_features, image_attention_mask, input_ids, attention_mask, epoch):
        self.switch_to_train()
        image_to_common, text_to_common = self.forward(image_features, image_attention_mask, input_ids, attention_mask, epoch)
        self.optimizer.zero_grad()

        # Compute loss
        loss = self.mrl_loss(text_to_common, image_to_common)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()
        return loss.item()
  
    def step_scheduler(self):
        if self.use_lr_scheduler and not self.frozen:
            self.lr_scheduler.step()
        else:
            self.lr_scheduler_0.step()

    def evaluate(self,val_image_dataloader, val_text_dataloader, k):
        self.switch_to_eval()
        # Load image features
        with torch.no_grad():
            image_features = []
            image_ids = []
            for ids, features, image_attention_mask in val_image_dataloader:
                image_ids.append(torch.stack(ids))
                features = torch.stack(features).to(self.device)
                image_attention_mask = torch.stack(image_attention_mask).to(self.device)
                features = l2norm(features).detach()
                mha_features = l2norm(self.image_mha(features, image_attention_mask))
                image_features.append(self.image_encoder(mha_features))
            image_features = torch.cat(image_features, dim=0)
            image_ids = torch.cat(image_ids, dim=0).to(self.device)
            # Evaluate
            recall = 0
            total_query = 0
            pbar = tqdm(enumerate(val_text_dataloader),total=len(val_text_dataloader),leave=False, position=0, file=sys.stdout)
            for i, (image_files, input_ids, attention_mask) in pbar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                text_features = self.bert_model(input_ids, attention_mask=attention_mask)
                text_features = l2norm(text_features)
                text_features = self.text_encoder(text_features)
                image_files = torch.tensor(list(map(lambda x: int(re.findall(r'\d{12}', x)[0]), image_files))).to(device)
                top_k = get_top_k_eval(text_features, image_features, k)
                for idx, indices in enumerate(top_k):
                    total_query+=1
                    true_image_id = image_files[idx]
                    top_k_images = torch.gather(image_ids, 0, indices)
                    if (top_k_images==true_image_id).nonzero().numel()>0:
                        recall += 1
            recall = recall / total_query
            return recall