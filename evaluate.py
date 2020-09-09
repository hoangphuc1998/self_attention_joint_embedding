import json
from models import NeuralNetwork, CustomSelfAttention, BertFinetune
import torch
from utils import l2norm, get_top_k_eval
from tqdm import tqdm
import re
import numpy as np
import sys
import os
from dataset import TextDataset, FeatureDataset
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate_t2i(image_mha, image_encoder, bert_model, text_encoder, image_dataloader, text_dataloader, ks):
    # Load image features
    with torch.no_grad():
        image_features = []
        image_ids = []
        for ids, features in image_dataloader:
            image_ids.append(torch.stack(ids))
            mha_features = []
            for feature in features:
                feature = l2norm(feature.to(device))
                feature = l2norm(image_mha(feature))
                feature = torch.mean(feature, dim=0, keepdim=True)
                mha_features.append(feature)
            mha_features = torch.cat(mha_features, dim=0)
            image_features.append(image_encoder(mha_features))
        image_features = torch.cat(image_features, dim=0)
        image_ids = torch.cat(image_ids, dim=0).to(device)
        # Evaluate
        max_k = max(ks)
        recall = np.zeros(len(ks))
        total_query = 0
        pbar = tqdm(enumerate(text_dataloader),total=len(text_dataloader),leave=False, position=0, file=sys.stdout)
        for i, (image_files, input_ids, attention_mask) in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            text_features = bert_model(input_ids, attention_mask=attention_mask)
            text_features = l2norm(text_features)
            text_features = text_encoder(text_features)
            image_files = torch.tensor(list(map(lambda x: int(re.findall(r'\d{12}', x)[0]), image_files))).to(device)
            top_k = get_top_k_eval(text_features, image_features, max_k)
            for idx, indices in enumerate(top_k):
                total_query+=1
                true_image_id = image_files[idx]
                sorted_image_ids = torch.gather(image_ids, 0, indices)
                for i, k in enumerate(ks):
                    top_k_images = sorted_image_ids[:k]
                    if (top_k_images==true_image_id).nonzero().numel()>0:
                        recall[i] += 1
        recall = recall / total_query
        return recall

def evaluate_i2t(image_mha, image_encoder, bert_model, text_encoder, image_dataloader, text_dataloader, ks):
    with torch.no_grad():
        all_text_features = []
        text_index = 0
        res_dict = dict()
        for filenames, input_ids, attention_masks in text_dataloader:
            for filename in filenames:
                image_id = int(re.findall(r'\d{12}', filename)[0])
                if image_id not in res_dict:
                    res_dict[image_id] = []
                res_dict[image_id].append(text_index)
                text_index+=1
            # Get text features
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            text_features = bert_model(input_ids, attention_mask=attention_masks)
            text_features = l2norm(text_features)
            text_features = text_encoder(text_features)
            all_text_features.append(text_features)
        all_text_features = torch.cat(all_text_features, dim=0)
        recall = np.zeros(len(ks))
        max_k = max(ks)
        total_query = 0
        pbar = tqdm(enumerate(image_dataloader),total=len(image_dataloader),leave=False, position=0, file=sys.stdout)
        for i, (image_ids, features) in pbar:
            mha_features = []
            for feature in features:
                feature = l2norm(feature.to(device))
                feature = l2norm(image_mha(feature))
                feature = torch.mean(feature, dim=0, keepdim=True)
                mha_features.append(feature)
            mha_features = torch.cat(mha_features, dim=0)
            image_features = image_encoder(mha_features)
            all_indices = get_top_k_eval(image_features, all_text_features, max_k)
            for idx, indices in enumerate(all_indices):
                total_query+=1
                image_id = image_ids[idx].item()
                true_text_indices = torch.tensor(res_dict[image_id])
                
                for i, k in enumerate(ks):
                    top_k_text = indices[:k].to('cpu')
                    relevant_text = np.intersect1d(top_k_text, true_text_indices)
                    if relevant_text.shape[0] > 0:
                        recall[i] += 1
        recall = recall / total_query
    return recall


if __name__ == "__main__":
    # Read config file
    with open('options.json', 'r') as f:
        opt = json.load(f)
    
    # Define model
    text_encoder = NeuralNetwork(input_dim=opt['text_dim'], 
                              output_dim=opt['image_dim'], 
                              hidden_units=opt['text_encoder_hidden'], 
                              hidden_activation=opt['text_encoder_hidden_activation'], 
                              output_activation=opt['text_encoder_output_activation'],
                              use_dropout=opt['use_dropout'],
                              use_batchnorm=opt['use_batchnorm']).to(device)

    image_encoder = NeuralNetwork(input_dim=opt['image_dim'], 
                                output_dim=opt['common_dim'], 
                                hidden_units=opt['image_encoder_hidden'], 
                                hidden_activation=opt['image_encoder_hidden_activation'], 
                                output_activation=opt['image_encoder_output_activation'],
                                use_dropout=opt['use_dropout'],
                                use_batchnorm=opt['use_batchnorm']).to(device)
    
    image_mha = CustomSelfAttention(opt['common_dim'], dropout=opt['mha_dropout']).to(device)

    # Define tokenizer for text
    if opt['text_model_type'] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(opt['text_model_pretrained'])
    else:
        tokenizer = BertTokenizer.from_pretrained(opt['text_model_pretrained'])
    # Define text model
    if opt['text_model_type'] == 'roberta':
        bert = RobertaModel.from_pretrained(opt['text_model_pretrained'], output_hidden_states=True).to(device)
    else:
        bert = BertModel.from_pretrained(opt['text_model_pretrained'], output_hidden_states=True).to(device)
    bert_model = BertFinetune(bert, output_type=opt['output_bert_model'])
    # Load model
    version = opt['version']
    folder = './' + version
    image_encoder.load_state_dict(torch.load(os.path.join(folder, 'image_encoder.pth'), map_location=device))
    text_encoder.load_state_dict(torch.load(os.path.join(folder, 'text_encoder.pth'), map_location=device))
    bert_model.load_state_dict(torch.load(os.path.join(folder, 'bert_model.pth'), map_location=device))
    image_mha.load_state_dict(torch.load(os.path.join(folder, 'image_mha.pth'), map_location=device))
    image_encoder.eval()
    text_encoder.eval()
    bert_model.eval()
    image_mha.eval()
    image_mha = None
    # Define dataloader
    def my_collate_fn(batch):
        return tuple(zip(*batch))
    image_folder = opt['feature_folder']
    test_image_dataset = FeatureDataset(image_folder, opt['test_file'])
    test_image_dataloader = DataLoader(test_image_dataset, batch_size=64, shuffle=False, collate_fn=my_collate_fn)
    test_text_dataset = TextDataset(opt['test_file'], tokenizer, opt['max_seq_len'])
    test_text_dataloader = DataLoader(test_text_dataset, batch_size = 32, shuffle=False)

    # Print result
    ks = [1,5,10]
    t2i_recall = evaluate_t2i(image_mha, image_encoder, bert_model, text_encoder, test_image_dataloader, test_text_dataloader, ks)
    print('Text to Image: R@1: {}, R@5: {}, R@10: {}'.format(t2i_recall[0], t2i_recall[1], t2i_recall[2]))

    i2t_recall = evaluate_i2t(image_mha, image_encoder, bert_model, text_encoder, test_image_dataloader, test_text_dataloader, ks)
    print('Image to Text: R@1: {}, R@5: {}, R@10: {}'.format(i2t_recall[0], i2t_recall[1], i2t_recall[2]))

    for i in range(5):
        test_image_dataset = FeatureDataset(image_folder, os.path.splitext(opt['test_file'])[0] + '_fold_' + str(i) + '.json')
        test_image_dataloader = DataLoader(test_image_dataset, batch_size=64, shuffle=False, collate_fn=my_collate_fn)
        test_text_dataset = TextDataset(os.path.splitext(opt['test_file'])[0] + '_fold_' + str(i) + '.json', tokenizer, opt['max_seq_len'])
        test_text_dataloader = DataLoader(test_text_dataset, batch_size = 32, shuffle=False)

        # Print result
        ks = [1,5,10]
        print("Fold " + str(i))
        t2i_recall = evaluate_t2i(image_mha, image_encoder, bert_model, text_encoder, test_image_dataloader, test_text_dataloader, ks)
        print('Text to Image: R@1: {}, R@5: {}, R@10: {}'.format(t2i_recall[0], t2i_recall[1], t2i_recall[2]))

        i2t_recall = evaluate_i2t(image_mha, image_encoder, bert_model, text_encoder, test_image_dataloader, test_text_dataloader, ks)
        print('Image to Text: R@1: {}, R@5: {}, R@10: {}'.format(i2t_recall[0], i2t_recall[1], i2t_recall[2]))
