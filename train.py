import json
from utils import seed_everything
from dataset import ImageFeatureDataset, PairFeatureDataset, TextDataset, FeatureDataset
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader
from models import NeuralNetwork, SAJEM, CustomSelfAttention, BertFinetune, MultiSelfAttention
import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def my_collate_fn(batch):
    batch = filter (lambda x:x is not None, batch)
    return tuple(zip(*batch))
if __name__ == "__main__":
    # Read config file
    with open('options.json', 'r') as f:
        opt = json.load(f)
    
    # Seed everything
    seed_everything(2019)

    # Define tokenizer for text
    if opt['text_model_type'] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(opt['text_model_pretrained'])
    else:
        tokenizer = BertTokenizer.from_pretrained(opt['text_model_pretrained'])
    
    # Define dataset
    image_dataset = ImageFeatureDataset(opt['feature_folder'])
    text_dataset = TextDataset(opt['train_file'], tokenizer, opt['max_seq_len'])
    dataset = PairFeatureDataset(image_dataset, text_dataset)
    dataloader = DataLoader(dataset, batch_size = opt['batch_size'], shuffle=True, collate_fn=my_collate_fn)
    # Dataset for evaluation
    val_image_dataset = FeatureDataset(opt['feature_folder'], opt['val_file'])
    val_image_dataloader = DataLoader(val_image_dataset, batch_size=opt['batch_size'], collate_fn=my_collate_fn)
    val_text_dataset = TextDataset(opt['val_file'], tokenizer, opt['max_seq_len'])
    val_text_dataloader = DataLoader(val_text_dataset, batch_size = opt['batch_size'], shuffle=False)
    #Test dataset
    test_image_dataset = FeatureDataset(opt['feature_folder'], opt['test_file'])
    test_image_dataloader = DataLoader(test_image_dataset, batch_size=opt['batch_size'], collate_fn=my_collate_fn)
    test_text_dataset = TextDataset(opt['test_file'], tokenizer, opt['max_seq_len'])
    test_text_dataloader = DataLoader(test_text_dataset, batch_size = opt['batch_size'], shuffle=False)

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
    
    image_mha = MultiSelfAttention(opt['common_dim'], num_layers=opt['num_attention_layers'], dropout=opt['mha_dropout']).to(device)
    if opt['text_model_type'] == 'roberta':
        bert = RobertaModel.from_pretrained(opt['text_model_pretrained'], output_hidden_states=True).to(device)
    else:
        bert = BertModel.from_pretrained(opt['text_model_pretrained'], output_hidden_states=True).to(device)
    bert_model = BertFinetune(bert, output_type=opt['output_bert_model'])
    model = SAJEM(image_encoder, text_encoder, image_mha, bert_model, optimizer = opt['optimizer'], lr = opt['lr'], l2_regularization = opt['l2_regularization'],
                    max_violation=opt['max_violation'], margin_loss=opt['margin_loss'], use_lr_scheduler=opt['use_lr_scheduler'], grad_clip=opt['grad_clip'],
                    num_training_steps = int((opt['epochs']-1) * len(dataloader)))
    
    # Save folder
    version = opt['version']
    save_folder = './' + version
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    # Tensorboard Summary writer
    logdir = './' + version + '/log'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    writer = SummaryWriter(log_dir=logdir)
    
    # Training loop
    num_steps = 0
    best_recall = 0
    best_epoch = 0
    for epoch in range(1,opt['epochs']+1):
        losses = []
        pbar = tqdm(enumerate(dataloader),total=len(dataloader),leave=False, position=0, file=sys.stdout)
        for i, (image_features, input_ids, attention_mask) in pbar:
            input_ids = torch.stack(input_ids).to(device)
            attention_mask = torch.stack(attention_mask).to(device)
            loss = model.train(image_features, input_ids, attention_mask, epoch)
            num_steps+=1
            writer.add_scalar('Training loss', loss, num_steps)
            for index, param_group in enumerate(model.optimizer.param_groups):
                writer.add_scalar('lr group ' + str(index), param_group['lr'], num_steps)
            losses.append(loss)
            # Step scheduler
            model.step_scheduler()
        epoch_loss = np.mean(losses)
        writer.add_scalar('Epoch training loss', epoch_loss, epoch)

        # Evaluate on validation set
        recall = model.evaluate(val_image_dataloader, val_text_dataloader, opt['k'])
        writer.add_scalar('Recall@' +str(opt['k']) +' on validation set', recall, epoch)
        # Evaluate on test set
        recall_test = model.evaluate(test_image_dataloader, test_text_dataloader, opt['k'])
        writer.add_scalar('Recall@' +str(opt['k']) +' on test set', recall_test, epoch)
        if best_recall <= recall:
            best_recall = recall
            best_epoch = epoch
            model.save_network(save_folder)
            with open(os.path.join(save_folder, 'best.txt'), 'w') as f:
                f.write('Best epoch: {}\n'.format(best_epoch))
                f.write('Best validation recall: {}\n'.format(recall))
                f.write('Best test recall: {}\n'.format(recall_test))
        sys.stdout.flush()
        writer.flush()
        print('Epoch: {}/{}'.format(epoch, opt['epochs']))
        print('Epoch loss: {:.5f}, validation recall: {:.3f}, test recall: {:.3f}'.
            format(epoch_loss, recall, recall_test))