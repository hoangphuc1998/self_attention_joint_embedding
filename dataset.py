from torch.utils.data import Dataset
import torch
import json
import os
import re

class TextDataset(Dataset):
  def __init__(self, json_file, tokenizer, max_seq_len):
    self.js = json.load(open(json_file))
    self.annotations = self.js['annotations']
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
  def __len__(self):
    return len(self.annotations)
  def __getitem__(self, idx):
    item = self.annotations[idx]
    caption = item['caption']
    tokenizer_res = self.tokenizer.encode_plus(caption, add_special_tokens=True, pad_to_max_length=True, max_length=self.max_seq_len, return_attention_mask=True, return_token_type_ids=False, truncation=True)
    input_ids = torch.tensor([tokenizer_res['input_ids']])
    attention_mask = torch.tensor([tokenizer_res['attention_mask']])
    filename = item['filename']
    return filename, input_ids.squeeze(), attention_mask.squeeze()

class ImageFeatureDataset(Dataset):
  def __init__(self, image_folder, max_num_regions=15):
    self.image_folder = image_folder
    self.max_num_regions = max_num_regions
  def get_by_image_filename(self, filename):
    path = os.path.join(self.image_folder, filename)
    feature = torch.load(path).detach().squeeze(-1).squeeze(-1)
    num_regions = feature.shape[0]
    if self.max_num_regions > num_regions:
      feature_pad = torch.stack((self.max_num_regions - num_regions) * [torch.zeros(feature.shape[-1])]).to('cuda')
      feature = torch.cat([feature, feature_pad], dim=0)
    attention_mask = torch.ones(self.max_num_regions)
    attention_mask[num_regions:self.max_num_regions] = 0
    return feature, attention_mask
    
class PairFeatureDataset(Dataset):
  def __init__(self, image_dataset, text_dataset):
    self.image_dataset = image_dataset
    self.text_dataset = text_dataset
  def __getitem__(self, idx):
    image_file, input_ids, attention_mask = self.text_dataset.__getitem__(idx)
    image, image_attention_mask = self.image_dataset.get_by_image_filename(image_file)
    if image.shape[0]==0:
      return None
    return image, image_attention_mask, input_ids, attention_mask
  def __len__(self):
    return len(self.text_dataset)

class FeatureDataset(Dataset):
  def __init__(self, folder, json_file, max_num_regions=15):
    self.js = json.load(open(json_file))
    self.folder = folder
    self.feature_files = self.js['image_files']
    self.max_num_regions = max_num_regions
  def __getitem__(self, idx):
    file = self.feature_files[idx]
    image_id = torch.tensor(int(re.findall(r'\d{12}', file)[0]))
    feature = torch.load(os.path.join(self.folder, file)).detach().squeeze(-1).squeeze(-1)
    if feature.shape[0]==0:
      return None
    num_regions = feature.shape[0]
    if self.max_num_regions > num_regions:
      feature_pad = torch.stack((self.max_num_regions - num_regions) * [torch.zeros(feature.shape[-1])]).to('cuda')
      feature = torch.cat([feature, feature_pad], dim=0)
    attention_mask = torch.ones(self.max_num_regions)
    attention_mask[num_regions:self.max_num_regions] = 0
    return image_id, feature, attention_mask
  def __len__(self):
    return len(self.feature_files)