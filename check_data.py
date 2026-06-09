import sys, os
sys.path.insert(0, os.getcwd())
import torch
from data.preprocess import ChatDataset
ds = torch.load('data_cache/train.pt', weights_only=False)
item = ds[0]
print('input_ids:', item['input_ids'][:10])
print('labels:', item['labels'][:10])
print('nan in ids:', torch.isnan(item['input_ids'].float()).any())
print('max label:', item['labels'].max())
print('min label:', item['labels'].min())