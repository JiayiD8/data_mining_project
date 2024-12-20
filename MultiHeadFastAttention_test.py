# -*- coding: utf-8 -*-
"""DataMining Performer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z28M55m27ZNyZ03BpdZgR3aeoSO3uG-D
"""

import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100, FashionMNIST
from timeit import default_timer as timer

import torch
from performer_pytorch import FastAttention, SelfAttention
from model.performer import MultiHeadFAVORAttention
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# queries / keys / values with heads already split and transposed to first dimension
# 8 heads, dimension of head is 64, sequence length of 512
q = torch.randn(1, 4, 512, 64)
k = torch.randn(1, 4, 512, 64)
v = torch.randn(1, 4, 512, 64)

attn_fn = FastAttention(
    dim_heads = 64,
    nb_features = 256,
    causal = False
)

out = attn_fn(q, k, v) # (1, 8, 512, 64)
# now merge heads and combine outputs with Wo

out.shape

"""# Data"""

BATCH_SIZE = 128
PATCH_SIZE = 16

data_set = CIFAR100(root='.\Datasets\CIFAR100', download=True)
data_path = '.\Datasets\CIFAR100'

# data_set = FashionMNIST(root='.\Datasets\FashionMNIST', download=True)
# data_path = '.\Datasets\FashionMNIST'

transform_cifar = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.TrivialAugmentWide(num_magnitude_bins=10),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transform_fashion = transforms.Compose([
#     transforms.Resize((36, 36)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

train_data = CIFAR100(root=data_path, transform=transform_cifar, download=True)
test_data = CIFAR100(root=data_path, train=False, transform=transform_cifar, download=True)

# train_data = FashionMNIST(root=data_path, transform=transform_fashion, download=True)
# test_data = FashionMNIST(root=data_path, train=False, transform=transform_fashion, download=True)

# Use sampler to train and test on a tenth of the data
# train_sampler = SubsetRandomSampler(indices=[i for i in range(len(train_data)) if (i % 10) == 0])
# test_sampler = SubsetRandomSampler(indices=[i for i in range(len(test_data)) if (i % 10) == 0])

# train_data_loader = DataLoader(train_data, BATCH_SIZE, sampler=train_sampler)
# test_data_loader = DataLoader(test_data, BATCH_SIZE, sampler=test_sampler)

# Uncomment to train on the whole dataset
train_data_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, BATCH_SIZE, shuffle=True)

print(f"Total train batch: {len(train_data_loader)}")
print(f"Total test batch: {len(test_data_loader)}")
print(f"Total classes: {len(train_data.classes)}")

train_batch = next(iter(train_data_loader))
print(len(train_batch))
print(train_batch[0].shape)
print(train_batch[1].shape)

single_image = train_batch[0][0]
target = train_batch[1][0]
CHANNEL, HEIGHT, WIDTH = single_image.shape
EMBEDDING_SIZE = PATCH_SIZE**2 * 3

"""# Model"""

class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2, 
                                  end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1) 

# Use FastAttention
class MultiheadFastAttentionBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.attention = FastAttention(
                            dim_heads = 8,
                            nb_features = 256,
                            causal = False)

        self.num_heads = num_heads

    def forward(self, x):
        x = self.layer_norm(x)
        print(x.shape)
        out = torch.stack([x for _ in range(self.num_heads)], dim=1)
        print(out.shape)
        attn_output = self.attention(out, out, out)
        return attn_output

    # def forward(self, x):
    #     x = self.layer_norm(x)
    #     print(x.shape)
    #     out = torch.stack([x for _ in range(self.num_heads)], dim=1)
    #     print(out.shape)
    #     # attn_output, _ = self.multihead_attn(query=x,
    #     #                                      key=x,
    #     #                                      value=x,
    #     #                                      need_weights=False)
    #     attn_output, _ = self.multihead_attn(q=x,
    #                                   k=x,
    #                                   v=x)
        return attn_output

# Use MultiHeadFAVORAttention
# class MultiheadFastAttentionBlock(nn.Module):

#     def __init__(self,
#                  embedding_dim:int=768,
#                  num_heads:int=12,
#                  nb_random_features:int=256,
#                  attn_dropout:float=0,
#                  device:torch.device=None):
#         super().__init__()

#         self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

#         self.multihead_attn = MultiHeadFAVORAttention(head_num=num_heads,
#                                                     dim=embedding_dim,
#                                                     nb_random_features=nb_random_features,
#                                                     use_relu_kernel=True,
#                                                     device=device)

#         self.num_heads = num_heads
#     def forward(self, x):
#         x = self.layer_norm(x)
#         attn_output = self.multihead_attn(query=x,
#                                              key=x,
#                                              value=x)
#         return attn_output
    
class MLPBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, 
                      out_features=embedding_dim), 
            nn.Dropout(p=dropout) 
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768, 
                 num_heads:int=12, 
                 nb_random_features:int=256,
                 mlp_size:int=3072, 
                 mlp_dropout:float=0.1, 
                 device:torch.device=None): 
        super().__init__()

        self.mhfa_block = MultiheadFastAttentionBlock(num_heads=num_heads,
                                                  embedding_dim=embedding_dim,
                                                  nb_random_features=nb_random_features,
                                                  device=device)

        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    def forward(self, x):
        x =  self.mhfa_block(x) + x
        x = self.mlp_block(x) + x

        return x

class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 nb_random_features:int=256,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000,
                 device:torch.device=None):
        super().__init__()

        self.num_patches = (img_size * img_size) // patch_size**2

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            nb_random_features=nb_random_features,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            device=device) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x

# model = ViT(num_classes=10, num_heads=8, img_size=72, patch_size=PATCH_SIZE)
# print(f"Input image shape: {single_image.shape}")
# output = model(single_image.unsqueeze(dim=0))
# print(f"Output shape: {output.shape}")

"""# Training"""

from typing import Tuple, Dict, List
from tqdm.auto import tqdm
import torch

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

  model.train()
  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  model.eval()
  test_loss, test_acc = 0, 0

  with torch.inference_mode():
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(device), y.to(device)
          test_pred_logits = model(X)

          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):

  results = {"times": [],
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
      start_time = timer()
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      end_time = timer()


      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"time: {end_time-start_time:.3f}s"
      )

      results["times"].append(end_time-start_time)
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  return results

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Hyperparameters
NUM_EPOCHS = 100
IMAGE_SIZE = 72
PATCH_SIZE = 12
NUM_TRANSFORMER_LAYERS = [4]
NUM_HEADS = [8]
EMBEDDING_DIM = [512]
MLP_SIZE = [3072]
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
# CUDA setup
device = "cuda" if torch.cuda.is_available() else "cpu"

all_results = {}
for num_transformer_layers in NUM_TRANSFORMER_LAYERS:
    for num_heads in NUM_HEADS:
        for embedding_dim in EMBEDDING_DIM:
            for mlp_size in MLP_SIZE:
              model = ViT(img_size=IMAGE_SIZE,
                          in_channels=CHANNEL,
                          patch_size=PATCH_SIZE,
                          num_transformer_layers=num_transformer_layers,
                          embedding_dim=embedding_dim,
                          nb_random_features=16,
                          mlp_size=mlp_size,
                          num_heads=num_heads,
                          num_classes=100,
                          device=device).to(device)

              loss_fn = nn.CrossEntropyLoss()
              optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

              results = train(model,
                              train_data_loader,
                              test_data_loader,
                              optimizer,
                              loss_fn,
                              epochs=NUM_EPOCHS,
                              device=device)
              all_results[f'model_performer_{num_transformer_layers}_{num_heads}_{embedding_dim}_{mlp_size}'] = results
              torch.save(model.state_dict(), f'model_performer_weights_{num_transformer_layers}_{num_heads}_{embedding_dim}_{mlp_size}.pth')
pd.DataFrame(all_results).T.to_csv('results_performer.csv')

