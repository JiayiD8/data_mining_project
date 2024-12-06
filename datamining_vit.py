import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, FashionMNIST
from timeit import default_timer as timer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Data
#CIFAR100 96x96 8x8
data_set = CIFAR100(root='.\Datasets\CIFAR100', download=True)
data_path = '.\Datasets\CIFAR100'

RESIZED_IMAGE_SIZE = 96
transform_cifar_train = transforms.Compose([
    transforms.Resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE)),
    transforms.TrivialAugmentWide(num_magnitude_bins=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_cifar_test = transforms.Compose([
    transforms.Resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = CIFAR100(root=data_path, transform=transform_cifar_train, download=True)
test_data = CIFAR100(root=data_path, train=False, transform=transform_cifar_test, download=True)

data_set = CIFAR100(root='.\Datasets\CIFAR100', download=True)
data_path = '.\Datasets\CIFAR100'

# CIFAR100 72x72 12x12
# RESIZED_IMAGE_SIZE = 72
# transform_cifar_train = transforms.Compose([
#     transforms.Resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE)),
#     transforms.TrivialAugmentWide(num_magnitude_bins=10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# transform_cifar_test = transforms.Compose([
#     transforms.Resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_data = CIFAR100(root=data_path, transform=transform_cifar_train, download=True)
# test_data = CIFAR100(root=data_path, train=False, transform=transform_cifar_test, download=True)


# FashionMNIST 36x36 6x6
# data_set = FashionMNIST(root='.\Datasets\FashionMNIST', download=True)
# data_path = '.\Datasets\FashionMNIST'

# RESIZED_IMAGE_SIZE = 36
# transform_fashion_train = transforms.Compose([
#     transforms.Resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE)),
#     transforms.TrivialAugmentWide(num_magnitude_bins=10),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# transform_fashion_test = transforms.Compose([
#     transforms.Resize((RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# train_data = FashionMNIST(root=data_path, transform=transform_fashion_train, download=True)
# test_data = FashionMNIST(root=data_path, train=False, transform=transform_fashion_test, download=True)

BATCH_SIZE = 128
train_data_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, BATCH_SIZE, shuffle=True)

print(f"Total train batch: {len(train_data_loader)}")
print(f"Total test batch: {len(test_data_loader)}")
print(f"Total classes: {len(train_data.classes)}")


# Model
# Defult parameters values based on ViT paper
class PatchEmbedding(nn.Module):
    """Creates a patch embedding."""

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
                                  end_dim=3) # flatten the feature map dimensions into a single vector

    def forward(self, x):

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1) # Make the embedding on the last dimension
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block."""

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
        self.num_heads = num_heads

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block."""

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
    """Creates a Transformer Encoder block."""

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    def forward(self, x):
        x =  self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x

class ViT(nn.Module):
    """Creates a Vision Transformer."""
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0.1,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000):
        super().__init__()

        # Make sure we have a proper patch size
        assert img_size % patch_size == 0, f"Image size not divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        self.num_patches = (img_size * img_size) // patch_size**2

        # Class token based on the ViT paper
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # Positional embedding based on the ViT paper
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            attn_dropout=attn_dropout) for _ in range(num_transformer_layers)])
        # Last MLP Head
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


# Training steps
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Training step for a single epoch. Records the average loss and accuracy for the epoch."""

    # Training mode
    model.train()
    train_loss, train_acc = 0, 0

    # Iterate over the dataloader
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

    # Calculate the average loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Testing steps
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Testing step for a single epoch. Records the average loss and accuracy for the epoch."""

    # Evaluation mode
    model.eval()
    test_loss, test_acc = 0, 0

    # Iterate over the dataloader
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Calculate the average loss and accuracy
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Training for all epochs
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    """Training loop for multiple epochs. 
       Records the loss and accuracy for train and test dataset.
       Records training time for each epoch.
       Returns a dictionary with the results."""

    # Results dictionary
    results = {"times": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Training loop
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

# Seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Hyperparameters
# CIFAR100 96x96 8x8
CLASS = 100
NUM_EPOCHS = 150
CHANNEL = 3
IMAGE_SIZE = RESIZED_IMAGE_SIZE
PATCH_SIZE = 8
NUM_TRANSFORMER_LAYERS = [2, 4]
NUM_HEADS = [4, 8]
EMBEDDING_DIM = [256, 512]
MLP_SIZE = [2048, 3072]
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# CIFAR100 72x72 12x12
# CLASS = 100
# NUM_EPOCHS = 150
# CHANNEL = 3
# IMAGE_SIZE = RESIZED_IMAGE_SIZE
# PATCH_SIZE = 12
# NUM_TRANSFORMER_LAYERS = [2, 4]
# NUM_HEADS = [4, 8]
# EMBEDDING_DIM = [256, 512]
# MLP_SIZE = [2048, 3072]
# LEARNING_RATE = 0.0001
# WEIGHT_DECAY = 0.0001

# FashionMNIST 36x36 6x6
# CLASS = 10
# NUM_EPOCHS = 100
# CHANNEL = 1
# IMAGE_SIZE = RESIZED_IMAGE_SIZE
# PATCH_SIZE = 6
# NUM_TRANSFORMER_LAYERS = [2, 4]
# NUM_HEADS = [4, 8]
# EMBEDDING_DIM = [128, 256]
# MLP_SIZE = [1024, 2048]
# LEARNING_RATE = 0.0001
# WEIGHT_DECAY = 0.0001

# CUDA setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Train the model
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
                          mlp_size=mlp_size,
                          num_heads=num_heads,
                          num_classes=CLASS).to(device)

              loss_fn = nn.CrossEntropyLoss()
              optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

              results = train(model,
                              train_data_loader,
                              test_data_loader,
                              optimizer,
                              loss_fn,
                              epochs=NUM_EPOCHS,
                              device=device)
              all_results[f'model_{num_transformer_layers}_{num_heads}_{embedding_dim}_{mlp_size}'] = results
              torch.save(model.state_dict(), f'model_weights_{num_transformer_layers}_{num_heads}_{embedding_dim}_{mlp_size}.pth')

# Save results to csv files
# CIFAR100 96x96 8x8
pd.DataFrame(all_results).T.to_csv('results_cifar100_96_8.csv')

# CIFAR100 72x72 12x12
# pd.DataFrame(all_results).T.to_csv('results_cifar100.csv')

# FashionMNIST
# pd.DataFrame(all_results).T.to_csv('results_FashionMNIST.csv')


