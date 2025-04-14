import torch
from torch import nn
from torch.utils.data import DataLoader
from data.download_and_prepare import download_and_prepare_tiny_imagenet
from dataset.tiny_imagenet import get_tiny_imagenet_datasets
from models.custom_net import CustomNet
from utils.training import train
from utils.validation import validate

import wandb

wandb.init(project="tiny-imagenet-lab")  # Crea nuovo esperimento su W&B

config = wandb.config
config.learning_rate = 0.0001
config.batch_size = 64
config.epochs = 10
config.model = "CustomNet"


# Setup
download_and_prepare_tiny_imagenet()

train_dataset, val_dataset = get_tiny_imagenet_datasets()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_acc = 0
num_epochs = 10
for epoch in range(1, num_epochs + 1):
     
     train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer)

     val_loss, val_acc = validate(model, val_loader, criterion)

    # loggiamo tutto qui
     wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
     })

     best_acc = max(best_acc, val_acc)

print(f'Best validation accuracy: {best_acc:.2f}%')
torch.save(model.state_dict(), 'checkpoints/customnet_best.pt')
