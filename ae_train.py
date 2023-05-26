import torch
from torch.nn import L1Loss
from torch.optim import Adam
from imageData import Data
from torch.utils.data import DataLoader
from models import AE
from hypParam import *
from statistics import mean
from torchsummary import summary

model = AE().to(device)
summary(model, (3, 256, 256))
loss_fn = L1Loss()
optim = Adam(model.parameters(), ae_lr)

train_data = Data(noised_data_path, orig_data_path)
val_data = Data(noised_data_path, orig_data_path, training = False)

data_loader = DataLoader(train_data, ae_batch_size, shuffle=True, pin_memory=True, num_workers=no_workers)
val_loader = DataLoader(val_data, ae_batch_size, shuffle=True, pin_memory=True, num_workers=no_workers)

for epoch in range(1, ae_no_epochs+1):
    losses = []
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        y_pred = torch.squeeze(model(X))
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    with torch.no_grad():
        model.eval()
        val_losses = []
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            y_val_pred = torch.squeeze(model(X))
            val_losses.append(loss_fn(y_val_pred, y).item())
        model.train()

    print(f"Epoch: {epoch},  train loss: {round(mean(losses), 4)}, val loss: {round(mean(val_losses), 4)}")

torch.save(model.state_dict(), './parameters/model_param.pt')