import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
def train_classifier(
        model,
        dataloader,
        optimizer,
        epochs,
        n_classes,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    model.to(device)
    model.train()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    correct = 0.0
    ctr = 0
    for epoch in tqdm(range(epochs)):

        epoch_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            x,label = batch

            #Extract Image
            x = x.squeeze().to(device)
            label = label.to(device)
            # Extract Onehot Vectors
            y = F.one_hot(label,num_classes=n_classes).float().to(device)
            # Input to the network
            yhat = model(x)
            
            # Evaluate loss function
            loss = loss_fn(yhat,y)

            
            label_pred = torch.sigmoid(yhat).argmax(dim=-1)
            # Evaluate the score
            correct += (label_pred == label).sum().item()
            ctr+=label_pred.size(0)
            # Update
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            

        print(f'Epoch{epoch}:',epoch_loss/len(dataloader))
        print(f'Score{epoch}:',correct/ctr)
        
    