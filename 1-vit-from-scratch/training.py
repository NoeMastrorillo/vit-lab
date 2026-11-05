import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

import time


def train_classif(model, train_loader, val_loader, device, n_epochs=1, log_steps=100, lr=0.001, catch_interrupt=True):
    """ Train the classification model with AdamW optimizer and cross entropy loss.
        Returns the training history.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    running_loss = torch.zeros((), device=device)
    step = 0
    history = []

    start = time.time()
    try:
        for epoch in range(n_epochs):
            model.train()
            n_steps = len(train_loader)

            for i, batch in enumerate(train_loader):
                imgs, labels = batch["img"].to(device), batch["label"].to(device)
                optimizer.zero_grad()

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.detach()
                step += 1
                if step % log_steps == 0:
                    # print(f"epoch {epoch+1} | step {i+1}/{n_steps} | loss: {running_loss.item()/log_steps:.3f}")
                    history.append({
                        "epoch": epoch+1,
                        "step": step,
                        "train_loss": running_loss.item()/log_steps,
                        "time": time.time() - start
                    })
                    print(history[-1])
                    running_loss.zero_()

            val_loss, val_acc = eval_classif(model, val_loader, device)
            # print(f"epoch {epoch+1} | val loss: {val_loss:.3f}")
            history.append({
                "epoch": epoch+1,
                "step": step,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            print(history[-1])
    
    except KeyboardInterrupt:
        if not catch_interrupt:
            raise
        print("Learning interrupted. Returning history.")

    return pd.DataFrame(history)


def eval_classif(model, loader, device):
    """ Evaluate the classification model.
        Returns the mean cross entropy loss and the accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).detach()
            correct += (outputs.argmax(dim=-1) == labels).sum()
            n += imgs.shape[0]
    
    return total_loss.item()/len(loader), correct.item()/n

