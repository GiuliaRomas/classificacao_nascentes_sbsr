# Autora: Giulia Katherine Romas Tomazeli

import optuna
import torch
import torch.nn as nn

from torch.optim import AdamW

from models.final_model import create_model
from utils.train import calculate_class_weights

def objective(trial, train_loader, val_loader, train_subset, device, num_classes, class_names):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)

    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])

    model = create_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    model = model.to(device)

    # pesos das classes
    class_weights, _ = calculate_class_weights(train_subset, num_classes)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")

    patience = 5
    patience_counter = 0

    num_epochs = 30

    for epoch in range(num_epochs):
        # treino
        model.train()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        # validacao
        model.eval()

        running_val_loss = 0.0

        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                running_val_loss += (loss.item() * images.size(0))
                predictions = outputs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        val_loss = (running_val_loss / len(val_loader.dataset))

        # pruning
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # melhor resultado
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= patience:
            break

    return best_val_loss