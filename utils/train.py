# Autora: Giulia Katherine Romas Tomazeli

import torch
import time
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import numpy as np

from utils.plots import (plot_training_history, plot_f1_history)
from utils.metrics import calculate_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, class_names, freeze_backbone=False):
    """
    Executa uma época de treinamento.
    """
    model.train()

    # mantem batchnorm congelado
    if freeze_backbone:
        freeze_batchnorm(model)

    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Treino", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(y_true, y_pred, class_names)

    return epoch_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device, class_names):
    """
    Executa a validação do modelo.
    """
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Validação", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
       
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(y_true, y_pred, class_names)

    return epoch_loss, metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, early_stopping, device, num_epochs, class_names, logger, freeze_backbone=False):
    """
    Treinamento com validação.
    Utilizado nos folds do K-fold espacial.
    """

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": [], "val_accuracy": [], "val_precision_macro": [], "val_recall_macro": []}

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # -=-=-=-=-=- treinamento -=-=-=-=-=- #
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            class_names=class_names,
            freeze_backbone=freeze_backbone
        )

        # -=-=-=-=-=- validação -=-=-=-=-=- #
        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )

        val_f1 = val_metrics["f1_macro"]

        current_lr = optimizer.param_groups[0]["lr"]

        # -=-=-=-=-=- histórico -=-=-=-=-=- #
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_metrics["f1_macro"])
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])

        # -=-=-=-=-=- resultados -=-=-=-=-=- #
        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1-macro: {train_metrics['f1_macro']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1-macro: {val_f1:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Precision Macro: {val_metrics['precision_macro']:.4f} | "
            f"Val Recall Macro: {val_metrics['recall_macro']:.4f}" 
        )

        # -=-=-=-=-=- early stopping -=-=-=-=-=- #
        early_stopping(val_loss, model)

        remaining = (early_stopping.patience - early_stopping.counter)
        logger.info(f"Early stopping - patience: {remaining}")

        if early_stopping.should_stop:
            logger.info("Early stopping.")
            break

    return history

def train_final_model(model, train_loader, criterion, optimizer, device, num_epochs, class_names, model_name, checkpoint_dir, results_dir, logger, freeze_backbone=False):
    """
    Treinamento final utilizando todo o conjunto de treino.
    O número de épocas é definido previamente com base nos resultados do K-fold.
    """

    history = {"train_loss": [], "train_f1": [], "learning_rate": []}

    # -=-=-=-=-=- diretorios -=-=-=-=-=- #
    checkpoint_dir = Path(checkpoint_dir) / model_name
    results_dir = Path(results_dir) / model_name / "final_model"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (checkpoint_dir / "final_model.pth")

    # -=-=-=-=-=- cronometro -=-=-=-=-=- #
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # -=-=-=-=-=- treinamento -=-=-=-=-=- #
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            class_names=class_names,
            freeze_backbone=freeze_backbone
        )

        train_f1 = train_metrics["f1_macro"]
        current_lr = optimizer.param_groups[0]["lr"]

        # -=-=-=-=-=- historico -=-=-=-=-=- #

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["learning_rate"].append(current_lr)

        # -=-=-=-=-=- resultados -=-=-=-=-=- #
        logger.info(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} ")

    # -=-=-=-=-=- tempo -=-=-=-=-=- #
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    training_time = (time.perf_counter() - start_time)

    # -=-=-=-=-=- salvar modelo -=-=-=-=-=- #
    torch.save(model.state_dict(), checkpoint_path)

    # -=-=-=-=-=- graficos -=-=-=-=-=- #
    plot_training_history(history, results_dir / "loss.png", False)
    plot_f1_history(history, results_dir / "f1.png", False)

    # -=-=-=-=-=- resumo -=-=-=-=-=- #
    logger.info("=-" * 30)
    logger.info("Treinamento final concluído!")
    logger.info("=-" * 30)

    logger.info(f"Tempo de treinamento: {training_time / 60:.2f} minutos")
    logger.info(f"Modelo salvo em: {checkpoint_path}")
    logger.info(f"Resultados salvos em: {results_dir}")

    return history, training_time

def freeze_batchnorm(model):
    """
    Mantém as camadas BatchNorm em modo de avaliação.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def calculate_class_weights(dataset, num_classes):
    """
    Calcula pesos das classes 
    """
    labels = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        labels.append(item["label"])

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)

    # peso inversamente proporcional a frequência
    weights = len(labels) / (num_classes * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32)

    return weights, class_counts

