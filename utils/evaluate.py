# Autora: Giulia Katherine Romas Tomazeli

import torch
import numpy as np

from utils.metrics import calculate_metrics

@torch.no_grad()
def predict(model, loader, device):
    """
    Gera previsões para todas as imagens do DataLoader.
    """
    model.eval()

    y_true = []
    y_pred = []
    fids = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        outputs = model(images)

        predictions = outputs.argmax(dim=1).cpu()

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())
        fids.extend(batch["fid"])

    return (np.array(y_true), np.array(y_pred), fids)


def evaluate_model(model, loader, device, class_names):
    """
    Avalia o modelo em um conjunto de dados.
    """
    y_true, y_pred, fids = predict(model=model, loader=loader, device=device)
    metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)

    return metrics, y_true, y_pred, fids