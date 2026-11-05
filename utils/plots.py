# Autora: Giulia Katherine Romas Tomazeli

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import torch
import shutil

# -=-=-=-=- configuracao visual -=-=-=-=- #
sns.set_theme(style="white", context="notebook", font_scale=1.1)

def _remove_top_right_spines(ax):
    """Remove as bordas superior e direita do gráfico."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -=-=-=-=- loss -=-=-=-=- #
def plot_training_history(history, save_path, val=True):
    """
    Plota a loss de treino e validação por época.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=epochs, y=history["train_loss"], linewidth=2, label="Treino", ax=ax)

    if val:
        sns.lineplot(x=epochs, y=history["val_loss"], linewidth=2, label="Validação", ax=ax)

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss durante o treinamento", pad=12)

    _remove_top_right_spines(ax)
    ax.legend(frameon=False)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# -=-=-=-=- f1 -=-=-=-=- #
def plot_f1_history(history, save_path, val=True):
    """
    Plota o F1-macro por época.
    """
    if val:
        epochs = range(1, len(history["val_f1"]) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=epochs, y=history["val_f1"], linewidth=2, ax=ax)

        ax.set_xlabel("Época")
        ax.set_ylabel("F1-macro")
        ax.set_title("F1-macro de validação", pad=12)
    else:
        epochs = range(1, len(history["train_f1"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=epochs, y=history["train_f1"], linewidth=2, ax=ax)

        ax.set_xlabel("Época")
        ax.set_ylabel("F1-macro")
        ax.set_title("F1-macro de treinamento", pad=12)

    _remove_top_right_spines(ax)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# -=-=-=-=- matriz de confusao -=-=-=-=- #
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plota a matriz de confusão na ordem definida em class_names.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],  # Preservada, Moderada, Baixa, Não Preservada
        display_labels=class_names,
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=True
    )

    ax.set_title("Matriz de Confusão", pad=12)
    ax.set_xlabel("Rótulo Predito")
    ax.set_ylabel("Rótulo Verdadeiro")
    
    _remove_top_right_spines(ax)

    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -=-=-=-=-=-=-=- salva predicoes erradas -=-=-=-=-=-=-=- #
def save_misclassified_images(model, dataloader, device, class_names, output_dir):
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            fids = batch["fid"]
            paths = batch["path"]

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            predictions = outputs.argmax(dim=1)
            confidences = probabilities.max(dim=1).values

            for i in range(len(labels)):
                true_idx = labels[i].item()
                pred_idx = predictions[i].item()

                if true_idx != pred_idx:
                    fid = fids[i]
                    image_path = paths[i]

                    true_label = class_names[true_idx]
                    pred_label = class_names[pred_idx]

                    filename = (f"{fid}__real_{true_label}__pred_{pred_label}.png")
                    shutil.copy(image_path, output_dir / filename)

                    errors.append({
                        "fid": fid,
                        "path": str(image_path),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": confidences[i].item()
                    })

    return pd.DataFrame(errors)