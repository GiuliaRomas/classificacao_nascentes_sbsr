# Autora: Giulia Katherine Romas Tomazeli

from pathlib import Path
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)

def calculate_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    for class_name in class_names:
        metrics[f"precision_{class_name}"] = report[class_name]["precision"]
        metrics[f"recall_{class_name}"] = report[class_name]["recall"]
        metrics[f"f1_{class_name}"] = report[class_name]["f1-score"]
        metrics[f"support_{class_name}"] = report[class_name]["support"]

    return metrics

def print_metrics(metrics, class_names):
    """
    Imprime as métricas gerais e as métricas por classe.
    """
    print("\n" + "=-" * 30)
    print("RESULTADOS")
    print("=-" * 30)

    print(f"Acurácia:        {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro:    {metrics['recall_macro']:.4f}")
    print(f"F1-macro:        {metrics['f1_macro']:.4f}")

    print("\nMétricas por classe:")
    print(
        f"{'Classe':<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}")

    print("-" * 60)

    for class_name in class_names:
        print(
            f"{class_name:<12}"
            f"{metrics[f'precision_{class_name}']:>12.4f}"
            f"{metrics[f'recall_{class_name}']:>12.4f}"
            f"{metrics[f'f1_{class_name}']:>12.4f}"
            f"{metrics[f'support_{class_name}']:>12.0f}"
        )

    print("=-" * 30)

def save_metrics(metrics, class_names, save_path):
    """
    Salva as métricas em um arquivo CSV.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # -=-=-=-=- metricas -=-=-=-=- #
    rows.append({
        "classe": "Geral",
        "precision": metrics["precision_macro"],
        "recall": metrics["recall_macro"],
        "f1": metrics["f1_macro"],
        "support": sum(metrics[f"support_{class_name}"] for class_name in class_names),
        "accuracy": metrics["accuracy"]
    })

    # -=-=-=-=- metricas por classe -=-=-=-=- #
    for class_name in class_names:
        rows.append({
            "classe": class_name,
            "precision": metrics[f"precision_{class_name}"],
            "recall": metrics[f"recall_{class_name}"],
            "f1": metrics[f"f1_{class_name}"],
            "support": metrics[f"support_{class_name}"],
            "accuracy": None
        })

    df = pd.DataFrame(rows)

    df.to_csv(save_path, index=False, encoding="utf-8-sig")