from pathlib import Path
import torch

#  -=-=-=-=-=-=- caminhos -=-=-=-=-=-=- #
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "dataset"
TRAIN_DIR = DATA_DIR / "treino"
TEST_DIR = DATA_DIR / "teste"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

#  -=-=-=-=-=-=- dataset -=-=-=-=-=-=- #
IMAGE_SIZE = 128
NUM_CLASSES = 4

CLASS_NAMES = ["Preservada", "Moderada", "Baixa", "Não Preservada"]

BATCH_SIZE = 32
NUM_WORKERS = 0

SEED = 42

# sem augmentation inicialmente
USE_AUGMENTATION = False

#  -=-=-=-=-=-=- normalizacao - ImageNet -=-=-=-=-=-=- #
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

#  -=-=-=-=-=-=- treinamento -=-=-=-=-=-=- #
NUM_EPOCHS = 50

OPTIMIZER = "AdamW"

#  -=-=-=-=-=-=- early stopping -=-=-=-=-=-=- #
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

MONITOR_METRIC = "loss"

#  -=-=-=-=-=-=- transfer learning -=-=-=-=-=-=- #
PRETRAINED = True
FREEZE_BACKBONE = True

#  -=-=-=-=-=-=- device -=-=-=-=-=-=- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")