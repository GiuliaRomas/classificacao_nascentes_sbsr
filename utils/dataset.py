# Autora: Giulia Katherine Romas Tomazeli

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class NascentesDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, image_size, mean, std, augmentation=False):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),

                transforms.RandomChoice([
                    transforms.Lambda(lambda img: img),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]),

                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name

            for image_path in sorted(class_dir.glob("*.png")):
                samples.append({"path": image_path, "label": label, "fid": image_path.stem})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)

        return {"image": image, "label": sample["label"], "idx": idx, "fid": str(sample["fid"]), "path": str(sample["path"])}

def samples_to_dataframe(dataset):
    return pd.DataFrame(dataset.samples)