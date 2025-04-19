
from torch.utils.data import Dataset

class ImageConditionDataset(Dataset):
    def __init__(self, images, conditions):
        self.images = images
        self.conditions = conditions.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "condition": self.conditions[idx]
        }
