import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class SimulatedDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(f'{img_path}')
        img = self.transform(image=img)
        return {'path': img_path, 'img': img['image']}
