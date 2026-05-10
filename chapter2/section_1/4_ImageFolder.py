import random
import shutil
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 分类任务通用的ImageFolder读取形式

source_root = Path('./dataset/PetImages')
train_root = source_root / 'train'
test_root = source_root / 'test'
class_names = ['Cat', 'Dog']
valid_suffixes = {'.jpg', '.jpeg', '.png', '.bmp'}


def split_dataset(train_ratio=0.8, seed=42):
    random.seed(seed)

    for class_name in class_names:
        source_dir = source_root / class_name
        train_dir = train_root / class_name
        test_dir = test_root / class_name

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            file_path for file_path in source_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in valid_suffixes
        ]
        image_files.sort()
        random.shuffle(image_files)

        split_index = int(len(image_files) * train_ratio)
        train_files = image_files[:split_index]
        test_files = image_files[split_index:]

        for file_path in train_files:
            shutil.copy2(file_path, train_dir / file_path.name)

        for file_path in test_files:
            shutil.copy2(file_path, test_dir / file_path.name)

        print(f'{class_name}: train={len(train_files)}, test={len(test_files)}')


def build_dataloader(dataset_root, transform, shuffle):
    dataset = ImageFolder(str(dataset_root), transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=shuffle, num_workers=8)
    return dataset, dataloader


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    split_dataset(train_ratio=0.8, seed=42)

    train_dataset, train_loader = build_dataloader(train_root, train_transform, shuffle=True)
    test_dataset, test_loader = build_dataloader(test_root, test_transform, shuffle=False)

    print('train_dataset size:', len(train_dataset))
    print('test_dataset size:', len(test_dataset))
    print('classes:', train_dataset.classes)



