# -*- coding: utf-8 -*-
import json
import os
import shutil
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'tianchi_SVHN'
RAW_DATA_PATH = DATA_PATH / 'raw'
TRAIN_DIR = DATA_PATH / 'train'
VAL_DIR = DATA_PATH / 'val'
TEST_DIR = DATA_PATH / 'test_a'
TRAIN_JSON_PATH = DATA_PATH / 'train.json'
VAL_JSON_PATH = DATA_PATH / 'val.json'
SUBMIT_TEMPLATE_PATH = DATA_PATH / 'test_A_sample_submit.csv'
MODEL_PATH = BASE_DIR / 'model.pt'

VAL_RATIO = 0.1
PREPARE_SEED = 42


def recreate_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def convert_svhn_label(label):
    label = int(label)
    return 0 if label == 10 else label


def save_svhn_split(dataset, indices, image_dir, json_path=None):
    recreate_dir(image_dir)
    annotations = {}

    for output_index, dataset_index in enumerate(indices):
        image = np.transpose(dataset.data[dataset_index], (1, 2, 0))
        file_name = f'{output_index:05d}.png'
        image_path = image_dir / file_name
        Image.fromarray(image).save(image_path)

        if json_path is not None:
            label = convert_svhn_label(dataset.labels[dataset_index])
            annotations[file_name] = {'label': [label]}

    if json_path is not None:
        with open(json_path, 'w', encoding='utf-8') as file_obj:
            json.dump(annotations, file_obj, ensure_ascii=False, indent=2)


def save_test_split(dataset, image_dir, submit_template_path):
    recreate_dir(image_dir)
    file_names = []

    for output_index in range(len(dataset)):
        image = np.transpose(dataset.data[output_index], (1, 2, 0))
        file_name = f'{output_index:05d}.png'
        image_path = image_dir / file_name
        Image.fromarray(image).save(image_path)
        file_names.append(file_name)

    submit_template = pd.DataFrame({
        'file_name': file_names,
        'file_code': [''] * len(file_names),
    })
    submit_template.to_csv(submit_template_path, index=False)


def prepare_dataset():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.SVHN(
        root=str(RAW_DATA_PATH),
        split='train',
        download=False,
    )
    testset = torchvision.datasets.SVHN(
        root=str(RAW_DATA_PATH),
        split='test',
        download=False,
    )

    indices = np.arange(len(trainset))
    rng = np.random.default_rng(PREPARE_SEED)
    rng.shuffle(indices)

    val_size = int(len(indices) * VAL_RATIO)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    save_svhn_split(trainset, train_indices, TRAIN_DIR, TRAIN_JSON_PATH)
    save_svhn_split(trainset, val_indices, VAL_DIR, VAL_JSON_PATH)
    save_test_split(testset, TEST_DIR, SUBMIT_TEMPLATE_PATH)


def require_path(path):
    if not path.exists():
        raise FileNotFoundError(f'Required path not found: {path}')
    return path


prepare_dataset()
require_path(TRAIN_DIR)
require_path(VAL_DIR)
require_path(TEST_DIR)
require_path(TRAIN_JSON_PATH)
require_path(VAL_JSON_PATH)
require_path(SUBMIT_TEMPLATE_PATH)


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int64)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)


with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as file_obj:
    train_json = json.load(file_obj)
train_file_names = list(train_json.keys())
train_path = [str(TRAIN_DIR / file_name) for file_name in train_file_names]
train_label = [train_json[file_name]['label'] for file_name in train_file_names]
print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=True,
    num_workers=10,
)

with open(VAL_JSON_PATH, 'r', encoding='utf-8') as file_obj:
    val_json = json.load(file_obj)
val_file_names = list(val_json.keys())
val_path = [str(VAL_DIR / file_name) for file_name in val_file_names]
val_label = [val_json[file_name]['label'] for file_name in val_file_names]
print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = []

    for input_tensor, target in train_loader:
        if use_cuda:
            input_tensor = input_tensor.cuda()
            target = target.cuda()

        c0, c1, c2, c3, c4 = model(input_tensor)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3]) + \
               criterion(c4, target[:, 4])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for input_tensor, target in val_loader:
            if use_cuda:
                input_tensor = input_tensor.cuda()
                target = target.cuda()

            c0, c1, c2, c3, c4 = model(input_tensor)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for input_tensor, _ in test_loader:
                if use_cuda:
                    input_tensor = input_tensor.cuda()

                c0, c1, c2, c3, c4 = model(input_tensor)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy(),
                    ], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy(),
                    ], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0

use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()

for epoch in range(10):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_loss = validate(val_loader, model, criterion)

    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x != 10])))

    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

    print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
    print('Val Acc', val_char_acc)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)


df_submit = pd.read_csv(SUBMIT_TEMPLATE_PATH)
test_file_names = df_submit['file_name'].tolist()
test_path = [str(TEST_DIR / file_name) for file_name in test_file_names]
test_label = [[1]] * len(test_path)
print(len(test_path), len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda' if use_cuda else 'cpu'))

test_predict_label = predict(test_loader, model, 1)
print(test_predict_label.shape)

test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x != 10])))

df_submit['file_code'] = test_label_pred
df_submit.to_csv(BASE_DIR / 'submit.csv', index=None)