import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import csv
import numpy as np
from sklearn.metrics import f1_score
import os
from PIL import Image

# constants
class_plants = 14
class_disease = 21

# opts
img_size = 224 #256? 224?
ckp_name = "IN-60_plant_best.pt"
log_interval = 5

try:
  import google.colab
  print("running on colab")
  rootPath = "/content/drive/My Drive/Colab Notebooks"
  batch_size = 100
except:
  print("running on local")
  rootPath = r"C:\Users\jadoh\PycharmProjects\IngGongJiNeung"
  batch_size = 30

class PlantsDataset_test(torch.utils.data.Dataset):
    def __init__(self, split, file_path, root_dir, transform=None):
        # /drive/My\ Drive/Colab\ Notebooks/dataset/train/train.tsv'
        self.dataset = pd.read_csv(os.path.join(root_dir, file_path),
                                   names=['image'])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("using gpu")

test_loader = torch.utils.data.DataLoader(
    PlantsDataset_test('test', "test.tsv", os.path.join(rootPath,'dataset','test'),
                  transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])),
    batch_size=batch_size, shuffle=False)


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, class_plants + class_disease)

if use_gpu:
  model.cuda()

model.load_state_dict(torch.load(os.path.join(rootPath,'run',ckp_name)))
print("Model loaded")
model.eval()

f = open(os.path.join(rootPath, 'test_result', '60_plant_test.tsv'), 'wt')
tsv_writer = csv.writer(f, delimiter='\t')

img_idx = 0

with torch.no_grad():
    for i_batch, data in enumerate(test_loader):
        img = data
        if use_gpu:
            img=img.cuda()
        preds = model(img)
        pred_plant = torch.softmax(preds[:, :class_plants], dim=1)
        pred_disease = torch.softmax(preds[:, class_plants:], dim=1)
        if i_batch % log_interval == 0:
          print("Test [{}/{}]".format(i_batch * batch_size, len(test_loader.dataset)))
        for i in range(preds.shape[0]):
            pred_plant_num = torch.argmax(pred_plant, dim=1)[i]
            pred_plant_num = pred_plant_num.item()
            pred_disease_num = torch.argmax(pred_disease, dim=1)[i]
            pred_disease_num = pred_disease_num.item()
            tsv_writer.writerow([str(img_idx) + '.jpg', pred_plant_num, pred_disease_num])
            img_idx += 1

f.close()
print("Test finished")
