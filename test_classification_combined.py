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
class_combined_label = 20
combined_label = [(3,5), (3,20), (4,2), (4,7), (4,11),(5,8), (7,1), (7,20), (8,6), (8,9), (10,20), (11,14), (13,1), (13,6), (13,9), (13,15), (13,16), (13,17), (13,18), (13,20)]

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
model.fc = nn.Linear(num_ftrs, class_combined_label)

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
        pred = torch.softmax(preds, dim=1)
        if i_batch % log_interval == 0:
          print("Test [{}/{}]".format(i_batch * batch_size, len(test_loader.dataset)))
        for i in range(preds.shape[0]):
            pred_num = torch.argmax(pred, dim=1)[i]
            pred_num = pred_num.item()
            (pred_plant_num, pred_disease_num)=combined_label[pred_num]
            tsv_writer.writerow([str(img_idx) + '.jpg', pred_plant_num, pred_disease_num])
            img_idx += 1

f.close()
print("Test finished")
