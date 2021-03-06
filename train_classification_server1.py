import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
from PIL import Image
import neptune

# constants
class_combined_label = 20
num_train = 16000
combined_label = [(3,5), (3,20), (4,2), (4,7), (4,11),(5,8), (7,1), (7,20), (8,6), (8,9), (10,20), (11,14), (13,1), (13,6), (13,9), (13,15), (13,16), (13,17), (13,18), (13,20)]

# opts
img_size = 224 #256? 224?
log_interval = 10
epochs = 130
validation_ratio = 0.1

try:
  import google.colab
  print("running on colab")
  rootPath = "/content/drive/My Drive/Colab Notebooks"
  batch_size = 100
except:
  print("running on local")
  rootPath = r"/mnt/DB1/JDH/IngGongJiNeung"
  batch_size = 100

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_f1_score(preds, labels):
  preds = preds.data.cpu().numpy()
  labels = labels.data.cpu().numpy()
  batchsize = preds.shape[0]
  prediction = np.argmax(preds, axis=1)
  f1 = f1_score(prediction, labels, average = 'macro')
  accuracy = np.sum(prediction == labels) / batch_size
  return f1, accuracy

class PlantsDataset(torch.utils.data.Dataset):
  def __init__(self, split, file_path, root_dir, transform=None):
    #/drive/My\ Drive/Colab\ Notebooks/dataset/train/train.tsv'
    self.dataset = pd.read_csv(os.path.join(root_dir, file_path), delimiter='\t', names=['image','plant','disease'])
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    img_path = os.path.join(self.root_dir, self.dataset.iloc[idx,0])
    plant_class = self.dataset.iloc[idx,1]
    disease_class = self.dataset.iloc[idx,2]
    for i in range(len(combined_label)):
      if (plant_class, disease_class) == combined_label[i]:
        combined_class = i
    img = Image.open(img_path).convert('RGB')
    img = self.transform(img)
    return img, combined_class

def train(train_loader, model, criterion, optimizer, epoch, use_gpu, log_interval, total_num):
  losses = AverageMeter()
  acces = AverageMeter()
  f1s = AverageMeter()

  model.train()

  for i_batch, data in enumerate(train_loader):
    img, combined = data
    if use_gpu:
      img, combined = img.cuda(), combined.cuda()
    img, combined = Variable(img), Variable(combined)

    optimizer.zero_grad()

    preds = model(img)
    pred_combined = torch.softmax(preds, dim = 1)

    loss = criterion(pred_combined, combined)

    loss.backward()
    optimizer.step()

    f1, acc = get_f1_score(pred_combined.detach(), combined)

    losses.update(loss.data.cpu(), batch_size)
    acces.update(acc, batch_size)
    f1s.update(f1, batch_size)

    if i_batch % log_interval == 0:
      print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) f1:{:.3f}({:.3f}) Accuracy:{:.2f} ({:.2f})'.format(epoch, i_batch * batch_size, total_num, losses.val,losses.avg, f1s.val, f1s.avg, acces.val,acces.avg))
      neptune.log_metric("train_losses_avg", losses.avg)
      neptune.log_metric("train_f1", f1s.avg)
  return losses.avg, f1s

def validate(val_loader, model, epoch, use_gpu, log_interval, total_num):
  acces = AverageMeter()
  f1s = AverageMeter()

  model.eval()

  with torch.no_grad():
    for i_batch, data in enumerate(val_loader):
      img, combined = data
      if use_gpu:
        img, combined = img.cuda(), combined.cuda()
      img, combined = Variable(img), Variable(combined)

      preds = model(img)
      pred = torch.softmax(preds, dim = 1)

      f1, acc = get_f1_score(pred.detach(), combined)

      acces.update(acc, batch_size)
      f1s.update(f1, batch_size)

      if i_batch == 15:
        print('Validate Epoch:{} [{}/{}] f1:{:.3f}({:.3f}) Accuracy:{:.2f} ({:.2f})'.format(epoch, i_batch * batch_size, total_num, f1s.val, f1s.avg,  acces.val,acces.avg))
        neptune.log_metric("val_f1", f1s.avg)
  return f1s.avg


neptune.init('jadohu/IngGongJiNeung', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiM2JiNmYzMzQtMDYxYS00ZGVhLTk4NmMtZDY3YjA0NTc2NDAxIn0=')
neptune.create_experiment(name='Ing')
exp_id = neptune.get_experiment()._id

use_gpu = torch.cuda.is_available()
if use_gpu:
  print("using gpu")

indices = np.array(list(range(num_train)))
# np.random.shuffle(indices)
split = int(np.floor(validation_ratio*num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(
    PlantsDataset('train', "train.tsv", os.path.join(rootPath, 'dataset', 'train'),
                  transform=transforms.Compose([
                                                transforms.RandomResizedCrop(img_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(30),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
                  batch_size = batch_size, sampler = train_sampler)

val_loader = torch.utils.data.DataLoader(
    PlantsDataset('val', "train.tsv", os.path.join(rootPath,'dataset','train'),
                  transform=transforms.Compose([
                                                transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
                  batch_size = batch_size, sampler = valid_sampler)


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, class_combined_label)

model = nn.DataParallel(model)

if use_gpu:
  model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50,70,90], gamma=0.3)

best=0
second=0

for epoch in range(epochs):
  scheduler.step()
  print("starting training epoch: ", epoch)
  loss, f1 = train(train_loader, model, criterion, optimizer, epoch, use_gpu, log_interval, len(train_idx))
  f1_val = validate(val_loader, model, epoch, use_gpu, log_interval, len(valid_idx))
  print("finished training epoch: ", epoch)

  if f1_val > best:
    file_name = os.path.join(rootPath, 'run', exp_id + "_plant_best.pt")
    sec_file_name = os.path.join(rootPath, 'run', exp_id + "_plant_second.pt")
    if best != 0:
      os.rename(file_name,sec_file_name)
    torch.save(model.state_dict(), file_name)
    print("Model saved (best). f1:{:.3f}".format(f1_val))
  elif f1_val>second:
    file_name = os.path.join(rootPath, 'run', exp_id + "_plant_second.pt")
    torch.save(model.state_dict(), file_name)
    second=f1_val
    print("Model saved (second). f1:{:.3f}".format(f1_val))
  best = max(f1_val, best)

print("training finished")
