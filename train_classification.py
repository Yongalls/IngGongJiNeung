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
    img = Image.open(img_path).convert('RGB')
    img = self.transform(img)
    return img, plant_class, disease_class

class PlantsDataset_test(torch.utils.data.Dataset):
    def __init__(self, split, file_path, root_dir, transform=None):
        # /drive/My\ Drive/Colab\ Notebooks/dataset/train/train.tsv'
        self.dataset = pd.read_csv(os.path.join(root_dir, file_path),
                                   names=['image'])
        # fix later to be distributed
        # if split == 'train':
        #   self.dataset = dataset.iloc[:12800]
        # else:
        #   self.dataset = dataset.iloc[12800:]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

def train(train_loader, model, criterion, optimizer, epoch, use_gpu, log_interval, total_num):
  losses = AverageMeter()
  acces_plant = AverageMeter()
  acces_disease = AverageMeter()
  f1s_plant = AverageMeter()
  f1s_disease = AverageMeter()

  model.train()

  for i_batch, data in enumerate(train_loader):
    img, plant, disease = data
    if use_gpu:
      img, plant, disease = img.cuda(), plant.cuda(), disease.cuda()
    img, plant, disease = Variable(img), Variable(plant), Variable(disease)

    optimizer.zero_grad()

    preds = model(img)
    pred_plant = torch.softmax(preds[:,:class_plants], dim = 1)
    pred_disease = torch.softmax(preds[:, class_plants:], dim = 1)

    loss_plant = criterion(pred_plant, plant)
    loss_disease = criterion(pred_disease, disease)
    loss = loss_plant + loss_disease

    loss.backward()
    optimizer.step()

    f1_plant, acc_plant = get_f1_score(pred_plant.detach(), plant)
    f1_disease, acc_disease = get_f1_score(pred_disease.detach(), disease)

    losses.update(loss.data.cpu(), batch_size)
    acces_plant.update(acc_plant, batch_size)
    acces_disease.update(acc_disease, batch_size)
    f1s_plant.update(f1_plant, batch_size)
    f1s_disease.update(f1_disease, batch_size)

    if i_batch % log_interval == 0:
      print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) f1_plant:{:.3f}({:.3f}) f1_disease:{:.3f}({:.3f}) Accuracy_plant:{:.2f} ({:.2f}) Accuracy_disease:{:.2f} ({:.2f})'.format(epoch, i_batch * batch_size, total_num, losses.val,losses.avg, f1s_plant.val, f1s_plant.avg, f1s_disease.val, f1s_disease.avg, acces_plant.val,acces_plant.avg, acces_disease.val,acces_disease.avg))
      neptune.log_metric("train_losses_avg", losses.avg)
      neptune.log_metric("train_f1_plant", f1s_plant.avg)
      neptune.log_metric("train_f1_disease", f1s_disease.avg)
  return losses.avg, f1s_plant.avg, f1s_disease.avg

def validate(val_loader, model, epoch, use_gpu, log_interval, total_num):
  acces_plant = AverageMeter()
  acces_disease = AverageMeter()
  f1s_plant = AverageMeter()
  f1s_disease = AverageMeter()

  model.eval()

  with torch.no_grad():
    for i_batch, data in enumerate(val_loader):
      img, plant, disease = data
      if use_gpu:
        img, plant, disease = img.cuda(), plant.cuda(), disease.cuda()
      img, plant, disease = Variable(img), Variable(plant), Variable(disease)

      preds = model(img)
      pred_plant = torch.softmax(preds[:,:class_plants], dim = 1)
      pred_disease = torch.softmax(preds[:, class_plants:], dim = 1)

      f1_plant, acc_plant = get_f1_score(pred_plant.detach(), plant)
      f1_disease, acc_disease = get_f1_score(pred_disease.detach(), disease)

      acces_plant.update(acc_plant, batch_size)
      acces_disease.update(acc_disease, batch_size)
      f1s_plant.update(f1_plant, batch_size)
      f1s_disease.update(f1_disease, batch_size)

      if i_batch % log_interval == 0:
        print('Validate Epoch:{} [{}/{}] f1_plant:{:.3f}({:.3f}) f1_disease:{:.3f}({:.3f}) Accuracy_plant:{:.2f} ({:.2f}) Accuracy_disease:{:.2f} ({:.2f})'.format(epoch, i_batch * batch_size, total_num, f1s_plant.val, f1s_plant.avg, f1s_disease.val, f1s_disease.avg, acces_plant.val,acces_plant.avg, acces_disease.val,acces_disease.avg))
        neptune.log_metric("val_f1_plant", f1s_plant.avg)
        neptune.log_metric("val_f1_disease", f1s_disease.avg)
  return f1s_plant.avg, f1s_disease.avg

def test(test_loader, model):

    model.eval()
    answer_list=[]
    f = open("answer.csv", "w")

    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            img = data
            if use_gpu:
                img=img.cuda()
            preds = model(img)
            pred_plant = torch.softmax(preds[:, :class_plants], dim=1)
            pred_disease = torch.softmax(preds[:, class_plants:], dim=1)
            for i in range(preds.shape[0]):
                pred_plant_num = torch.argmax(pred_plant, dim=1)[i]
                pred_plant_num = pred_plant_num.item()
                pred_disease_num = torch.argmax(pred_disease, dim=1)[i]
                pred_disease_num = pred_disease_num.item()
                answer=str(i_batch*batch_size+i)+'.jpg'+'_'+str(pred_plant_num)+'_'+str(pred_disease_num)
                answer_list.append(answer)
    for i in range(len(answer_list)):
        f.write(answer_list[i]+'\n')
    f.close()

    return len(answer_list)

neptune.init('jadohu/IngGongJiNeung', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiM2JiNmYzMzQtMDYxYS00ZGVhLTk4NmMtZDY3YjA0NTc2NDAxIn0=')
neptune.create_experiment(name='Ing')

use_gpu = torch.cuda.is_available()
if use_gpu:
  print("using gpu")

# constants
class_plants = 14
class_disease = 21

# opts
img_size = 224 #256? 224?
batch_size = 100
log_interval = 10
epochs = 10

validation_ratio = 0.2

num_train = 16000;
indices = np.array(list(range(num_train)))
np.random.shuffle(indices)
split = int(np.floor(validation_ratio*num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(
    PlantsDataset('train', "train.tsv", "/content/drive/My Drive/Colab Notebooks/dataset/train",
                  transform=transforms.Compose([
                                                transforms.RandomResizedCrop(img_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(30),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
                  batch_size = batch_size, sampler = train_sampler)

val_loader = torch.utils.data.DataLoader(
    PlantsDataset('val', "train.tsv", "/content/drive/My Drive/Colab Notebooks/dataset/train",
                  transform=transforms.Compose([
                                                transforms.CenterCrop(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
                  batch_size = batch_size, sampler = valid_sampler)

test_loader = torch.utils.data.DataLoader(
    PlantsDataset_test('test', "test.tsv", r"C:\Users\jadoh\PycharmProjects\IngGongJiNeung\Dataset\test",
                  transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])),
    batch_size=batch_size, shuffle=False)


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, class_plants + class_disease)

if use_gpu:
  model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)

for epoch in range(epochs):
  scheduler.step()
  print("starting training epoch: ", epoch)
  loss, f1_train_plant, f1_train_disease = train(train_loader, model, criterion, optimizer, epoch, use_gpu, log_interval, len(train_idx))
  f1_val_plant, f1_val_disease = validate(val_loader, model, epoch, use_gpu, log_interval, len(valid_idx))
  print("finished training epoch: ", epoch)
  print("loss: ", loss)
    
test(test_loader, model)
