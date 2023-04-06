import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)
  print('---')
  print(maxk)
  print(batch_size)

  # reduce dimension to 1
  _, pred = output.topk(1, dim = 1, largest = True, sorted = True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def acc(loader, model):
  correct = 0.
  total = 0.

  for data in loader:
    x, y = data
    x = Variable(x, requires_grad=True).cuda()
    y = Variable(y).cuda()
    #print('label is: %f', y)
    #print(y.shape)
    output = model(x)
    output = torch.round(torch.sigmoid(output))
    #print('output is: %f', output)
    #print(output.shape)
    #_, predict = torch.max(output.data, dim=1)
    #rint(predict.shape)
    batch = y.size(0) * y.size(1)
    total += batch # batch
    #correct += (output == y).sum()
    
    output = output.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    #print('prediction and label')
    #print(output)
    #print(y)
    n_pos = np.count_nonzero(y) # the number of pos label
    #print('n pos')
    #print(n_pos)
    xor_list = np.logical_xor(output, y) # false (0) is the correct prediction
    #print('xor_list')
    #print(xor_list)
    count = batch - np.count_nonzero(xor_list) # the number of correct predictions
    #print('count')
    #print(count)
    and_list = np.logical_and(output, y)
    #print('and_list')
    #print(and_list)
    pos_list = np.logical_or(xor_list, and_list) # false is the correct prediction of neg label
    #print('pos_list')
    #print(pos_list)
    count_neg = batch - np.count_nonzero(pos_list)
    #print('count_neg')
    #print(count_neg)
    correct += (count_neg/(batch - n_pos) + (count - count_neg)/n_pos)
    
    """
    for i in range(y.size(0)):
      for j in range(y.size(1)):
        if y[i][j] == output[i][j]:
          correct += 1/9 if y[i][j] == 0 else 1"""  
  correct *= (batch/2)
  return 100 * correct / total