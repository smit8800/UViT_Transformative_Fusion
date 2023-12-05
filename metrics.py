import torch
import torch.nn as nn


def acc_sen_iou(pred, mask) :

    pred = torch.round(pred)
    #TP = (mask * pred).sum(1).sum(1).sum(1)
    TP = (mask * pred).sum()
    #TN = ((1 - mask) * (1 - pred)).sum(1).sum(1).sum(1)
    TN = ((1 - mask) * (1 - pred)).sum()
    #FP = pred.sum(1).sum(1).sum(1) - TP
    FP = pred.sum() - TP
    #FN = mask.sum(1).sum(1).sum(1) - TP
    FN = mask.sum() - TP
    acc = (TP + TN)/ (TP + TN + FP + FN)
    acc = torch.sum(acc)
    iou = (TP)/(TP + FN + FP)
    iou = torch.sum(iou)

    #iou = jsc(mask.cpu().numpy().reshape(-1), pred.cpu().numpy().reshape(-1))

    sen = TP / (TP + FN)
    sen = torch.sum(sen)
    return acc, sen, iou

def diceCoeff(pred, gt, smooth=1e-5, activation='none'):
    """ computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N

#calculation of precision and recall
def calc_prerec(mask, pred):
  pred = torch.round(pred)
  #TP = (mask * pred).sum(1).sum(1).sum(1)
  TP = (mask * pred).sum()
  #FP = pred.sum(1).sum(1).sum(1) - TP
  FP = pred.sum() - TP
  #FN = mask.sum(1).sum(1).sum(1) - TP
  FN = mask.sum() - TP
  prec = (TP)/ (TP + FP)
  prec = torch.sum(prec)
  recc = TP / (TP + FN)
  recc = torch.sum(recc)
  return prec, recc

#calculate DSC
def dice_score(mask,pred):
  pred = torch.round(pred)
  #TP = (mask * pred).sum(1).sum(1).sum(1)
  TP = (mask * pred).sum()
  #FP = pred.sum(1).sum(1).sum(1) - TP
  FP = pred.sum() - TP
  #FN = mask.sum(1).sum(1).sum(1) - TP
  FN = mask.sum() - TP
  dice=(2*TP)/(2*TP+FP+FN)
  dice=torch.sum(dice)
  return dice


class MyFrame():
    def __init__(self, net, learning_rate, device, evalmode=False):
      self.net = net().to(device)
      self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=learning_rate, weight_decay=0.0001)
      self.loss = horny_loss().to(device)
      self.lr = learning_rate

    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch

    def optimize(self):
      self.optimizer.zero_grad()
      pred = self.net.forward(self.img)
      #print(list(mask.shape),mask.dtype)
      #print(list(pred.shape),pred.dtype)
      loss = self.loss(self.mask, pred)
      loss.backward()
      self.optimizer.step()
      return loss, pred

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):

        if factor:
            new_lr = self.lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print ('update learning rate: %f -> %f' % (self.lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr

class horny_loss(nn.Module):
    def __init__(self, batch=True):
        super(horny_loss, self).__init__()
        self.batch = batch
        self.mae_loss = torch.nn.L1Loss()
        self.bce_loss = torch.nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def iou_loss(self, inputs, targets):
        smooth = 0.0
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)

        intersection = (inputs * targets).sum(1).sum(1).sum(1)
        total = (inputs + targets).sum(1).sum(1).sum(1)
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return (1 - IoU.mean())

    def forward(self, y_true, y_pred):
        a = self.mae_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        c = self.bce_loss(y_pred, y_true)
        d = self.iou_loss(y_pred, y_true)
        loss = 0.15*a + 0.4*b  + 0.15*c + 0.3*d
        #loss = 0.5*b + 0.5*c
        return loss







