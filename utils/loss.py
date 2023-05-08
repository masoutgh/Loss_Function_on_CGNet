import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None, ignore_label= 255):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        super().__init__()

        #self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        self.loss = nn.NLLLoss(ignore_index= ignore_label)
    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class DiceLoss(nn.Module):
    '''
    This file defines Dice Loss
    '''
    def __init__(self):

        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):
        
        smooth = 1.
        labels = F.one_hot(targets, num_classes = 12).permute(0,3,1,2)[:,:-1,:,:].contiguous() 
        iflat = outputs.contiguous().view(-1)
        tflat = labels.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class TverskyLoss(nn.Module):
    '''
    This file defines Dice Loss
    '''
    def __init__(self, alpha = 0.7, beta = 0.3, weight=None, ignore_label= 255) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6

    def forward(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor) -> torch.Tensor:

        outputs = F.softmax(outputs, dim=1)

        labels = F.one_hot(targets, num_classes=12).permute(0,3,1,2)[:,:-1,:,:].contiguous() 

        iflat = outputs.contiguous().view(-1)
        tflat = labels.contiguous().view(-1)

        intersection = (iflat * tflat).sum()
        fps = (iflat * ( 1 - tflat )).sum()
        fns = (( 1-iflat ) * tflat ).sum()

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        return torch.mean(1. - tversky_loss)
