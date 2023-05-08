# The effect of different loss functions on a semantic segmentation model
## Introduction
The effect of different loss functions on a semantic segmentation model was evaluated by selecting CGNet and experimenting with several popular loss functions, including Dice loss, Negative Log Likelihood, Tversky Loss, and Focal Loss. The chart below presents the results, which demonstrate that faster and more accurate training was achieved using Tversky loss with a beta value of 0.7 compared to the other loss functions.

![Alt text](result.png?raw=true "train accuracy")

The result is obtained from the train set of CamVid, with a size of 480*360. 

The conclusion drawn was that a good loss function, when chosen, could help the model to train faster and achieve better results. In semantic segmentation models, mIoU is commonly used as an evaluation criterion. Due to the similarity between $Dice = 2 TP / (2 TP + FP + FN)$ and $IoU = TP / (TP + FP + FN)$, it assists in speeding up the model's training.
