# My Model training Cheat Sheet

## TL;DR;


## Importing Data
* go with Pytorch as soon as data is purely numerical
* use fast.ai for image manipulation
* Typical imports: 
```
import torch
import torch.nn.functional as F
import torch.nn as nn
```
* create a tensor with t = torch.tensor(df.values, dtype=torch.float) df.values is a numpy array
* t = torch.unsqueeze(t, 0) to add a dimension (ie typically from a list of scalars to a vector)
* t = torch.transpose(t, 0, 1) to transpose a vector
* 

## Find a good validation set
* with scikit-learn

## Image manipulation
* with PILImage but usually imported in higher level tools
* best with fast.ai
* convnext the base model to try first
* Must create a data loader: ImageDataLoaders.from_folder() and use vision_learner()
* TTA (average predictions at inference) and Gradient accumulation (save RAM) to optimize 
* Ensemble learning done through averaging predictions is useful at the end

# Loss function and metrics
* F.cross_entropy good for image classification

## Learning Loop
* Basic loop: note that zero_grad() not needed (unless between batches is using them)
```
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()
print(f"{loss:.3f}", end="; ")
```
* t = torch.where(condition on t, if True, if False) for a if-then on the data
* use fast.ai for learn rate search with learn.lr_find(suggest_funcs=(valley, slide))

## TODO
* TODO topic of initializing the weights in Pytorch

## Useful Links

- Pytorch docs https://pytorch.org/docs/stable/index.html
- Pytorch reference tutorial https://pytorch.org/docs/stable/dynamo/get-started.html
- About cross entropy loss function https://chris-said.io/2020/12/26/two-things-that-confused-me-about-cross-entropy/


