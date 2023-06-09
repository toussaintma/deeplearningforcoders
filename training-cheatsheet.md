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
* initialize tensor of zero with torch.zeros()
* t = torch.unsqueeze(t, 0) to add a dimension (ie typically from a list of scalars to a vector)
* t = torch.transpose(t, 0, 1) to transpose a vector
* 

## Find a good validation set
* with scikit-learn use cross validation cross_val_score(clf, df_x, df_y, cv=StratifiedKFold(n_splits=5))
* check imbalance for classification problems, train with balancing parameters
* cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
* in scikit learn, from sklearn.metrics import make_scorer must wrap a custom scoring function to be used by cross_val_score

## Feature Reduction
* find feature importance with Random Forest: forest_importances = pd.Series(clf.feature_importances_, index=df.columns)
* PCA
* Boruta algorithm: from boruta import BorutaPy will select df.head(clf.n_features_) sur la base de clf.ranking_ == 1
* other scikit-learn algorithms

## Imbalanced datasets
* minority augmentation: use SMOTE but only on numerical scaled features. Optimize the whole SMOTE + model
* may algorithms have a balanced parameters to optimize

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

## Ensemble
* VotingClassifier with or without weight
* StackingClassifier with a final estimator LogisticRegression(class_weight='balanced')

## TODO
* TODO topic of initializing the weights in Pytorch
nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

## Useful Links

- Pytorch docs https://pytorch.org/docs/stable/index.html
- Pytorch reference tutorial https://pytorch.org/docs/stable/dynamo/get-started.html
- About cross entropy loss function https://chris-said.io/2020/12/26/two-things-that-confused-me-about-cross-entropy/


