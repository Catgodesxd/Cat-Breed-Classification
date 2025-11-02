# Fine-Tuning EfficientNet-B3 for Cat Breed Image Classification

## By Transfer Learning

## Why I choose this?

The reason I choose this is because it applies artificial intelligence to a real-world visual
recognition problem. This task is challenging for computers due to the subtle visual
similarities among breeds, such as color patterns, fur texture, and facial structure.

## Overview

Cat breed classification is a complex image recognition task that involves non-linear
relationships between features. Traditional algorithms cannot easily capture these patterns.
Deep Learning, particularly **Convolutional Neural Networks (CNNs)** , excels at
automatically learning hierarchical features from raw images. In this project, I will use
**Transfer Learning** with **EfficientNet-B3** , a high-performance CNN architecture pretrained
on ImageNet. **Fine-tuning** this model on the cat-breed dataset allows it to adapt its learned
representations to specific breed characteristics, achieving strong accuracy even with a
limited dataset.

## Dataset

I will be using the Gano Cat Breed Image Collection dataset from Kaggle to train the model.
This dataset contains a comprehensive collection of images categorized to assist the
development of a machine learning model.This dataset includes **15 different types of cat
breeds**. The breed consist of Abyssinian, American Bobtail, American Shorthair, Bengal,
Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian
Blue, Siamese, Sphynx, Tuxedo"


## CNN

Model **:** PyTorch EfficientNet-B3 (fine-tuned via transfer learning)
**Layer**
● **Input** - 3 × 300 × 300 (3 color channels, 300x300 pixels)
● **Convolution** - 3×3 convolution, 40 filters, learns basic edges, colo
● **MBConv (2×)** - Activation = Swish (SiLU), 24 filters, captures low-level shapes
● **MBConv (3×)** - Activation = Swish (SiLU), 40 filters, captures mid-level texture
● **MBConv (3×)** - Activation = Swish (SiLU), 80 filters, Wider kernel captures more
complex textures
● **MBConv (5×)** - Activation = Swish (SiLU), 112 filters, deeper semantic features
● **MBConv (4×)** - Activation = Swish (SiLU), 192 filters, complex shape recognition
● **MBConv (5×)** - Activation = Swish (SiLU), 320 filters, high-level concept features
● **MBConv (2×)** - Activation = Swish (SiLU), 512 filters, Final set of high-level
conceptual features.
● **Convolution** - 1×1 convolution, 1536 nodes, combine all feature maps
● **Pooling** - Average each feature map to a single value, flatten spatial info
● **Classifier** - Softmax, 15 nodes,1node for each breed


## Training

● Data Preparation
○ The cat image dataset was loaded using torchvision.datasets.ImageFolder()
and processed with various data augmentation techniques such as random
cropping, horizontal flipping, rotation, color jittering, and resizing to 300×
pixels.
○ The dataset was split into training (80%) and validation (20%) sets. To handle
class imbalance, a WeightedRandomSampler was used so that each breed
contributed equally during training.
○ The images were normalized using ImageNet statistics to match the
pre-trained model’s expected input.
● Model Construction
○ The base model EfficientNet-B3 was imported from torchvision.models with
pre-trained ImageNet weights.
○ The final classification layer was replaced with a new nn.Linear(in_features,15) 
layer to match the number of target classes (15 breeds).
○ A Dropout(0.5) layer was added before the classifier to reduce overfitting, and
the model was transferred to GPU for faster computation.
● Training Process
○ The model was trained using the Cross-Entropy Loss function with label
smoothing (0.1) to make predictions more robust.
○ The AdamW optimizer was used to update parameters efficiently with weight
decay to control overfitting.
○ A CosineAnnealingLR scheduler dynamically adjusted the learning rate for
smooth convergence.
○ To further improve generalization, MixUp and CutMix augmentation methods
(from the timm library) were applied randomly during training.
○ Each epoch involved a forward pass on the training set, gradient
back-propagation, and parameter updates, followed by validation evaluation.

## Evaluation

● Overall Accuracy: 83.1%
● Macro-average F1-score: 83.2%
● Weighted-average F1-score: 82.5%
● High-performing breeds include:
○ Sphynx and Siamese - F1-scores close to 1.00 (excellent classification)
○ Persian, Bombay, and Bengal - F1-scores above 0.
● Lower-performing classes include:
○ American Bobtail and American Shorthair, which showed moderate precision
and recall due to limited training samples or visual similarity with other
breeds.


## Summary

In this project, a deep learning model was developed to classify different **cat breeds** using
the **EfficientNet-B3** convolutional neural network architecture implemented in **PyTorch**. The
dataset consisted of 15 cat breeds, and extensive data augmentation techniques such as
random cropping, flipping, color jittering, and MixUp/CutMix were used to improve
generalization.
The model was fine-tuned from pre-trained ImageNet weights using the **AdamW optimizer**
with a cosine annealing learning rate scheduler. After training for multiple epochs, the model
achieved a **validation accuracy of 83.1%** with a **macro F1-score of 0.83** , indicating strong
classification performance across all breeds. The confusion matrix showed high accuracy for
visually distinctive breeds such as _Sphynx_ , _Siamese_ , and _Persian_ , while some overlap was
observed between breeds with similar appearances such as _American Shorthair_ and _British
Shorthair_.
For evaluation, the trained model was also tested on an external image not included in the
An image of my own beloved cat. The model successfully predicted the correct breed
(Ragdoll), demonstrating its ability to generalize to unseen real-world images.
In summary, the EfficientNet-B3 model proved to be an effective and efficient architecture for
cat breed classification, achieving reliable accuracy and robust performance. Future work
could involve expanding the dataset, using larger model variants (e.g., EfficientNet-B4 or
B5), or combining multiple models in an ensemble to further increase accuracy.




