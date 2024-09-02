# LRFinder
This contains the changes to the pytorch-lightning LR Finder as detailed in https://savvavy.wordpress.com/2024/07/20/pytorch-lightning-learning-rate-finder/

For results, download the code and run lr_find.py
To customize, replace train_dataset and test_dataset with your own datasets, or replace ResNetMNIST with your own model. If running your own model, make sure to training_step and validation_step as required.

Compiled with:
lightning                 2.4.0
lightning-utilities       0.11.6
matplotlib                3.8.2
matplotlib-inline         0.1.6
numpy                     1.24.1
pytorch-lightning         2.3.3
torch                     2.1.2+cu118
torchaudio                2.1.2+cu118
torchinfo                 1.8.0
torchmetrics              1.4.1
torchvision               0.16.2+cu118
types-python-dateutil     2.8.19.14
typing_extensions         4.9.0
