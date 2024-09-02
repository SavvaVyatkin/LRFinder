# LRFinder
This contains the changes to the pytorch-lightning LR Finder as detailed in https://savvavy.wordpress.com/2024/07/20/pytorch-lightning-learning-rate-finder/

For results, download the code and run lr_find.py
To customize, replace train_dataset and test_dataset with your own datasets, or replace ResNetMNIST with your own model. If running your own model, make sure to training_step and validation_step as required.
