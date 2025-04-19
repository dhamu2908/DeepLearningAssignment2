# DeepLearningAssignment2  CS24M027  Dhamodharan Muthu Muniyandi
This repository contains Python scripts and Jupyter notebooks used to produce the results presented in the accompanying report. For an in-depth explanation, please refer to the report included.

Report Link :- https://api.wandb.ai/links/m_dhamu2908/ofejtk2z

🚀 How to Run
🔧 partA_train.py

This script trains a CNN using custom hyperparameters. Use the following command:

python train_partA.py -e <epochs> -b <batch_size> -lr <learning_rate> -a <activation> -nf <num_filters> -ks <kernel_size> -dp <dropout_prob> -nd <neuron_dense> -fo <filter_org> -datao <data_aug> -bn <batch_norm> -train <train_folder> -test <test_folder>

Command-line arguments:

    -e, --epochs: Number of training epochs (default: 5)

    -b, --batch_size: Batch size (default: 32)

    -lr, --learning_rate: Learning rate (default: 0.001)

    -a, --activation: Activation function [relu, gelu, mish] (default: 'relu')

    -nf, --num_filters: Number of filters (default: 32)

    -ks, --kernel_size: Kernel size (default: 5)

    -dp, --dropout_prob: Dropout probability (default: 0)

    -nd, --neuron_dense: Neurons in the dense layer (default: 50)

    -fo, --filter_org: Filter organization [same, halve, double] (default: 'same')

    -datao, --data_aug: Enable data augmentation [True, False] (default: False)

    -bn, --batch_norm: Enable batch normalization [True, False] (default: False)

    -train, --train_folder: Path to training dataset

    -test, --test_folder: Path to test dataset

    Replace placeholders with actual values.

🔧 partB_train.py

This script fine-tunes a pre-trained model. Run it with:

python train_partB.py -e <epochs> -b <batch_size> -lr <learning_rate> -train <train_folder> -test <test_folder>

Arguments:

    -e, --epochs: Number of training epochs (default: 5)

    -b, --batch_size: Batch size (default: 32)

    -lr, --learning_rate: Learning rate (default: 0.001)

    -train, --train_folder: Path to training data

    -test, --test_folder: Path to test data

📂 Repository Code Content 

The code is organized into Part A and Part B, with Jupyter notebooks, Code was implement in python Notebook and to run explicitly we can use train.py
📘 Part A
    Builds a custom CNN with five convolutional layers (each followed by activation + max pooling), a dense layer, and a 10-class output layer.
    Performs a hyperparameter sweep to find optimal configurations.
    Runs the best model from the sweep, evaluates test accuracy, and plots predictions vs. ground truth.

📘 Part B
    Implements transfer learning using a pretrained model (ResNet50). Fine-tunes by freezing all layers except the final one and evaluates test accuracy.
