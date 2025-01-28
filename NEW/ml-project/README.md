# Machine Learning Project: VGG-16 and Other Algorithms

This project implements various machine learning algorithms, including VGG-16, ResNet, Probabilistic Neural Network (PNN), K-Nearest Neighbors (KNN), and Support Vector Machine (SVM). The goal is to provide a comprehensive framework for experimentation and evaluation of these models on a given dataset.

## Project Structure

- **data/**: Contains the dataset files.
  - **raw/**: Raw dataset files used for training and testing.
  - **processed/**: Processed dataset files ready for model use.

- **notebooks/**: Jupyter notebooks for model implementation and experimentation.
  - **vgg16.ipynb**: Implementation of the VGG-16 algorithm.
  - **resnet.ipynb**: Implementation of the ResNet algorithm.
  - **pnn.ipynb**: Implementation of the Probabilistic Neural Network.
  - **knn.ipynb**: Implementation of the K-Nearest Neighbors algorithm.
  - **svm.ipynb**: Implementation of the Support Vector Machine algorithm.

- **src/**: Source code for model implementations and utilities.
  - **models/**: Contains model architecture and training methods.
    - **vgg16.py**: VGG-16 model implementation.
    - **resnet.py**: ResNet model implementation.
    - **pnn.py**: Probabilistic Neural Network implementation.
    - **knn.py**: K-Nearest Neighbors implementation.
    - **svm.py**: Support Vector Machine implementation.
  - **utils/**: Utility functions for data preprocessing.
    - **data_preprocessing.py**: Functions for normalization, augmentation, and dataset splitting.
  - **train.py**: Main training script for orchestrating the training process.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place the raw dataset files in the `data/raw/` directory.
   - Run the data preprocessing script to generate processed data.

4. Run the Jupyter notebooks for experimentation:
   ```
   jupyter notebook notebooks/
   ```

## Usage

- Each notebook contains detailed instructions on how to run the models, including data loading, training, evaluation, and visualization of results.
- The `train.py` script can be used to train models programmatically.

## Models Implemented

- **VGG-16**: A deep convolutional neural network architecture known for its performance in image classification tasks.
- **ResNet**: A residual network that allows for training very deep networks by using skip connections.
- **Probabilistic Neural Network**: A type of neural network that uses probability distributions for classification.
- **K-Nearest Neighbors**: A simple, instance-based learning algorithm used for classification and regression.
- **Support Vector Machine**: A supervised learning model used for classification and regression tasks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.