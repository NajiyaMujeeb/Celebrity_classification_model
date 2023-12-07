# Celebrity Image Classification using Convolutional Neural Networks (CNN)

This project aims to develop a CNN model capable of accurately differentiating between images of five distinct celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.

## Dataset Description
The dataset comprises varying quantities of images per person:
- Lionel Messi: 36 images
- Maria Sharapova: 34 images
- Roger Federer: 28 images
- Serena Williams: 29 images
- Virat Kohli: 41 images

## Model Architecture
- **Input Layer**: Supports pictures with sizes of 128x128 pixels and 3 channels (RGB).
- **Convolutional Layer**: Utilizes a (3, 3) kernel with ReLU activation to incorporate 32 filters.
- **Max Pooling Layer**: Reduces the spatial dimensions to extract essential features.
- **Flatten Layer**: Converts the 2D matrix data into a vector representation for further processing.
- **Dense Layers**: Fully connected layers with ReLU activation, including a dropout of 0.1 regularization. Softmax activation is used in the last layer for multi-class classification.

## Data Preprocessing
- All images were resized to 128x128 pixels, and pixel values were adjusted for optimal training.
- Dataset split: 80% for training the model and 20% for model performance validation.

## Training Process
- **Loss Function & Optimizer**: sparse_categorical_crossentropy loss function and adam optimizer.
- **Preventing Overfitting**: Early stopping technique applied, stopping training if there's no improvement for 10 consecutive rounds of learning.
- **Epochs & Batch Size**: Model trained over 25 epochs using a batch size of 32 for each iteration.

## Performance
- **Training Accuracy**: Started at 0.24 and peaked at 0.97 on the training dataset.
- **Test Accuracy**: Achieved 79.41% accuracy on the test dataset, demonstrating the model's ability to generalize well to unseen images.

## Conclusion
The CNN model designed for celebrity image classification showed promising performance, achieving an accuracy of 79.41% on the test dataset. Further optimization and fine-tuning could potentially enhance the model's performance for better differentiation between the specified celebrity images.

