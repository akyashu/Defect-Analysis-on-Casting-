# Casting_Product_Defect_Detection_CNN

###

CONTENTS

# Table of contents
1. [What is the Casting Product Defect Detection?](#Sec1)
2. [Quantifying Defect Detection](#sec2)
3. [Creating the Dataset](#sec3)
4. [Selecting the ML Algorithm](#sec4)
5. [Results](#sec5)
6. [Findings](#sec6)



## What is the Casting Product Defect Detection? <a name="Sec1"></a>
The Casting Product Defect Detection project focuses on identifying defects in casting products using a Convolutional Neural Network (CNN). This project aims to automate the quality control process in manufacturing by classifying images of casting products into two categories: "OK" and "Defective."

The model utilizes grayscale images from an industrial dataset and is trained to detect and classify defects accurately. The primary goal is to enhance quality assurance by predicting product defects with high precision, ultimately reducing manual inspection efforts and improving manufacturing efficiency.


## Quantifying Defect Detection <a name="sec2"></a>

To assess the performance of the CNN model in detecting defects in casting products, several metrics and methods are employed. The quantification of defect detection involves evaluating the model's accuracy, precision, recall, and F1 score, as well as using confusion matrices to visualize performance.

### Metrics

- **Accuracy**: This measures the proportion of correctly classified images (both "OK" and "Defect") out of the total number of images. It provides an overall sense of the model's performance.

- **Precision**: This indicates the proportion of true positive defect detections among all the instances where the model predicted a defect. It is crucial for understanding how well the model avoids false positives.

- **Recall**: This measures the proportion of actual defects correctly identified by the model. It is important for evaluating how well the model detects true positives.

- **F1 Score**: This is the harmonic mean of precision and recall, providing a single metric that balances both concerns. It is particularly useful when dealing with imbalanced datasets.

### Confusion Matrix

A confusion matrix is used to evaluate the performance of the classification model by presenting the number of true positives, true negatives, false positives, and false negatives. This matrix helps in visualizing the performance of the model and identifying areas where it may be making errors.


In the confusion matrix above:
- **True Positives (TP)**: Number of defect images correctly classified as defects.
- **True Negatives (TN)**: Number of OK images correctly classified as OK.
- **False Positives (FP)**: Number of OK images incorrectly classified as defects.
- **False Negatives (FN)**: Number of defect images incorrectly classified as OK.

### Evaluation

The model’s performance is evaluated on a separate test set that was not used during training. This ensures that the evaluation metrics reflect the model's ability to generalize to new, unseen data. Performance metrics are calculated to ensure that the model meets the desired accuracy and reliability for defect detection.

### Results

The performance metrics are summarized as follows:
- **Accuracy**: X% (replace with actual accuracy)
- **Precision**: X% (replace with actual precision)
- **Recall**: X% (replace with actual recall)
- **F1 Score**: X% (replace with actual F1 score)

These metrics indicate how well the CNN model is performing in detecting defects in casting products. The evaluation results help in understanding the strengths and weaknesses of the model and provide insights for further improvements.


## Creating the Dataset <a name="sec3"></a>

The dataset for detecting defects in casting products was created using real-life industrial data. This section details the process of generating and structuring the dataset for training and evaluating the CNN model.

### Dataset Source

The dataset is sourced from industrial casting processes and contains images of casting products labeled as either "OK" or "Defect." The dataset is divided into two main categories:
- **OK**: Images of casting products that are defect-free.
- **Defect**: Images of casting products with identified defects.

### Dataset Structure

The dataset is organized into subfolders to facilitate easy access and management.
![Image](https://github.com/user-attachments/assets/08f15c3a-1a89-475e-84b9-0ff156963651)
![Image](https://github.com/user-attachments/assets/bf59c228-f197-4d33-8ea3-e8a65814bb03)



## Selecting the ML Algorithm <a name="sec4"></a>

In this project, we focused on applying various machine learning algorithms to the task of image segmentation for defect detection in casting products. The following algorithms were evaluated:

### Algorithms

1. **Convolutional Neural Networks (CNN)**
   - **Description**: CNNs are a class of deep neural networks that are particularly well-suited for image classification and segmentation tasks. They use convolutional layers to automatically and adaptively learn spatial hierarchies of features from images.
   - **Architecture**: The CNN model used in this project consists of several convolutional layers (`Conv2D`), max-pooling layers (`MaxPooling2D`), dropout layers to prevent overfitting, and dense layers for final classification. The activation functions used are ReLU for hidden layers and Sigmoid for the output layer.
   - **Pros**: Effective for handling high-dimensional image data; capable of learning complex patterns.
   - **Cons**: Requires significant computational resources; can be prone to overfitting if not properly regularized.

2. **Data Augmentation Techniques**
   - **Description**: Data augmentation is used to artificially expand the size of the training dataset by applying various transformations to the images. Techniques include rotation, zoom, brightness adjustment, and shifting.
   - **Purpose**: Helps improve model generalization by introducing variability into the training data, which can enhance the model's ability to handle different scenarios in real-world applications.
   - **Impact**: Improved model robustness and accuracy by simulating diverse conditions that the model may encounter.

### Model Training and Evaluation

- **Training Process**: The CNN model was trained with a dataset of grayscale images (300x300 pixels) representing casting products. The training involved splitting the data into training and validation sets, using early stopping to prevent overfitting, and saving the best model using `ModelCheckpoint`.
  
- **Evaluation Metrics**: The performance of the CNN model was assessed using metrics such as accuracy, precision, recall, and F1-score. The evaluation involved using a confusion matrix to visualize the model's performance in classifying images as "OK" or "Defect."

### Summary

The CNN-based approach, combined with data augmentation techniques, provided a robust framework for image segmentation in defect detection tasks. This method proved effective in accurately identifying defects in casting products, demonstrating high performance and reliability compared to traditional image processing techniques.


## Results <a name="sec5"></a>

The results section provides an overview of the performance of the CNN model used for detecting defects in casting products. The key metrics and evaluation findings are as follows:

### Performance Metrics

- **Accuracy**: The CNN model achieved high accuracy in classifying images into "OK" and "Defect" categories. This reflects the model’s effectiveness in learning and distinguishing between the different classes of casting products.
  
- **Confusion Matrix**: The confusion matrix for the CNN model highlights the number of true positives, true negatives, false positives, and false negatives, showing how well the model performs in classifying each category.

![Image](https://github.com/user-attachments/assets/3e354e71-a81f-44d5-b86a-a63f24edc8c4)

- **Loss and Accuracy Curves**: The training and validation loss and accuracy curves indicate that the model converged well during training, with a steady decrease in loss and an increase in accuracy over epochs.

### Comparison of Algorithms

The CNN model outperformed traditional machine learning algorithms in defect detection tasks:

- **Convolutional Neural Networks (CNN)**
  - **Performance**: The CNN model, using ReLU activation functions, dropout for regularization, and MaxPooling layers for dimensionality reduction, achieved superior performance in segmenting images. The model's ability to capture complex features in images resulted in high classification accuracy and precision.
  - **Metrics**: Achieved high scores in accuracy, precision, recall, and F1-score.


## Findings <a name="sec6"></a>

![Image](https://github.com/user-attachments/assets/7fbee8f0-1a70-4f72-89bf-121046b2b8b9)
![Image](https://github.com/user-attachments/assets/6abbeaeb-a1cd-464f-8fe7-160e1f65f970)
![Image](https://github.com/user-attachments/assets/00ad5b6b-9bd2-4d07-bac3-bb10b8709d4b)

The objective of this project was to develop and evaluate a CNN-based model for detecting defects in casting products. The key findings from the project are as follows:

- **Model Effectiveness**: The CNN model demonstrated significant effectiveness in identifying defects in casting products. The use of ReLU activation functions, combined with dropout and MaxPooling layers, allowed the model to accurately classify images into "OK" and "Defect" categories. This indicates that CNNs are highly suited for image segmentation tasks in industrial contexts.

- **Data Augmentation Impact**: Data augmentation techniques, including rotation, zoom, brightness adjustments, and shifts, played a crucial role in enhancing the model’s generalization capability. These techniques helped the model perform better by exposing it to a more diverse set of training images.

- **Comparison with Traditional Algorithms**: Compared to traditional machine learning algorithms such as XGBoost, Random Forest, and KNN, the CNN model provided superior performance in image classification tasks. This underscores the importance of using specialized neural network architectures for handling high-dimensional image data.

- **Challenges and Limitations**: While the CNN model achieved high accuracy, challenges included ensuring the model's robustness against variations in image quality and defect types. The model's performance can be further improved with more data and fine-tuning of hyperparameters.

- **Future Directions**: Future research could explore additional advanced techniques to further enhance defect detection. Investigating other neural network architectures or hybrid models that combine CNNs with techniques such as transfer learning or generative adversarial networks (GANs) could provide additional improvements in model performance. 

- **Practical Implications**: The project highlights the practical benefits of using CNNs for defect detection in manufacturing. By accurately identifying defects, the model can help in improving product quality, reducing waste, and increasing efficiency in the production process.

Overall, the project demonstrates that CNNs are a powerful tool for image-based defect detection and provides a strong foundation for future advancements in the field.
