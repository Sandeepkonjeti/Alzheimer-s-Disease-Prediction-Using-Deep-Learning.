# Alzheimer's Disease Prediction Using Deep Learning

## Overview
This project focuses on predicting Alzheimer's Disease using advanced deep learning techniques. Leveraging Convolutional Neural Networks (CNNs) and pre-trained architectures like **VGG** and **MobileNet**, the model classifies MRI images to detect signs of Alzheimer's. The dataset is sourced from **Kaggle**.  

### Features:
- Utilized **Keras** and **TensorFlow** for model development.
- Visualized data insights with **Matplotlib**.
- Implemented transfer learning techniques for better accuracy.
- Collaborative project with a team of four, led by **Sandeep Konjeti**.

---

## Dataset
The dataset used in this project is publicly available on Kaggle. It contains MRI images categorized into different stages of Alzheimer's Disease.  

- **Source:** [Kaggle](https://www.kaggle.com/)
- **Data Preprocessing:** Images were resized and normalized for training the CNN models.

---

## Methodology
### 1. Preprocessing:
- Resized images to match the input size of CNN architectures.
- Performed normalization for faster convergence.

### 2. Model Architecture:
- **Convolutional Neural Networks (CNN):** Custom layers were designed to extract spatial features.
- **Pre-trained Models:** VGG and MobileNet were used for transfer learning.

### 3. Evaluation:
- Metrics: Accuracy, precision, recall, and F1-score.
- Visualization: Confusion matrix and loss/accuracy plots for training and validation.

---

## Results
The implemented models achieved significant accuracy in predicting Alzheimer's stages, demonstrating the effectiveness of deep learning in medical imaging applications.

---

## Installation and Usage

### Prerequisites:
1. Python 3.8+
2. Libraries:
   - TensorFlow
   - Keras
   - Matplotlib
   - NumPy
   - Pandas

### Steps to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Alzheimers-Prediction.git
   cd Alzheimers-Prediction
2. Install dependencies:
  ```bash
      pip install -r requirements.txt
```
3. Download the dataset from Kaggle and place it in the `data/` folder.  
  4. Run the training script:  
     ```bash
     python train_model.py
     ```  
  5. Evaluate the model:  
     ```bash
     python evaluate_model.py
     ``` 
## Folder Structure
Alzheimers-Prediction/

│

├── data/                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    # Dataset folder

├── models/             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # Saved model files

├── notebooks/          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Jupyter notebooks for analysis

├── train_model.py      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # Script for training the model

├── evaluate_model.py    &nbsp;&nbsp;&nbsp;&nbsp;   # Script for evaluating the model

├── requirements.txt     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    # Dependencies

└── README.md           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Project documentation

## Future Work
Incorporate additional pre-trained models to enhance accuracy.

Develop a web application for real-time Alzheimer’s detection.

Use Explainable AI (XAI) techniques to interpret predictions.
## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Sandeep Konjeti

Email: sandeepk200322@gmail.com
