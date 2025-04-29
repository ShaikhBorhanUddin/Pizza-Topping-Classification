# 🍕 Pizza Topping Classification
<p align="left">
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white&label=Made%20With" alt="Made with Colab">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/Pizza-Topping-Classification-Project" alt="Repo Size">
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/Pizza-Topping-Classification-Project" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/ShaikhBorhanUddin/Pizza-Topping-Classification-Project" alt="Issues">
  <img src="https://img.shields.io/badge/Data%20Visualization-Python-yellow?logo=python" alt="Data Visualization: Python">
  <img src="https://img.shields.io/badge/Result%20Visualization-Grad--CAM-red?style=flat" alt="Result Visualization: Grad-CAM">
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Version Control: Git">
  <img src="https://img.shields.io/badge/Host-GitHub-black?logo=github" alt="Host: GitHub">
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/Pizza-Topping-Classification-Project?style=social" alt="Forks">
  <img src="https://img.shields.io/badge/Project-Completed-brightgreen" alt="Project Status">
</p>

![Dashboard](https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/pizza_title_edit.png?raw=true)

## 📝 Project Overview

Pizza Topping Classification is a deep learning project aimed at automatically classifying different pizza toppings from images. Using a Convolutional Neural Network (CNN) architecture, the model is trained to accurately distinguish between multiple topping categories, helping automate tasks in food ordering, quality control, and restaurant management systems.

The project covers the full deep learning workflow, including:

- Dataset preprocessing and augmentation to enhance model robustness.

- Building and training an efficient CNN model from scratch.

- Performance evaluation using accuracy, loss curves, and classification reports.

- Visualizing predictions to validate real-world applicability.

This repository is structured for clarity and reproducibility, making it easy for anyone to understand, retrain, and deploy the model for their own pizza classification tasks or adapt it to other food classification problems.

## 📂 Dataset

The dataset [Pizza Toppings Classification](https://www.kaggle.com/datasets/gauravduttakiit/pizza-toppings-classification) used in this project is sourced from Kaggle. In real-world pizza images, many visible features can commonly appear, such as olives, onions, green peppers, sausage, extra cheese, and various seasoning elements. However, this dataset specifically focuses on only three toppings: **Basil**, **Mushroom**, and **Pepperoni**. Care was taken during dataset construction to ensure that no feature overlapping occurred — meaning each image contains only one distinct topping type without mixing multiple features.
The dataset  is sourced from Kaggle.

![Dashboard](https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/dataset_image.png?raw=true)

It is important to note that the dataset is biased towards **Pepperoni** pizzas, with **1,309** images, compared to **751** images for **Basil** and only **304** images for **Mushroom**. This imbalance needed to be considered carefully during model evaluation and interpretation.

Originally, the dataset size was approximately **312 MB**, with high-resolution images at **512×512 pixels** in `.jpg` format. To better suit transfer learning models, all images were resized to **224×224 pixels**, a standard input size for architectures like ConvNeXt, EfficientNet, ResNet, and VGG. After resizing, the dataset size was reduced significantly to **31.2 MB**, making it much faster to load and process during training without losing essential visual information.

## 📁 Folder Structure
```bash
Pizza-Topping-Classification-Project/ 
│ 
├── images/                       # Project images (dataset visualization, confusion matrix, ROC curves, Grad-CAMs)
│ 
├── notebooks/                    # Jupyter notebooks for training and evaluation 
|          │ 
|          ├── convnextbase_training.ipynb 
|          │ 
|          ├── efficientnetb4_training.ipynb 
|          │
|          ├── resnet101v2_training.ipynb 
|          │ 
|          └── vgg19_training.ipynb 
|
├── dataset/                        # Not included in this repository due to large number of images
│ 
├── README.md                       # Project overview and documentation 
|
├── requirements.txt                # Python dependencies 
|
└── LICENSE                         # License file
```

## ⚙️ Workflow

The project workflow was designed to ensure efficient experimentation and evaluation.
After preparing the dataset, four pre-trained models were selected and fine-tuned with a unified classification head. Training strategies were adjusted based on early metric stabilization to optimize computing resources.
Model performance was evaluated through confusion matrices, ROC curves, and Grad-CAM visualizations to understand both classification ability and feature focus.
Finally, the models were compared to identify the best balance between accuracy, interpretability, and computational efficiency.

## 🧪 Experiments

In this project, four different deep learning models were tested to classify pizza toppings: `ConvNeXtBase`, `EfficientNetB4`, `ResNet101V2`, and `VGG19`. All models were trained using preprocessed images of size 224×224 pixels to ensure compatibility with popular transfer learning architectures.

A consistent custom head was used across all models for fair comparison: a Dense layer with 512 units and ReLU activation, followed by a Dropout layer with a 0.5 rate to prevent overfitting. The training was done with a batch size of 256 for all experiments, promoting efficient GPU utilization. Training time was negligible, as nVIDIA A100 gpu was utilized in this experiment.

The total number of trainable parameters varied across models:

| Model           | Parameters (MB) | Training Epochs |
|-----------------|-----------------|-----------------|
| ConvNeXtBase    | 336.05 MB        | 70 epochs       |
| EfficientNetB4  | 70.93 MB         | 30 epochs       |
| ResNet101V2     | 166.62 MB        | 40 epochs       |
| VGG19           | 77.39 MB         | 30 epochs       |

Throughout training, it was observed that all models' performance metrics — including accuracy, loss, precision, recall, and F1-score — stabilized within the first 10 epochs. After initially testing ConvNeXtBase for 70 epochs, it was decided to reduce the number of training epochs for the subsequent models to optimize computational resources without compromising final performance.

## 📊 Results

Performance Matrix summary for all tested models are given in the following chart.

| Model          | Accuracy | F1-Score | Loss   | Precision | Recall  |
|----------------|----------|----------|--------|-----------|---------|
| ConvNeXtBase   | 0.9362   | 0.9454   | 0.2286 | 0.9412    | 0.9362  |
| EfficientNetB4 | 0.8777   | 0.8225   | 0.2804 | 0.9011    | 0.8723  |
| ResNet101V2    | 0.8830   | 0.8691   | 0.3582 | 0.8877    | 0.8830  |
| VGG19          | 0.8830   | 0.7914   | 0.2685 | 0.8824    | 0.8777  |

Based on the performance matrix, ConvNeXtBase clearly outperforms the other models with an impressive accuracy of 93.62%, an F1-score of 94.54%, and the lowest loss of 0.2286, indicating both high precision and recall across the pizza topping classes. EfficientNetB4, while showing strong precision at 90.11%, lags behind in recall and F1-score, reflecting that it struggles more to correctly identify all positive cases despite making fewer false predictions. ResNet101V2 delivers a balanced performance with an accuracy of 88.30% and a decent F1-score of 86.91%, showing it maintains reliability but with slightly higher loss and variability. VGG19, although achieving the same accuracy as ResNet101V2, has a lower F1-score and suggests that its predictions are less consistent across the different classes. Overall, ConvNeXtBase is the most balanced and high-performing model across all key evaluation metrics.

## 📈 ROC Curve Analysis

The ROC curves for EfficientNetB4 (on the left) show that while the model generally performs well, the curve for the Mushroom class is slightly less steep compared to Basil and Pepperoni, indicating a relatively higher false positive rate for that category. Overall, it still achieves good separability, but not as tightly packed near the top-left corner as a near-perfect classifier would. On the other hand, ConvNeXtBase (on the right) demonstrates consistently strong ROC curves across all three classes — Basil, Mushroom, and Pepperoni. The curves for ConvNeXtBase are closer to the ideal shape, hugging the top-left corner more tightly and indicating higher true positive rates with lower false positives across the board. This confirms that ConvNeXtBase is able to discriminate between pizza toppings more reliably compared to EfficientNetB4.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_b4.png?raw=true" alt="ROC B4" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_conv.png?raw=true" alt="ROC Conv" width="49.5%" />
</p>

For ResNet101V2 (on the left) and VGG19 (on the right), the ROC curves also show strong class separability, although they slightly lag behind ConvNeXtBase. In ResNet101V2, the Pepperoni and Basil classes maintain high AUC behavior, but the Mushroom curve dips slightly compared to others, hinting at minor difficulty in distinguishing that category. Similarly, VGG19's ROC curves are overall strong but exhibit slight variability in the Mushroom class as well. Both models perform admirably, especially for Pepperoni, but the minor inconsistencies, especially around Mushroom classification, reveal that their class separability is not as uniformly high as ConvNeXtBase.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_resnet.png?raw=true" alt="ROC ResNet" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_vgg.png?raw=true" alt="ROC VGG" width="49.5%" />
</p>

Overall, ConvNeXtBase again stands out with the most consistent and highest-performing ROC curves, confirming it as the top model in terms of both classification accuracy and class discrimination.

## 📉 Confusion Matrix

In the confusion matrices, EfficientNetB4 shows strong performance in correctly identifying Pepperoni pizzas but struggles more with distinguishing Basil pizzas, misclassifying a significant portion as Pepperoni. Mushroom classification also shows moderate confusion, which suggests that EfficientNetB4 has difficulty in separating the Mushroom category clearly from others. In contrast, ConvNeXtBase delivers an overall much stronger and more balanced classification across all three toppings. It maintains high true positives for Basil, Mushroom, and Pepperoni, with only minimal misclassifications between classes. Notably, it achieves particularly good separation for the Mushroom class, an area where EfficientNetB4 had visible confusion, reflecting ConvNeXtBase’s superior feature extraction and generalization ability in this task.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/cm_b4.png?raw=true" alt="EfficientNetB4 Confusion Matrix" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/cm_conv.png?raw=true" alt="ConvNeXt Confusion Matrix" width="49.5%" />
</p>

Looking at ResNet101V2 and VGG19, both models exhibit strong performance for Pepperoni pizzas, correctly classifying the majority of samples. However, they show slightly more confusion when it comes to Basil and Mushroom, particularly with ResNet101V2 where a noticeable portion of Basil samples are misclassified as Pepperoni. VGG19 manages better balance overall but still misclassifies some Mushroom pizzas as either Basil or Pepperoni, indicating a slight weakness in differentiating the more subtle topping differences. While both models handle Pepperoni exceptionally well, their slight struggles with Mushroom prevent them from matching the clean separation achieved by ConvNeXtBase.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/cm_resnet.png?raw=true" alt="Confusion Matrix ResNet" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/cm_vgg.png?raw=true" alt="Confusion Matrix VGG" width="49.5%" />
</p>

Overall, **ConvNeXtBase** emerges as the most balanced and robust model among all, demonstrating superior accuracy and minimal class confusion across all pizza topping categories.

## 🔥 Grad-CAM Visualization

Since the dataset used for the Pizza Topping Classification project is relatively small and somewhat biased, perfect visual focus from the models was not expected. Grad-CAM visualizations help in interpreting how the models "see" the toppings and where they concentrate their attention during prediction. While basic visual localization was achieved, some inconsistencies and diffused focus areas were understandable given the data limitations. Additionally, more advanced visualization techniques like Grad-CAM++ or Score-CAM were not applied, as they would have been unnecessarily complex for the scope of this project.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_b4_basil.png?raw=true" alt="GradCAM B4 Basil" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_b4_mushroom.png?raw=true" alt="GradCAM B4 Mushroom" width="49.5%" />
</p>

The first two Grad-CAM images belong to EfficientNetB4, focusing on Basil and Mushroom pizzas. For the Basil pizza, the model predominantly concentrates its attention around the center where basil leaves typically appear, though there is noticeable attention spillover toward the edges. For the Mushroom pizza, EfficientNetB4 captures a broader activation across the pizza surface, identifying mushroom patches but with less sharply defined regions, indicating that while the model recognizes the topping, it does not localize it very precisely.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_conv_mushroom.png?raw=true" alt="GradCAM Mushroom" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_conv_pepperoni.png?raw=true" alt="GradCAM Pepperoni" width="49.5%" />
</p>

The second row features ConvNeXtBase Grad-CAMs for Mushroom and Pepperoni pizzas. The Mushroom heatmap shows a tighter and more concentrated activation around mushroom areas compared to EfficientNetB4, suggesting better feature localization. For Pepperoni, ConvNeXtBase achieves very sharp focus, highlighting the individual pepperoni slices with high intensity. This suggests that ConvNeXtBase not only recognizes the topping well but also pinpoints its physical locations much more accurately.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_resnet_pepperoni.png?raw=true" alt="GradCAM ResNet Pepperoni" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_resnet_basil.png?raw=true" alt="GradCAM ResNet Basil" width="49.5%" />
</p>

The third row shows the Grad-CAM outputs for ResNet101V2, visualizing Pepperoni and Basil pizzas. ResNet101V2 captures the general area of the Pepperoni toppings fairly well but with slightly more diffused and scattered attention compared to ConvNeXtBase. For the Basil pizza, ResNet101V2's attention is reasonably centered but tends to spread toward irrelevant parts of the pizza crust, suggesting the model picks up both topping-specific and some background features during classification.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_vgg_basil.png?raw=true" alt="GradCAM VGG Basil" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/gradcam_vgg_mushroom.png?raw=true" alt="GradCAM VGG Mushroom" width="49.5%" />
</p>

The fourth and final row contains the Grad-CAM visualizations for VGG19, focusing on Basil and Mushroom pizzas. For both toppings, VGG19 shows broader and less concentrated heatmaps, indicating a less precise understanding of the toppings compared to the other models. Although the model correctly identifies the topping regions, the activations often cover unnecessary parts of the image, such as the pizza base or surrounding background, which may slightly impact classification clarity.

In conclusion, ConvNeXtBase produced the sharpest and most accurate Grad-CAM attention maps, clearly focusing on the toppings and outperforming the other models in visual localization.

## 🔮 Future Developments

Although the current project successfully demonstrates pizza topping classification using deep learning and Grad-CAM visualization, there are several potential future improvements:

- The current dataset focuses only on three toppings: basil, mushroom, and pepperoni. Future work could introduce more toppings (such as olives, onions, sausages, and green peppers) and mixed toppings scenarios, making the model more robust and closer to real-world pizzas.

- The dataset is notably biased toward pepperoni images. Future datasets could be better balanced to ensure the model does not develop a bias toward the majority class.

- Currently, each image has only one topping. Extending the model for multi-label classification would allow detecting multiple toppings present in a single pizza image — a more realistic and challenging setup.

- Grad-CAM was used for explainability. Future improvements could involve techniques like **Grad-CAM++** and **Score-CAM** for even sharper and more localized explanations without significantly increasing model complexity.

- The ConvNeXtBase model is highly accurate but large (around 1GB). Model compression, pruning, quantization, or knowledge distillation techniques could make deployment lighter without sacrificing much accuracy.

- A **Flask** or **FastAPI**-based deployment where users can upload pizza images and get topping predictions with Grad-CAM visualizations would enhance usability and accessibility.

- Synthetic pizza images could be generated using **GAN**s (Generative Adversarial Networks) to expand the dataset without manual labeling, particularly for underrepresented toppings.

- Exploring newer or more lightweight models like **ConvNeXtV2**, **MobileViT**, or **CoAtNet** could offer better trade-offs between accuracy and computational cost.

## 🛠️ Technology Used

`Python 3.10+` `TensorFlow 2.15` `Keras` `OpenCV` `Matplotlib ` `Seaborn` `Scikit-learn` `Google Colab Pro` `Kaggle` `Grad-CAM`

This project was developed using ***Python 3.10+*** as the core programming language. The deep learning models were built and trained using ***TensorFlow 2.15*** alongside ***Keras***, which provided a high-level, user-friendly API for model design and training. ***OpenCV*** was utilized for image preprocessing tasks, including reading and resizing the input images to match the models' input size requirements. For visualization and performance analysis, ***Matplotlib*** and ***Seaborn*** were employed to plot confusion matrices, ROC curves, and Grad-CAM outputs. ***Scikit-learn*** was instrumental in calculating evaluation metrics such as accuracy, F1-score, precision, and recall, as well as for generating ROC curve data. The training was conducted on ***Google Colab Pro*** to leverage powerful GPU resources (A100) for faster experimentation. The pizza topping dataset was sourced from ***Kaggle***, where initial exploration and verification of data quality were also performed.

## 📄 Licence

## 📬 Contact


