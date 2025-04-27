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

## 📂 Dataset
![Dataset](https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/dataset_image.png?raw=true)

## 📁 Folder Structure

## ⚙️ Workflow

## 🧪 Experiments

## 📊 Results

| Model          | Accuracy | F1-Score | Loss   | Precision | Recall  |
|----------------|----------|----------|--------|-----------|---------|
| ConvNeXtBase   | 0.9362   | 0.9454   | 0.2286 | 0.9412    | 0.9362  |
| EfficientNetB4 | 0.8777   | 0.8225   | 0.2804 | 0.9011    | 0.8723  |
| ResNet101V2    | 0.8830   | 0.8691   | 0.3582 | 0.8877    | 0.8830  |
| VGG19          | 0.8830   | 0.7914   | 0.2685 | 0.8824    | 0.8777  |

## 📈 ROC Analysis

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_b4.png?raw=true" alt="ROC B4" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_conv.png?raw=true" alt="ROC Conv" width="49.5%" />
</p>

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_resnet.png?raw=true" alt="ROC ResNet" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Pizza-Topping-Classification-Project/blob/main/images/roc_vgg.png?raw=true" alt="ROC VGG" width="49.5%" />
</p>

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

## 🚀 Future Developments

## 🛠️ Technology Used

## 📄 Licence

## 📬 Contact


