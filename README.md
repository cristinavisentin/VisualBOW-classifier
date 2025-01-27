# Image Classifier Using Bag-of-Words Approach

## Overview
This project implements an image classifier using the **Bag-of-Words (BOW)** approach. The goal is to categorize images into one of 15 predefined scene categories (e.g., "Office", "Forest") using a **visual vocabulary** built from SIFT (Scale-Invariant Feature Transform) descriptors. The classifier uses various models including **$k$-Nearest Neighbors (k-NN)** and **Support Vector Machines (SVM)**.

## Approach
1. **Visual Vocabulary Construction**:
   - SIFT features are extracted from the training images.
   - K-means clustering is applied to group these features into visual words, forming a vocabulary.

2. **Image Representation**:
   - Images are represented as histograms of visual words based on the visual vocabulary.
   - **Hard assignment** and **soft assignment** strategies are used to match descriptors to visual words.

3. **Classification**:
   - **$k$-Nearest Neighbors (k-NN)**: Classifies test images by finding the closest histogram in the training set.
   - **Support Vector Machines (SVM)**: Implements linear SVM and **ECOC-based SVM** (Error-Correcting Output Code) for multi-class classification.
   - **Spatial Pyramid Kernel SVM**: Incorporates spatial information to improve classification.

---

## Implementation Details
- **Image Preprocessing**: Images were resized to **256x256** with cubic interpolation.
- **Feature Extraction**: Up to **800** SIFT features per image were used, resulting in about **200,000** features.
- **Vocabulary Sizes**: We experimented with **50**, **200**, and **400** visual words. The best results were achieved with **400 words**.
- **Distance Metrics**: Cosine similarity and $\chi^2$ distance were tested for k-NN classification. The $\chi^2$ distance provided the best performance.
