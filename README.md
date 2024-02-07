# Project Description

This is the second assignment of Data Mining and Visualization. In this project, we were provided with 327 word vectors, each made up of 300 dimensions. Our task involved performing data preprocessing, constructing three different types of K-means algorithms: standard K-means, K-means++, and Bisecting K-means from scratch, without using the Scikit-learn K-means library. Additionally, we need to use the Silhouette coefficient to determine the optimal number of clusters for dividing the data.

# Project Aims

1. Understand the application of Natural Language Processing (NLP) and unsupervised learning.
2. Demonstrate proficiency in using Numpy, Pandas, and Matplotlib libraries in Python.

# Result

By executing the `main.py` file, you can obtain the results. In the Python program, I performed Principal Component Analysis (PCA) to reduce the data dimension to 2 and normalized the data initially. The Silhouette coefficients of three K-Means algorithms—standard K-Means, K-Means++, and Bisecting K-Means—were computed for values of k ranging from 2 to 9. A Silhouette coefficient closer to 1 indicates better algorithm performance. I observed that all three K-Means algorithms exhibit optimal performance when k = 3. Therefore, I selected 3 as the optimal number of clusters for dividing the data.

<p align="center">
  <img src="https://github.com/andy41860/DataMining_ca2/blob/main/images/Figure_1.png" alt="Picture 1" width="500">
</p>

Upon manual inspection of the dataset, I identified that the word vectors fall into 3 categories: animal, country, and vegetable. This observation aligns with the conclusion drawn from the Silhouette coefficient. The clustering results of the three K-Means algorithms using k=3 were saved as CSV files in the `results_csv` folder. Additionally, I created scatter plots for the three K-Means algorithms using k=3 to illustrate the clustering effect.

<p align="center">
  <img src="https://github.com/andy41860/DataMining_ca2/blob/main/images/Figure_2.png" alt="Picture 1" width="500">
</p>
<p align="center">
  <img src="https://github.com/andy41860/DataMining_ca2/blob/main/images/Figure_3.png" alt="Picture 1" width="500">
</p>
<p align="center">
  <img src="https://github.com/andy41860/DataMining_ca2/blob/main/images/Figure_4.png" alt="Picture 1" width="500">
</p>
