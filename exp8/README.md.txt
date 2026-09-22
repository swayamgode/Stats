# Pima Indians Diabetes — K-Means Clustering & PCA

 ## 📌 Project Overview

 This project applies **unsupervised machine learning techniques** to the **Pima Indians Diabetes Dataset** to discover hidden patterns and visualize relationships among patients.

 The experiment uses:

 - **K-Means Clustering** to group similar patients
- **Elbow Method** to examine the suitable number of clusters
- **Silhouette Score** to evaluate cluster quality
- **Principal Component Analysis (PCA)** to reduce dimensionality
- **Matplotlib and Seaborn** for visualization

 The `Outcome` variable is **not used during clustering**. It is used afterward to compare the discovered clusters with the known diabetes categories.

---

 ## 🎯 Objectives

 - Apply clustering techniques to a high-dimensional medical dataset.
- Preprocess and standardize numerical features.
- Identify patient groups using K-Means clustering.
- Determine a suitable number of clusters.
- Evaluate cluster quality using the Silhouette Score.
- Apply PCA for dimensionality reduction.
- Visualize clusters in a two-dimensional space.
- Compare discovered clusters with the known diabetes outcomes.

---

 ## 📊 Dataset

 **Pima Indians Diabetes Dataset**

 The dataset contains:

 - **768 female patients**
- **8 medical features**
- **1 target variable (`Outcome`)**

 ### Features

 | Feature | Description |
| --- | --- |
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | Diabetes category: 0 or 1 |

---

 ## 🛠️ Technologies Used

 - Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

 ## 🔄 Methodology

 The experiment follows these steps:

```
Dataset
   ↓
Data Preprocessing
   ↓
Handle Invalid Values
   ↓
Feature Selection
   ↓
Standardization
   ↓
K-Means Clustering
   ↓
Elbow Method
   ↓
Silhouette Score
   ↓
PCA
   ↓
2D Visualization
   ↓
Cluster vs Outcome Analysis
```

---

 ## 🧹 Data Preprocessing

 Some medical variables contain zero values that are not physiologically meaningful.

 Zero values were treated as missing values for:

 - Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

 These values were replaced with `NaN` and subsequently imputed using the **median** of the corresponding feature.

 The `Outcome` variable was excluded from the clustering process.

---

 ## 📏 Feature Standardization

 Since K-Means uses distance calculations, the features were standardized using `StandardScaler`.

 This transforms the variables so that they are on a comparable scale.

---

 ## 🔵 K-Means Clustering

 K-Means was applied to the standardized data.

 Different values of `k` from **2 to 10** were evaluated.

 The final analysis selected:

```
Number of clusters = 2
```

 The resulting cluster sizes were:

```
Cluster 0 = 351 patients
Cluster 1 = 417 patients
```

---

 ## 📉 Elbow Method

 The Elbow Method was used to examine the relationship between the number of clusters and K-Means inertia.

 The elbow analysis indicated that **2 clusters** provide a reasonable representation of the data, with additional clusters providing diminishing improvement.

 The Silhouette analysis also selected `k = 2`.

---

 ## 📐 Silhouette Score

 The highest Silhouette Score obtained was:

```
0.1963
```

 for:

```
k = 2
```

 A Silhouette Score closer to 1 indicates well-separated clusters. The score of **0.1963** indicates that the clusters have **relatively weak separation and considerable overlap**.

 Therefore, the discovered clusters represent patterns in the data, but they are not strongly separated groups.

---

 ## 🧬 Principal Component Analysis

 PCA was used to reduce the original **8-dimensional feature space** to two dimensions.

 ### Explained Variance

 | Component | Variance Explained |
| --- | --- |
| PC1 | 28.54% |
| PC2 | 18.69% |
| **Total** | **47.23%** |

Thus, PC1 and PC2 together explain **47.23% of the total variance**.

---

 ## 🔍 PCA Feature Contributions

 ### PC1

 The variables with relatively large contributions to PC1 were:

 - Glucose — 0.424
- BMI — 0.402
- SkinThickness — 0.397
- Age — 0.386
- BloodPressure — 0.377

 ### PC2

 The variables with relatively large contributions to PC2 were:

 - Pregnancies — 0.558
- Age — 0.518
- BMI — -0.398
- SkinThickness — -0.309

 These loadings help explain which original variables contribute most strongly to the two-dimensional representation.

---

 ## 🩺 Cluster vs. Diabetes Outcome

 The `Outcome` variable was not used during clustering. It was only used afterward to compare the discovered groups with the known categories.

 ### Cluster Distribution

 | Cluster | Outcome 0 | Outcome 1 |
| --- | --- | --- |
| Cluster 0 | 156 | 195 |
| Cluster 1 | 344 | 73 |

### Percentage Distribution

 | Cluster | Outcome 0 | Outcome 1 |
| --- | --- | --- |
| Cluster 0 | 44.44% | **55.56%** |
| Cluster 1 | **82.49%** | 17.51% |

Cluster 0 contains a higher proportion of observations with `Outcome = 1`, while Cluster 1 contains a higher proportion with `Outcome = 0`.

 This indicates an **association** between the unsupervised clusters and the known diabetes categories. It should not be interpreted as a supervised prediction because `Outcome` was not used to create the clusters.

---

 ## 📈 Visualizations

 The program generates the following visualizations:

 1. **Elbow Method plot**
2. **Silhouette Score plot**
3. **Cluster feature heatmap**
4. **Standardized cluster-center heatmap**
5. **PCA cluster visualization**
6. **PCA visualization by diabetes Outcome**
7. **PCA feature-loading heatmap**

---

 ## 📁 Project Structure

```
Pima-Diabetes-Clustering/
│
├── diabetes.csv
├── exp8.py
└── README.md
```

---

 ## ▶️ How to Run

 ### 1\. Install Python

 Install Python 3.x from the official Python website.

 ### 2\. Install Dependencies

 Open Command Prompt or Terminal:

```
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

 ### 3\. Place the Dataset

 Make sure `diabetes.csv` is in the same folder as `exp8.py`.

 Example:

```
python/
├── exp8.py
└── diabetes.csv
```

 ### 4\. Run the Program

```
python exp8.py
```

 On Windows, you can also use:

```
py exp8.py
```

---

 ## 📝 Key Findings

 - The dataset contains **768 observations and 9 columns**.
- Eight numerical medical features were used for unsupervised learning.
- Invalid zero values were handled using median imputation.
- Features were standardized before clustering.
- **2 clusters** were selected.
- The Silhouette Score was **0.1963**, indicating relatively weak cluster separation.
- The first two PCA components explained **47.23%** of the total variance.
- Glucose, BMI, SkinThickness, BloodPressure, and Age contributed strongly to PC1.
- Pregnancies and Age contributed strongly to PC2.
- The clusters showed different distributions of the known `Outcome` categories.

---

 ## 📌 Conclusion

 The experiment successfully demonstrates the application of **K-Means clustering and PCA** to the Pima Indians Diabetes Dataset.

 K-Means identified two groups of patients based solely on their medical characteristics. PCA provided a two-dimensional representation for visualizing these groups. The comparison with `Outcome` showed that the discovered clusters have different distributions of diabetes categories, demonstrating that unsupervised learning can reveal meaningful structure without using the target variable during model development.

 However, the relatively low Silhouette Score (**0.1963**) indicates that the clusters overlap considerably, so the discovered groups should be interpreted as broad patterns rather than sharply separated patient categories.