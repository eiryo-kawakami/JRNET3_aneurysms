# Proposal for machine learning-based tailored treatment for ruptured cerebral aneurysms

This is a code repository for [???](link to paper)

## Main Analysis 

The `main_analysis.py` script is designed to perform clustering analysis on medical data. Below is an explanation of its functionality:

### Key Functionalities

#### Data Loading and Preprocessing
- **`load_and_preprocess_data`**: Loads data from CSV and Excel files, removes unnecessary columns, and renames columns for consistency.
- **`prepare_dataset`**: Filters the data to include only relevant cases (e.g., embolization-only treatments) and prepares the dataset for analysis by separating:
    - Features (`X`)
    - Target values (`y`)
    - Additional metadata (`X_other_info`)

#### Random Forest-Based Dissimilarity Calculation
- **`estimate_random_forest_dissimilarity`**:
    - Trains a Random Forest model to calculate a proximity matrix based on leaf node assignments.
    - Converts the proximity matrix into a distance matrix, which is saved for further analysis.
    - Calculates and visualizes feature importances as a bar plot.

#### Dimensionality Reduction
- **`perform_dimensionality_reduction`**:
    - Uses UMAP (Uniform Manifold Approximation and Projection) to reduce the dimensionality of the distance matrix.
    - Determines the optimal number of dimensions (`optimal_umap_k`) using a precomputed loss file (`umap_loss.csv`).

#### Optimal Cluster Number Estimation
- **`estimate_optimal_cluster_numbers`**:
    - Reads BIC (Bayesian Information Criterion) values from CSV files to determine the optimal number of clusters (`optimal_k`).

#### Clustering
- **`perform_clustering`**:
    - Applies Gaussian Mixture Models (GMM) to cluster the reduced data.
    - Adds the clustering results to the original dataset and saves them as a CSV file (`state_clustered_dataframe.csv`).

### Main Workflow
The **`main`** function orchestrates the entire process:
1. Loads and preprocesses the data.
2. Prepares the dataset for analysis.
3. Calculates the Random Forest-based distance matrix.
4. Performs dimensionality reduction using UMAP.
5. Estimates the optimal number of clusters.
6. Performs clustering and saves the results.

## Outputs
- **Distance Matrix**: Saved as `state_dist_matrix.npz`.
- **Reduced Matrix**: Saved as `state_scatter_matrix.npz`.
- **Feature Importances**: Saved as `feature_importances.pdf`.
- **Clustered Data**: Saved as `state_clustered_dataframe.csv`.

## Execution
The script can be executed using:

```bash
python main_analysis.py
```

### Summary of Python scripts in `scripts/`

**`main_analysis.py`**  
    Performs the main clustering analysis on medical data using Random Forest-based dissimilarity, UMAP dimensionality reduction, and Gaussian Mixture Models (GMM). Outputs clustered data and visualizations.

**`estimate_umap_loss.py`**  
    Calculates UMAP loss for different numbers of components (`n_components`) to determine the optimal dimensionality for UMAP. Outputs a CSV file with loss values.

**`calc_BIC.py`**  
    Computes the Bayesian Information Criterion (BIC) for Gaussian Mixture Models (GMM) with varying cluster numbers (`k`). Outputs a CSV file with BIC values.

**`statistical_analysis.py`**  
    Analyzes clustered data to identify significant features and their impact on mRS differences. Aggregates p-values from Wilcoxon tests and generates visualizations.

**`ProxMatrix.py`**  
    Provides utility functions for calculating proximity and distance matrices from Random Forest leaf assignments.  
    - **`get_upper_prox_matrix`**: Computes the upper triangular proximity matrix using leaf node assignments.  
    - **`get_dist_matrix`**: Converts the proximity matrix into a symmetric distance matrix normalized by the number of trees.

**`utils.py`**  
    Contains shared utility functions and constants used across scripts, such as column definitions, file handling, and helper functions.

**`feature_value_variations.py`**
Analyzes variations in feature values across clusters by calculating standard deviations for continuous features and entropies for categorical features. Saves results as CSV files.

**`preprocess_excel_data.py`**
Preprocesses raw Excel data by renaming columns, imputing missing values with MissForest, and preparing datasets for analysis. Outputs cleaned and imputed CSV files.

**`visualization.py`**
Generates visualizations for clustered data, including bubble plots, heatmaps, and scatter plots. Saves visualizations as PDF files.

### Summary of R scripts in `R_scripts/`

**`multiple_exact_wicox_test.R`**
Performs statistical hypothesis testing (Wilcoxon rank-sum test) on multiple datasets and adjusts p-values for multiple comparisons.
**`c50_plot.R`**
Builds and visualizes a decision tree model to predict a binary outcome based on clinical or demographic features.
