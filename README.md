# **Cosmic Morphology**

<p align="center">
  <img src="https://smd-cms.nasa.gov/wp-content/uploads/2023/04/potw2109a-jpg.webp?resize=2000,790" width="50%" alt="Hubble Beholds a Big, Beautiful Blue Galaxy">
  <br>
  <small><em>Hubble Beholds a Big, Beautiful Blue Galaxy<br>NGC 2336, captured by the NASA/ESA Hubble Space Telescope.</em></small>
</p>

## **Overview**
This **project** focuses on building a scalable data infrastructure to store and analyze real cosmic data retrieved from the NASA API. The primary goal is to develop a robust **multiclass classification model** capable of identifying various celestial phenomena, such as galaxies, nebulae, and stars.

🪐 Current Progress:
- Data extraction from NASA API and preparation.
- Exploratory Data Analysis (EDA) to understand data distribution & images.
- Training of a **Baseline Model** with ResNet50 and **Fine-Tuning**.
- Model evaluation and visualization of results.
  
## **Project Structure**

### 📂 Directories:
- **`1-Data-Collection`**: Scripts for data extraction from NASA API, fetching cosmic images.
- **`2-EDA`**: Notebooks and scripts for Exploratory Data Analysis (EDA), including class imbalance visualizations.
- **`3-Train-Val-Test-Split`**: Code for splitting data into training, validation, and test sets for model training.
- **`4-Machine-Learning`**: Contains:
  - **Baseline Model**: Initial ResNet50-based training and evaluation.
  - **Fine-Tuning**: Transfer learning with additional layers and parameters.
  - **Evaluation**: Metrics visualizations and model performance analysis.
  - 

## **🚀 Next Steps**

1. **Expand Dataset**:
   - Incorporate **SpaceNet dataset** ([Kaggle link](https://www.kaggle.com/datasets/razaimam45/spacenet-an-optimally-distributed-astronomy-data)).
   - Evaluate the impact of synthetic images on model performance.
   
2. **Data Cleaning**:
   - Further refine the dataset to ensure no mislabeled or noisy images (e.g., human activities or non-cosmic objects).

3. **Model Enhancements**:
   - Experiment with other architectures like **EfficientNet** or **Vision Transformers** to improve classification accuracy.
   - Optimize hyperparameters for better performance.

---


## **Bibliography**
- Lintott, C. J. et al. (2008). Galaxy Zoo: Morphologies derived from visual inspection of galaxies from the Sloan Digital Sky Survey. *Monthly Notices of the Royal Astronomical Society*, 389(3), 1179–1189.
- Willett, K. W. et al. (2013). Galaxy Zoo 2: Detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey. *Monthly Notices of the Royal Astronomical Society*, 435(3), 2835–2860.
- Chollet, F. (n.d.). Image Classification from Scratch. *Keras*. Retrieved from [https://keras.io/examples/vision/image_classification_from_scratch/](https://keras.io/examples/vision/image_classification_from_scratch/)
- Chollet, F. (n.d.). Keras Metrics. *Keras*. Retrieved from [https://keras.io/api/metrics/](https://keras.io/api/metrics/)
- Nicholas Renotte. (n.d.). Build a Deep CNN Image Classifier with ANY Images. *YouTube*. Retrieved from [https://www.youtube.com](https://www.youtube.com)