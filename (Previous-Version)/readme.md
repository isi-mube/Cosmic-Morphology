# Cosmic Compendium

<p align="center">
  <img src="https://smd-cms.nasa.gov/wp-content/uploads/2023/04/potw2109a-jpg.webp?resize=2000,790" width="50%" alt="Hubble Beholds a Big, Beautiful Blue Galaxy">
  <br>
  <small><em>Hubble Beholds a Big, Beautiful Blue Galaxy<br>NGC 2336 is the quintessential galaxy — big, beautiful, and blue — and it is captured here by the NASA/ESA Hubble Space Telescope.</em></small>
</p>

1. [Python script](https://github.com/isi-mube/cosmic-compendium/blob/main/notebook.ipynb)

## About the Project
<p style="text-align: justify;">The objective of this <strong>project</strong> is to do binary image classification with galaxies, either spiral galaxies or non-spiral. The images are provided from the <strong>Galaxy Zoo 2 project</strong>, a Hubble Space Telescope open-source dataset.
</p>

## About the Hubble sequence
<p style="text-align: justify;">
The <strong>Hubble sequence</strong> is a morphological classification for galaxies, published by <a href="https://en.wikipedia.org/wiki/Edwin_Hubble"><strong>Edwin Hubble</strong></a> in 1926, dividing regular galaxies into three main classes; ellipticals, lenticulars, and <strong>spirals</strong>.
</p>

<p style="text-align: justify;">
Spiral galaxies, which are abundant in the universe, display a distinctive disk with spiraling arms and a gas and dust-rich central bulge. Studying spiral galaxies gives us a peek into the universe's past and helps us understand galaxy evolution and cosmology.
</p>

<p style="text-align: justify;">
The <strong>Galaxy Zoo</strong> involved human volunteers for visual and pattern recognition through a decision-tree process, answering questions progressively about a galaxy's structure.
</p>

<p align="center">
  <img src="https://galaxyzooblog.files.wordpress.com/2015/03/screen-shot-2015-03-31-at-9-48-30-am.png" width="35%">
</p>

## Project development:

* 01/06/23: Data collection and defining the problem; image binary classification to detect either spiral galaxy or not.
* 15/06/23: The script is nearly done.
  * Reduced the number of images used for the model training and testing to a subset of 1000. 
* 15/06/23 to 21/06/23: Fixing errors and cleaning the code.
* 29/06/23:
  * Further cleaning of the code and bugs.
  * Got an 81% accuracy predicting unseen galaxies. Total number of epochs: 35.
  * Removed some data directories from GitHub for optimization.
* 06/07/23: Finally got `val_accuracy` running (and not frozen) adapting [Sabina's](https://github.com/sabinagio) CNN structure in [Glaucoma detection](https://github.com/sabinagio/do-you-see-what-AI-see), I need to upgrade it further to get better scoring. Also:
  * 1.400 unique galaxies for the training subset and 600 unique galaxies for the validation subset.
  * Changed adam optimizer to adamax.
  * Added ImageDateGenerator parameters; horizontal flips, width and height shifts and zoom range to 0.2.
  * Augmented image size to 256x256 to get better resolution.
  * Created a cathartic [playlist](https://open.spotify.com/playlist/6UfNfescWaFLUzHC57LuwI?si=ccf07d382053420b) related to val_accuracy obsession to debug it.
* 07/07/23: Adapted the final CNN structure to:
  * Input layer 
  * 4 convolutional layers with 32, 64, 128 and 256 filters, followed by max pooling.
  * Flatten layer, converting 3D outputs to 1D vector.
  * 2 fully connected (dense) layers with 512 and 256 neurons.
  * An output layer with 1 neuron for binary classification

## Further project development:

* Develop a Streamlit app for more interactive model visualization.
* Take a break, keep focusing on Python basics, and move on to image segmentation and multiclassification.

## Model Results

<p align="center">
  <img src="https://i.ibb.co/xXk1Q6s/loss.png" width="45%">
</p>

<p align="center">
  <img src="https://i.ibb.co/MhHJVhB/accuracy.png" width="45%">
</p>

## Toolkit:

* **JupyterLab**: Enviorment for Python scripts and managing files.

**Libraries**

* **Pandas**: Data manipulation and analysis.
* **Numpy**: Arrays and mathematical functions.
* **Os**: File managment.
* **Warnings**: Roses are red, violets are blue --> Warnings are annoying.
* **Matplotlib**: Data visualization.
* **Seaborn**: Runs on top of matplotlib, HD data visualization.
* **Shutil**: File operations (copying, deleting...).
* **TensorFlow:** Machine Learning for Computer Vision.
* **Keras:** High-level neural networks API for Deep Learning, running on top of TensorFlow.
* **Sklearn:** Machine Learning metrics.
* **PIL:** Python Imaging Library to manipulate images.
* **Random:** To generate random subsets.
* **ImageDataGenerator:** To generate random data augmentation (flips, zoom...).

## Bibliography:

- Lintott, C. J. et al. (2008). Galaxy Zoo: Morphologies derived from visual inspection of galaxies from the Sloan Digital Sky Survey. Monthly Notices of the Royal Astronomical Society, 389(3), 1179–1189.
- Willett, K. W. et al. (2013). Galaxy Zoo 2: detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey. Monthly Notices of the Royal Astronomical Society, 435(3), 2835–2860.
- Chollet, F. (n.d.). Image Classification from Scratch. Keras. Retrieved from https://keras.io/examples/vision/image_classification_from_scratch/#introduction
- Chollet, F. (n.d.). Keras Metrics. Keras. Retrieved from https://keras.io/api/metrics/
- Nicholas Renotte. (n.d.). Build a Deep CNN Image Classifier with ANY Images. [Video]. YouTube. Available at: https://www.youtube.com/watch?v=jztwpsIzEGc
