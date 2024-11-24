## About the Data

<p align="justify">
This dataset consists of images of galaxies and their classification probabilities. The classification probabilities were determined by crowdsourced volunteer classifications as part of the Galaxy Zoo 2 project.</p>

## Dataset folders:

**From** Galaxy Zoo:
* **images_training:** JPG images of 61578 galaxies. Files are named according to their GalaxyId.
* **solutions_training:** Probability distributions for the classifications for each of the training images.
* **images_test:** JPG images of 79975 galaxies. Files are name according to their GalaxyId. You will provide probabilities for each of these images.

  * For GitHub optimization purposes, these datasets will not be available to download but they can be found [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).

**Created from** Python scripts:
* **spirals:** Labeled spiral galaxies, created during the python script. 
* **non_spirals:** Labeled non-spiral galaxies, created during the python script.
* **galaxies:** Master dataset folder. Contains training and validation subfolders
   * Training: 1400 unique galaxies, 700 per spiral and non-spiral.
   * Validation: 600 unique galaxies, 300 per spiral and non-spiral.
    
 * For GitHub optimization purposes, spiral and non_spiral datasets will not be available to download. Downloading [the following datasets](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) and running the notebook from scratch will create them.


<p align="justify">
The first column in each solution is labeled GalaxyID; this is a randomly-generated ID that only allows you to match the probability distributions with the images. 

The next 37 columns are all floating point numbers between 0 and 1 inclusive. These represent the morphology (or shape) of the galaxy in 37 different categories as identified by crowdsourced volunteer classifications as part of the Galaxy Zoo 2 project. 

These morphologies are related to probabilities for each category; a high number (close to 1) indicates that many users identified this morphology category for the galaxy with a high level of confidence. Low numbers for a category (close to 0) indicate the feature is likely not present.</p>

Visit the [Galaxy Zoo Decision Tree page](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/details/the-galaxy-zoo-decision-tree) for a detailed description of the data & [Galaxy Zoo - The Galaxy Challenge](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data) from Kaggle.
