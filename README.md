# Thesis Repository

This repository contains scripts and models used for processing and analyzing the DelftBikes Dataset.

---

## Prerequisites

Before you begin, make sure you have access to the dataset:  
[DelftBikes Dataset](https://github.com/oskyhn/DelftBikes?tab=readme-ov-file)

Clone or download the dataset and place it in the correct directory as shown below.

---

## Folder Structure

Your project directory should look like this:

```
project_root/
│
├── data/
│   ├── raw/          # Original dataset files
│   ├── processed/    # Cleaned / processed data files
│   └── images/       # Reference images
│
└── python/
    ├── preprocessing/    # Data preprocessing scripts
    ├── model_1/          # Model 1 scripts
    ├── model_2/          # Model 2 scripts
    └── ...
```

Make sure to create this folder structure before running any scripts.

---

## Requirements

You will need **Python 3.8+** to run the project.

To install dependencies, run:

```bash
pip install -r requirements.txt
```

or 

```bash
conda env create -f environment.yml
```

---

## Usage Instructions

Follow the steps below to prepare and run the project.

### 1. Set up the environment

Make sure Python and all required dependencies are installed.

---

### 2. Prepare the Data

Download the **DelftBikes dataset** and organize the files as follows:

1. Place all **training images** from the dataset’s `train` folder into your project’s `data/images/` directory:
   ```
   data/images/
   ├── image_1.jpg
   ├── image_2.jpg
   └── ...
   ```

2. Place the **`train_annotation.json`** file into the `data/raw/` directory:
   ```
   data/raw/train_annotation.json
   ```

Make sure both the images and annotations are correctly placed before running the preprocessing script.


### 3. Run the data preprocessing script

From the `python/` directory, run:

```bash
python data_processing.py
```

This script performs the following actions:
- Loads the raw dataset  
- Cleans and processes the data  
- Generates a new JSON file with updated annotations  
- Saves the outputs in the `data/processed/` folder

  (For this thesis, I used a version of `data_processing.py` that **did not treat occluded objects as missing**.  
To achieve the same behavior, modify the list that defines all labels when creating the `available_part` list.)

---

### 4. Image Processing

Image preprocessing (e.g., resizing, normalization) is handled automatically in the `Dataloader` class in each model file.  
You do not need to run a separate image script — it executes automatically when running the models.

---

### 5. Run the models

After data preprocessing is complete, you can train or evaluate models.  
Each model has its own subfolder under `python/`.

---

## Summary

1. Create the folder structure  
2. Download and place the DelftBikes dataset in `data/raw/`  
3. Run `data_processing.py` to generate processed data  
4. Execute the desired model scripts  

---

## Notes

- Each script can be run independently and modified as needed.  
- Outputs (logs, results, and model files) are saved in their respective directories.  
- For troubleshooting, review console output or any log files created by the scripts.
