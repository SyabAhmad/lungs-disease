# Lung Disease Classification with X-Ray Images

This project focuses on classifying lung diseases using X-ray images. The model is trained to differentiate between three categories: **Lung Opacity**, **Normal**, and **Viral Pneumonia**. The dataset consists of images processed to be uniform in size and format, which are then used to train a Convolutional Neural Network (CNN) for effective classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop a deep learning model that can accurately classify lung X-ray images into one of the three specified categories. This can aid healthcare professionals in diagnosing lung conditions more efficiently.

## Technologies Used

- Python
- Pandas
- NumPy
- OpenCV
- Keras (TensorFlow backend)
- Scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SyabAhmad/lung-disease-classification.git
   cd lung-disease-classification
   ```

```

```

2. Install the required packages:
   <pre class="!overflow-visible"><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative"><div class="flex items-center text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9">bash</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-main-surface-secondary px-2 font-sans text-xs text-token-text-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>Copy code</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">pip install pandas numpy opencv-python keras scikit-learn
   </code></div></div></pre>

## Usage

1. **Data Preparation**:
   * Dataset can be found on kaggle.
   * Place your X-ray images in separate folders named `Lung_Opacity`, `Normal`, and `Viral Pneumonia` under the `Lung X-Ray Image` directory.
   * Update the paths in the `load_and_preprocess_images` function in `data_processing.py` to reflect your local file structure.
2. **Run Data Processing**:

   * Execute the data processing script to prepare your dataset:

   <pre class="!overflow-visible"><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative"><div class="flex items-center text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9">bash</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-main-surface-secondary px-2 font-sans text-xs text-token-text-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>Copy code</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">python data_processing.py
   </code></div></div></pre>
3. **Model Training**:

   * Once the data is processed and saved as `data.csv`, train the model using the training script:

   <pre class="!overflow-visible"><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative"><div class="flex items-center text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9">bash</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-main-surface-secondary px-2 font-sans text-xs text-token-text-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>Copy code</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">python model_training.py
   </code></div></div></pre>
4. **Evaluate the Model**:

   * After training, the model's performance will be evaluated, and the test accuracy will be printed.

## Data Preprocessing

The data preprocessing script performs the following steps:

* Loads images from specified directories.
* Converts images to grayscale.
* Resizes images to 128x128 pixels.
* Flattens and normalizes the pixel values.
* Saves the processed data as a CSV file for later use.

## Model Training

The model training script utilizes a Convolutional Neural Network (CNN) to classify the preprocessed images. The architecture includes:

* Convolutional layers for feature extraction.
* Max pooling layers for down-sampling.
* Dense layers for classification.

The model is trained using the Adam optimizer and categorical crossentropy loss function.

## Evaluation

The trained model is evaluated on a test set, and the loss and accuracy are printed to assess its performance.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE]() file for details.
