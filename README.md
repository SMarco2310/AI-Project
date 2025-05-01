## README: Intro to AI Project

### Overview

Welcome to the **Intro to AI Project**, which is part of an academic course focused on exploring concepts in Artificial Intelligence. This repository contains a Jupyter Notebook, `Intro_to_AI_Project.ipynb`, that demonstrates key aspects of image preprocessing, face detection, alignment, resizing, and embedding extraction using a pre-trained **FaceNet (Inception-ResNet)** model.

---

### Key Features and Workflow

#### 1. **Image Preprocessing**
   - **Resizing**: Images are resized to a uniform dimension of 160x160 to ensure consistency for the FaceNet model.
   - **Color Format Conversion**: All images are converted to RGB format if not already in this format.
   - **Format Standardization**: Images are saved in PNG format to standardize the dataset.

#### 2. **Face Detection and Cropping**
   - Utilizes **Haar Cascade** and **MTCNN (Multi-Task Cascaded Convolutional Networks)** for accurate face detection.
   - Automatically crops detected faces from images and aligns them using detected key points (e.g., eyes).

#### 3. **Embedding Extraction**
   - Employs the **FaceNet (Inception-ResNet)** architecture to compute 128-dimensional embeddings for each cropped face.
   - These embeddings serve as a compact representation of facial features, enabling tasks such as clustering, classification, and face verification.

#### 4. **Visualization**
   - Visual tools to display color histograms of images and before-and-after views of preprocessing steps.

#### 5. **Dataset Management**
   - Automatically organizes processed images into structured directories.
   - Supports batch processing of datasets while maintaining folder hierarchy.

#### 6. **Embedding Storage**
   - Extracted embeddings are saved to CSV files for further analysis and machine learning tasks.

---

### Dependencies

Ensure you have the following libraries installed to run the notebook:
- **Python 3.7+**
- **Jupyter Notebook**
- **OpenCV**: `pip install opencv-python`
- **MTCNN**: `pip install mtcnn`
- **PyTorch**: For implementing and running the FaceNet model.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For distance calculations and dataset splitting.
- **Pandas**: For embedding storage and analysis.

---

### How to Use

#### 1. Clone the Repository
```bash
git clone https://github.com/SMarco2310/AI-Project.git
cd AI-Project
```

#### 2. Open the Notebook
- Use Jupyter Notebook:
  ```bash
  jupyter notebook Intro_to_AI_Project.ipynb
  ```
- Or open it in Google Colab by following the link provided in the first cell of the notebook.

#### 3. Prepare Your Dataset
- Place your image dataset in a directory (e.g., `/content/drive/MyDrive/dataset`).

#### 4. Run the Notebook Step-by-Step
- Preprocess the dataset: Resize, align, and crop faces.
- Extract embeddings using the FaceNet model.
- Save and analyze embeddings for downstream tasks.

---

### Example Workflow

1. **Preprocessing**
   - Input dataset: `/content/drive/MyDrive/raw_images`
   - Output directory: `/content/drive/MyDrive/processed_images`
   - All images are resized, aligned, and stored in the output directory.

2. **Embedding Extraction**
   - Load preprocessed images from the output directory.
   - Use the FaceNet model to compute embeddings for each face.
   - Save embeddings to a CSV file for analysis.

3. **Analysis**
   - Visualize embeddings or use them for tasks like clustering, classification, or verification.

---

### File Structure

- **Intro_to_AI_Project.ipynb**: Main notebook with all the steps for preprocessing, face detection, embedding extraction, and visualization.
- **Haar Cascade XML File**: Used for face detection (download if not already included).
- **Result Directories**:
  - `processed_images`: Contains preprocessed and aligned images.
  - `embeddings.csv`: Stores extracted embeddings with labels.

---

### FaceNet Model Details

The **FaceNet (Inception-ResNet)** model is a state-of-the-art architecture for face recognition. This project implements the following:
- **Block Structures**:
  - `Block35`: Shallow feature extraction.
  - `Block17`: Intermediate feature extraction.
  - `Block8`: Deep feature extraction.
- **Embeddings**:
  - Outputs 128-dimensional embeddings for each face.

The model is implemented using PyTorch with custom layers and blocks.

---

### Future Enhancements

- Add support for real-time face detection via webcam or video streams.
- Implement clustering algorithms to group similar faces based on embeddings.
- Extend the workflow to include fine-tuning the FaceNet model on custom datasets.

---

### License

This repository is licensed under the [MIT License](LICENSE).

---

Feel free to contribute to the project by submitting issues or pull requests!
