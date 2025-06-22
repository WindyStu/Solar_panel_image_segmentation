# Solar Panel Image Segmentation

## Project Overview

This project focuses on developing deep learning models for accurate semantic segmentation of solar panels in aerial or satellite images. Solar panel image segmentation plays a crucial role in renewable energy management, enabling applications such as:
- Solar farm monitoring and maintenance
- Efficiency analysis of solar panel installations
- Automated inventory management of solar panel arrays
- Damage detection in solar panel systems

## Technologies Used

- **Deep Learning Frameworks**: PyTorch / TensorFlow
- **Segmentation Models**: U-Net, DeepLab, Mask R-CNN (or other state-of-the-art segmentation architectures)
- **Data Processing**: OpenCV, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Scikit-learn, MONAI (for medical image analysis metrics applicable to segmentation)

## Project Structure

```
solar_panel_segmentation/
│
├── data/                # Dataset for training and evaluation
│   ├── raw/             # Original images and annotations
│   ├── processed/       # Preprocessed data
│   └── splits/          # Train-val-test splits
│
├── models/              # Model architectures and checkpoints
│   ├── configs/         # Model configuration files
│   ├── saved_models/    # Trained model weights
│   └── architectures/   # Model definition scripts
│
├── scripts/             # Utility and training scripts
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Model evaluation script
│   ├── predict.py       # Inference/prediction script
│   └── data_preprocessing.py  # Data preprocessing pipeline
│
├── utils/               # Helper functions and utilities
│   ├── metrics.py       # Evaluation metrics
│   ├── visualization.py # Result visualization tools
│   └── data_loader.py   # Data loading utilities
│
├── docs/                # Documentation and reports
│   ├── figures/         # Visualization results
│   └── reports/         # Experimental reports
│
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── LICENSE              # License information
```

## Dataset

### Dataset Description
The project may utilize:
- **Public Datasets**: Such as Solar Panel Datasets from Kaggle, IEEE GRSS Data Fusion Contest, or other renewable energy-related image datasets
- **Custom Datasets**: User-provided aerial/satellite images of solar panels with corresponding segmentation masks

### Dataset Format
- Images: High-resolution RGB or multispectral images
- Annotations: Segmentation masks in PNG format with class labels (solar panel, background, etc.)
- Dataset Structure: Organized in a standard format compatible with common deep learning frameworks

## Installation

### Prerequisites
- Python 3.7+
- CUDA (for GPU acceleration, recommended for training)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/WindyStu/Solar_panel_image_segmentation.git
cd solar_panel_image_segmentation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model
```bash
python run.py --config configs/base.yaml 
```

### Making Predictions
```bash
python src/predict.py --model_path outputs/checkpoints/best_model.pth --test_dir data/testA --output_dir submissions/22211870121
```

### watching training logs
```bash
tensorboard --logdir=outputs/logs
```

## Model Metrics

The models are evaluated using standard segmentation metrics:
- Intersection over Union (IoU)
- Pixel Accuracy
- Mean IoU (mIoU)
 
## Results and Visualization & Performance Metrics
see report.md

## Contribution Guidelines

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

This project is open-source. Please check the `LICENSE` file for details.

## Acknowledgments
- Contributions from the open-source community
- Inspiration from existing solar panel analysis projects
- Support from renewable energy research organizations

For more details and updates, please stay tuned to this repository. Feel free to raise issues or contribute to improve the project!