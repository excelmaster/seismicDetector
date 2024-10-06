# Seismic Detection Using Spectrogram Analysis

This project is focused on detecting seismic events using spectrogram analysis and a YOLO model for object detection. The primary goal is to process seismic data (e.g., from Mars or the Moon), generate spectrograms, and use a trained YOLO model to detect seismic events.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Detection Workflow](#detection-workflow)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project is designed for the NASA Space Apps Challenge 2024, under the team name "Seismic Military Forces". The project involves:
- Loading seismic data in MSEED format.
- Generating spectrograms using `scipy`.
- Detecting seismic events using a pre-trained YOLO model.
- Plotting the results and saving them as images and CSV files.


## Directory Structure
To set up the project, first, clone the repository and then install the dependencies.

```bash
git clone https://github.com/excelmaster/seismicDetector.git
cd seismic-detection
pip install -r requirements.txt
```

## Directory Structure
The project is organized as follows:
seismic-detection/
│
├── data/
│   ├── catalog/               # CSV files with catalog data (e.g., event labels, metadata)
│   └── mars/                  # MSEED files from Mars data (raw seismic data from Mars)
│
├── models/
│   └── train1000.pt           # Pre-trained YOLO model file
│
├── runs/                      # Folder to store detection results and plots
│
├── scripts/                   # Additional Python scripts for processing (if any)
│
├── requirements.txt           # List of dependencies required for the project
└── README.md                  # Project documentation (this file)

