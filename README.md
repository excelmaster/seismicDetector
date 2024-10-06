# Seismic Detection Using Spectrogram Analysis

This project is focused on detecting seismic events using spectrogram analysis and a YOLO model for object detection. The primary goal is to process seismic data (e.g., from Mars or the Moon), generate spectrograms, and use a trained YOLO model to detect seismic events.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
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


## Installation
To set up the project, first, clone the repository and then install the dependencies.

```bash
git clone https://github.com/excelmaster/seismicDetector.git
cd seismic-detection
pip install -r requirements.txt
```

## Usage
To run the project and perform detection on an example MSEED file, use the following command:

```bash
python detect_seismic.py
```
### Parameters
You can modify the following parameters in the script:

- **`data_directory`**: The path to the directory containing MSEED files for analysis. Example: `"data/mars/test/data/"`.
- **`model_path`**: The path to the YOLO model file to use for detection. Example: `"models/train1000.pt"`.
- **`results_path`**: The path to save detection results, such as spectrogram images and CSV files. Example: `"runs/detect/"`.
- **`test_filename`**: The name of the MSEED file to process (without extension). Example: `"XB.ELYSE.02.BHV.2021-10-11HR23_evid0011"`.

You can modify these parameters directly in the `detect_seismic.py` script or pass them as command-line arguments.

### Example Usage
Run the detection script on a specific MSEED file with the following command:

```bash
python detect_seismic.py --data_directory="data/mars/test/data/" --model_path="models/train1000.pt" --test_filename="XB.ELYSE.02.BHV.2021-10-11HR23_evid0011"


