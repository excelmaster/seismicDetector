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

## Installation
To set up the project, first, clone the repository and then install the dependencies.

```bash
git clone https://github.com/excelmaster/seismicDetector.git
cd seismic-detection
pip install -r requirements.txt
