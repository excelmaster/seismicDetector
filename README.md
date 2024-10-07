# Seismic Detection Across the Solar System: Spectrogram Analysis Using Machine Learning Models

This project is focused on detecting seismic events using spectrogram analysis and a YOLO model for object detection. The primary goal is to process seismic data (e.g., from Mars or the Moon), generate spectrograms, and use a trained YOLO model to detect seismic events.

[Ver video](https://www.youtube.com/watch?v=dERFpcr99HM&ab_channel=KarenDayanna)


# Web interactive application - A quicker option:
We created an [interactive web platform](http://161.35.123.191:5000/) that allows scientists and the general public to:

- Visualize spectrograms with detected seismic events.
- Test different trained models dedicated to specific celestial bodies (such as the Moon or Mars) or more generalized models for various environments.
- Use available seismic data from NASA or terrestrial seismic records.
- Upload new data in miniseed format to test real-time detection.

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
This project is designed for the NASA Space Apps Challenge 2024, under the team name "6_Mic Military Research Unit". The project involves:
- Loading seismic data in MSEED format.
- Generating spectrograms using `scipy`.
- Detecting seismic events using a pre-trained YOLO model.
- Plotting the results and saving them as images and CSV files.


## Installation
To set up the project, first, clone the repository and then install the dependencies.

```bash
git clone https://github.com/excelmaster/seismicDetector.git
cd seismicDetector
pip install -r requirements.txt
```

### In **conda** it would be:
```bash
git clone https://github.com/excelmaster/seismicDetector.git
cd seismicDetector
conda create --name seismic_env python=3.8
conda activate seismic_env
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
```

### Output
The script will generate the following outputs:

1. **Spectrogram images**: A visual representation of the signal's frequency content over time, with detection markers for seismic events.
2. **CSV file**: A CSV file containing the coordinates of detected events along with their corresponding timestamps and confidence scores.
3. **Signal plot**: A plot of the original seismic signal with markers indicating detected events.

The results will be stored in the `runs/detect/` directory, which will be created automatically if it does not exist.

### Troubleshooting
- **Issue**: The MSEED file is not found.
  - **Solution**: Ensure that the `data_directory` and `test_filename` parameters are correctly set, and the file exists in the specified path.
  
- **Issue**: No events are detected in the spectrogram.
  - **Solution**: Lower the detection confidence threshold in the `perform_detection` function by setting the `conf` parameter to a lower value (e.g., `conf=0.3`).

- **Issue**: The model file could not be found at the specified path.
  - **Solution**: Verify that the model file path is correctly specified in the configuration. Make sure the file exists at that location or download the model again if it's missing.

## Model Training
The YOLO model was pre-trained with spectrogram images of seismic data from various sources, including data from the Mars InSight mission. To train your own model, you can modify the `ultralytics` training command as follows:

```bash
yolo train model=train1000.pt data=data.yaml epochs=100
```

Ensure that you have prepared a `data.yaml` file with the correct paths to your training data and labels. The YOLO model can be fine-tuned with more epochs or different learning rates to improve performance.

### Preparing Your Own Data
1. Organize your spectrogram images in directories labeled according to the type of event (e.g., `earthquake` and `noise`).
2. Create a `data.yaml` file that defines the structure and paths of your dataset.
3. Use the `ultralytics` command to start training your model.

## Detection Workflow
The detection process consists of the following steps:
1. **Load MSEED Data**: Read and process the seismic data using the `load_mseed` function.
2. **Generate Spectrogram**: Create a spectrogram using the `calculate_spectrogram` function from `scipy`.
3. **Perform Detection**: Use the YOLO model to detect events in the spectrogram image with the `perform_detection` function.
4. **Save Results**: Save detected coordinates to a CSV file and mark them on the spectrogram using `save_detection_results` and `plot_spectrogram`.

## Visualization
The results include spectrograms with markers for expected events and detected events. Use the `plot_spectrogram` function to visualize the spectrogram and compare it with known events from the catalog.

To visualize the original signal and highlight detected events, use the `plot_signal` function, which overlays detected points on the time-domain representation of the signal.

## Contributing
Contributions are welcome! Please create an issue or a pull request if you have suggestions or improvements. Make sure to follow the coding style and include documentation for any new functions or features added.

### Guidelines
1. Fork the repository and create a new branch for your feature.
2. Test your feature thoroughly before submitting a pull request.
3. Include a detailed description of the changes and the impact on existing code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or support related to this project, please reach out to the maintainers via email or through the GitHub issue tracker.



