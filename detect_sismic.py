#"Import Libraries"
import os
import numpy as np
import pandas as pd  # Añadir pandas para guardar en CSV
import matplotlib.pyplot as plt
from ultralytics import YOLO
from obspy import read
from scipy import signal
from matplotlib import cm
import re

# ----------------- FUNCTIONS ----------------- #
def load_mseed(file_path):
    """Load the MSEED file and return the main trace."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    st = read(file_path)
    return st.traces[0]

def calculate_sampling_rate(tr_times):
    """Calculate the sampling frequency based on time differences."""
    return 1 / np.mean(np.diff(tr_times))

def calculate_spectrogram(tr_data, sampling_rate):
    """Calculate the spectrogram of the signal using scipy."""
    f, t, sxx = signal.spectrogram(tr_data, sampling_rate)
    vmax = np.percentile(sxx, 99)
    sxx_normalized = sxx / vmax 
    return f, t, sxx_normalized

def plot_spectrogram(t, f, sxx_normalized, save_path, w=10, h=3, labels=False, show_colorbar=False, x_coords=None, cmap_name='jet'):
    """Plot and save the spectrogram."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    cmap = ax.pcolormesh(t, f, sxx_normalized, cmap=cmap_name, vmax=1)
    if show_colorbar:
        fig.colorbar(cmap, ax=ax)
    lines = []
    labels = []
    if 'training' in data_directory:
        data_catalog = 'space_apps_2024_seismic_detection/data/catalog/'
        catalog_file = os.path.join(data_catalog, 'catalog.csv')
        try:
            catalog_df = pd.read_csv(catalog_file)
            mseed_filename = os.path.basename(mseed_file).replace('.mseed', '')
            matched_row = catalog_df[catalog_df['filename'] == mseed_filename]
            if not matched_row.empty:
                time_rel_value = matched_row['time_rel(sec)'].values[0]
                line = ax.axvline(x=time_rel_value, c='green', linestyle='-', linewidth=2, label='Expected time_rel')
                ax.text(time_rel_value, max(f)-3, f'x = {time_rel_value:.2f}', color='green', fontsize=12, verticalalignment='bottom')
                lines.append(line)
                labels.append('Expected time_rel')
            else:
                print(f"The filename '{mseed_filename}' was not found in the catalog.csv file.")
        except Exception as e:
            print(f"Error processing the file. {catalog_file}: {e}")

    if x_coords is not None:
        for x in x_coords:
            line = ax.axvline(x=x, color='red', linestyle='--', linewidth=2, label='Predicted Arrival')
            ax.text(x, max(f)-4, f'x = {x:.2f}', color='red', fontsize=12, verticalalignment='bottom')
            lines.append(line)
            labels.append('Predicted Arrival')
    ax.legend(lines, labels)
    if labels:
        ax.set_xlim([min(t), max(t)])
        ax.set_ylim([min(f), max(f)])
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Spectrogram')
        ax.axis('on')
    else:
        ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def perform_detection(model, source_image, save_dir):
    """Perform detection using YOLO and return the detection coordinates."""
    results = model.predict(
        source=source_image,
        save=True,
        save_txt=True,
        save_conf=True,
        save_dir=save_dir,
        conf=0.5
    )
    return results

def save_detection_results(results, tr_data_len, sampling_rate, output_path):
    """Save the detection results and calculate the scaled coordinates."""
    x_coords, probs, x_coords_scaled = [], [], []

    for result in results:
        for box in result.boxes:
            x1 = int(box.xyxy[0][0])
            x_coords.append(x1)
            probs.append(float(box.conf[0]))

    x_coords_scaled = [(x * tr_data_len) / ((size*100) * sampling_rate) for x in x_coords]
    df = pd.DataFrame({'filename':test_filename,'detect_coords': x_coords, 'probability': probs, 'time_rel(sec)': x_coords_scaled})
    df.to_csv(output_path, index=False)
    return x_coords_scaled

def get_latest_predict_folder(base_dir="runs/detect"):
    """Gets the latest predict# folder in the base directory."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "predict")
    
    predict_dirs = [d for d in os.listdir(base_dir) if re.match(r'predict\d+', d)]
    if not predict_dirs:
        return os.path.join(base_dir, "predict")
    
    max_predict = max([int(re.findall(r'\d+', d)[0]) for d in predict_dirs])
    return os.path.join(base_dir, f"predict{max_predict}")

def plot_signal(times, data, save_path, w=10, h=3, x_coords=None, mseed_file=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.plot(times, data, label='Original Signal')
    lines = []
    labels = []
    if 'training' in data_directory:
        data_catalog = 'space_apps_2024_seismic_detection/data/catalog/'
        catalog_file = os.path.join(data_catalog, 'catalog.csv')
        try:
            catalog_df = pd.read_csv(catalog_file)
            mseed_filename = os.path.basename(mseed_file).replace('.mseed', '')
            matched_row = catalog_df[catalog_df['filename'] == mseed_filename]

            if not matched_row.empty:
                time_rel_value = matched_row['time_rel(sec)'].values[0]
                line = ax.axvline(x=time_rel_value, c='green', linestyle='-', linewidth=2, label='Expected time_rel')
                ax.text(time_rel_value, max(data)*0.6, f'x = {time_rel_value:.2f}', color='green', fontsize=12, verticalalignment='bottom')
                lines.append(line)
                labels.append('Expected time_rel')
            else:
                print(f"The filename '{mseed_filename}' was not found in the catalog.csv file.")
        except Exception as e:
            print(f"Error processing the file. {catalog_file}: {e}")

    if x_coords is not None:
        for x in x_coords:
            line = ax.axvline(x=x, c='red', linestyle='--', linewidth=2, label='Predicted Arrival')
            ax.text(x, max(data)*0.5, f'x = {x:.2f}', color='red', fontsize=12, verticalalignment='bottom')
            lines.append(line)
            labels.append('Predicted Arrival')
    ax.legend(lines, labels)
    ax.set_xlim([min(times), max(times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{os.path.basename(mseed_file)}', fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ----------------- MAIN CODE ----------------- #

# Define the path to save the results.
results_path = "runs/detect"

# Load the YOLO model.
model_path = r"static\models\train100.pt"
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"The model file could not be found at the specified path: {model_path}")


model = YOLO(model_path)
nombre_modelo = os.path.basename(model_path)
# Verificar si el nombre del modelo contiene '1024' al final (antes de la extensión)
if nombre_modelo.endswith('1024.pt'):
    size=10.24
else:
    size=6.4

# Example MSEED file.
test_filename = 'XB.ELYSE.02.BHV.2021-10-11HR23_evid0011'
data_directory = 'source/mars/test/'
mseed_file = f'{data_directory}{test_filename}.mseed'

tr = load_mseed(mseed_file)
tr_times = tr.times()
tr_data = tr.data
longitud_tr_data = len(tr_data)

sampling_rate = calculate_sampling_rate(tr_times)
f, t, sxx_normalized = calculate_spectrogram(tr_data, sampling_rate)

temp_image_path = os.path.join(results_path, "temp_image.png")
plot_spectrogram(t, f, sxx_normalized, temp_image_path, w=size, h=size)

results = perform_detection(model, temp_image_path, save_dir=results_path)
results_path = get_latest_predict_folder()

detections_csv = os.path.join(results_path, "detections.csv")
x_coords_scaled = save_detection_results(results, longitud_tr_data, sampling_rate, detections_csv)

output_trace_path = os.path.join(results_path, "original_signal_detections.png")
plot_signal(tr_times, tr_data, output_trace_path, w=10, h=3, x_coords=x_coords_scaled, mseed_file=test_filename)

output = os.path.join(results_path, "Spectrogram_detections.png")
plot_spectrogram(t, f, sxx_normalized, output, labels=True, show_colorbar=True, x_coords=x_coords_scaled, cmap_name='jet')
    
results_path = "runs/detect"
temp_image_path = os.path.join(results_path, "temp_image.png")
os.remove(temp_image_path)