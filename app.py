import os
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from obspy import read
from scipy import signal
from matplotlib import cm
from flask import Flask, render_template, jsonify
from matplotlib.animation import FuncAnimation

# Cambiar el backend de Matplotlib para evitar errores relacionados con la interfaz gráfica
plt.switch_backend('Agg')

model_path = 'static/models/train300.pt'
# Cargar el modelo YOLO entrenado
#model = YOLO(model_path)

 # Archivo MSEED de ejemplo
test_filename = 'XB.ELYSE.02.BHV.2022-01-02HR04_evid0006'
data_directory = 'space_apps_2024_seismic_detection/data/mars/training/data/'
mseed_file = f'{data_directory}{test_filename}.mseed'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/dashboard')
def graphs():
    return render_template("dashboard.html")

'''
@app.route('/graphs')
def graphs():
    # Renderizar la plantilla `results.html`
    return render_template("results.html", results={})
'''
    
@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        # Procesar el archivo MSEED y generar resultados
        tr = cargar_mseed(mseed_file)
        tr_times = tr.times()
        tr_data = tr.data
        longitud_tr_data = len(tr_data)

        # Calcular la frecuencia de muestreo
        sampling_rate = calcular_sampling_rate(tr_times)

        results_path = obtener_ultima_carpeta_predict()
        temp_image_path = os.path.join(results_path, "temp_image.png")

        model = YOLO(model_path)
        # Realizar la detección en el espectrograma generado
        results = realizar_deteccion(model, temp_image_path, save_dir=results_path)

        # Guardar resultados de detección en un archivo CSV
        detections_csv = os.path.join(results_path, "detections_x_coords.csv")
        x_coords_scaled = guardar_resultados_deteccion(results, longitud_tr_data, sampling_rate, detections_csv)

        output_trace_path = os.path.join(results_path, "original_signal_with_detections.png")
        # Guardar gráfica con detecciones
        graficar_signal(tr_times, tr_data, output_trace_path, x_coords=x_coords_scaled, mseed_file=mseed_file)
        
        # Calcular el espectrograma y guardarlo como imagen temporal
        f, t_spec, sxx_normalized = calcular_spectrograma(tr_data, sampling_rate)
         # Ruta para guardar el espectrograma con las detecciones
        temp_image_path = os.path.join(results_path, "espectograma.png")
        # Crear la gráfica del espectrograma
        graficar_espectrograma(t_spec, f, sxx_normalized, temp_image_path, labels_active = True, show_colorbar=True, x_coords=x_coords_scaled, cmap_name='jet')

        # Resultados a devolver en la respuesta JSON
        response = {
            'success': True,
            'spectrogram_image': os.path.relpath(temp_image_path, 'static'),
            'detections_csv': os.path.relpath(detections_csv, 'static'),
            'output_trace_image': os.path.relpath(output_trace_path, 'static')
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_data', methods=['POST'])
def load_data():
    try:
        nombre_modelo = os.path.basename(model_path)
        # Verificar si el nombre del modelo contiene '1024' al final (antes de la extensión)
        if nombre_modelo.endswith('1024.pt'):
            size=10.24
        else:
            size=6.4

         # Definir la ruta de la carpeta predict dentro de static/images/detect
        predict_dir = os.path.join("static", "images", "detect", "predict")

        try:
            # Eliminar el contenido de la carpeta `predict` sin eliminar la carpeta en sí
            for filename in os.listdir(predict_dir):
                file_path = os.path.join(predict_dir, filename)
                
                # Verificar si es un archivo o una subcarpeta y eliminarlo
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Eliminar archivos o enlaces simbólicos
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Eliminar subcarpetas
                
        except Exception as e:
            print(f"Error al vaciar la carpeta '{predict_dir}': {e}")

        # Procesar el archivo MSEED y generar resultados
        tr = cargar_mseed(mseed_file)
        tr_times = tr.times()
        tr_data = tr.data

        # Calcular la frecuencia de muestreo
        sampling_rate = calcular_sampling_rate(tr_times)

        results_path = obtener_ultima_carpeta_predict()

        # -------------------------------------------- #
        # Guardar gráfica con detecciones
        output_trace_path = os.path.join(results_path, "original_signal_with_detections.png")
        graficar_signal(tr_times, tr_data, output_trace_path, x_coords=None, mseed_file=mseed_file)
        #-----------------------------------#


        # Calcular el espectrograma y guardarlo como imagen temporal
        f, t_spec, sxx_normalized = calcular_spectrograma(tr_data, sampling_rate)

        save_path = os.path.join(results_path, "temp_image.png")
        graficar_espectrograma(t_spec, f, sxx_normalized, save_path, size, size, labels_active = False, show_colorbar=False, x_coords=None)

        # Ruta para guardar el espectrograma con las detecciones
        temp_image_path = os.path.join(results_path, "espectograma.png")

        # Crear la gráfica del espectrograma
        graficar_espectrograma(t_spec, f, sxx_normalized, temp_image_path, labels_active = True, show_colorbar=True, x_coords=None, cmap_name='jet')

        # Resultados a devolver en la respuesta JSON
        response = {
            'success': True,
            'spectrogram_image': os.path.relpath(temp_image_path, 'static')
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----------------- FUNCIONES ----------------- #

def cargar_mseed(file_path):
    """Carga el archivo MSEED y retorna la traza principal."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    try:
        st = read(file_path)
        if len(st) == 0:
            raise ValueError(f"El archivo {file_path} no contiene datos.")
        return st[0]  # Retornar solo la primera traza si existen múltiples
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo MSEED: {e}")


def calcular_sampling_rate(tr_times):
    """Calcula la frecuencia de muestreo basada en las diferencias de tiempo."""
    return 1 / np.mean(np.diff(tr_times))

def calcular_spectrograma(tr_data, sampling_rate):
    """Calcula el espectrograma de la señal usando scipy."""
    f, t, sxx = signal.spectrogram(tr_data, sampling_rate)
    vmax = np.percentile(sxx, 99)
    sxx_normalized = sxx / vmax  # Normalizar el espectrograma
    return f, t, sxx_normalized

def graficar_espectrograma(t, f, sxx_normalized, save_path, w=10, h=3, labels_active=False, show_colorbar=False, x_coords=None, cmap_name='jet'):
    """Grafica y guarda el espectrograma."""
    # Asegurar que el directorio exista antes de guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)

    # Crear el espectrograma con el mapa de colores especificado
    cmap = ax.pcolormesh(t, f, sxx_normalized, cmap=cmap_name, vmax=1)

    # Mostrar la barra de colores si show_colorbar es True
    if show_colorbar:
        fig.colorbar(cmap, ax=ax)

    # Inicializar lista de líneas y etiquetas para la leyenda
    lines = []
    labels = []

    # Añadir detecciones si se proporcionan las coordenadas x
    if x_coords is not None:
        for x in x_coords:
            # Crear la línea roja para `Abs. Arrival`
            line = ax.axvline(x=x, color='red', linestyle='--', linewidth=2, label='Predicted Arrival')
            ax.text(x, max(f)-4, f'x = {x:.2f}', color='red', fontsize=12, verticalalignment='bottom')

            # Añadir la línea y la etiqueta a las listas
            lines.append(line)
            labels.append('Predicted Arrival')

        if 'training' in data_directory:
            print("data_directory: ", data_directory)
            # Reemplazar '/data/' con '/catalog/' en el directorio
            data_catalog = 'space_apps_2024_seismic_detection/data/catalog/'

            # Definir la ruta del archivo catalog.csv en el nuevo directorio
            catalog_file = os.path.join(data_catalog, 'catalog.csv')

            try:
                # Leer el archivo catalog.csv
                catalog_df = pd.read_csv(catalog_file)

                # Obtener solo el nombre base del archivo mseed sin extensión
                mseed_filename = os.path.basename(mseed_file).replace('.mseed', '')

                # Filtrar el DataFrame para encontrar el `time_rel(sec)` correspondiente al `mseed_filename`
                matched_row = catalog_df[catalog_df['filename'] == mseed_filename]

                # Si se encuentra una coincidencia, obtener el valor de `time_rel(sec)` y graficarlo
                if not matched_row.empty:
                    time_rel_value = matched_row['time_rel(sec)'].values[0]
                    # Crear la línea verde para `time_rel`
                    line = ax.axvline(x=time_rel_value, c='green', linestyle='-', linewidth=2, label='Expected time_rel')
                    ax.text(time_rel_value, max(f)-3, f'x = {time_rel_value:.2f}', color='green', fontsize=12, verticalalignment='bottom')

                    # Añadir la línea y la etiqueta a las listas
                    lines.append(line)
                    labels.append('Expected time_rel')
                else:
                    print(f"No se encontró el filename '{mseed_filename}' en el archivo catalog.csv.")
            except FileNotFoundError:
                print(f"El archivo {catalog_file} no fue encontrado.")
            except Exception as e:
                print(f"Error al procesar el archivo {catalog_file}: {e}")

    # Añadir las etiquetas de las líneas a la leyenda
    ax.legend(lines, labels)

    if labels_active:
        # Configurar los límites y etiquetas
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


def graficar_signal(times, data, save_path, w=10, h=3, x_coords=None, mseed_file=""):
    global data_directory

    # Asegurar que el directorio exista antes de guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.plot(times, data, label='Original Signal')

    # Inicializar listas para las líneas y etiquetas de la leyenda
    lines = []
    labels = []

    # Verificar si `training` está en data_directory
    if 'training' in data_directory:
        print("data_directory: ", data_directory)
        # Reemplazar '/data/' con '/catalog/' en el directorio
        data_catalog = 'space_apps_2024_seismic_detection/data/catalog/'
        
        # Definir la ruta del archivo catalog.csv en el nuevo directorio
        catalog_file = os.path.join(data_catalog, 'catalog.csv')

        try:
            # Leer el archivo catalog.csv
            catalog_df = pd.read_csv(catalog_file)

            # Obtener solo el nombre base del archivo mseed sin extensión
            mseed_filename = os.path.basename(mseed_file).replace('.mseed', '')

            # Filtrar el DataFrame para encontrar el `time_rel(sec)` correspondiente al `mseed_filename`
            matched_row = catalog_df[catalog_df['filename'] == mseed_filename]

            # Si se encuentra una coincidencia, obtener el valor de `time_rel(sec)` y graficarlo
            if not matched_row.empty:
                time_rel_value = matched_row['time_rel(sec)'].values[0]
                
                # Crear la línea verde para `time_rel`
                line = ax.axvline(x=time_rel_value, c='green', linestyle='-', linewidth=2, label='Expected time_rel')
                ax.text(time_rel_value, max(data)*0.6, f'x = {time_rel_value:.2f}', color='green', fontsize=12, verticalalignment='bottom')

                # Añadir la línea y la etiqueta a las listas
                lines.append(line)
                labels.append('Expected time_rel')
            else:
                print(f"No se encontró el filename '{mseed_filename}' en el archivo catalog.csv.")
        except FileNotFoundError:
            print(f"El archivo {catalog_file} no fue encontrado.")
        except Exception as e:
            print(f"Error al procesar el archivo {catalog_file}: {e}")

    # Graficar detecciones si se proporcionan las coordenadas x
    if x_coords is not None:
        for x in x_coords:
            # Crear la línea roja para `Abs. Arrival`
            line = ax.axvline(x=x, c='red', linestyle='--', linewidth=2, label='Predicted Arrival')
            ax.text(x, max(data)*0.5, f'x = {x:.2f}', color='red', fontsize=12, verticalalignment='bottom')

            # Añadir la línea y la etiqueta a las listas
            lines.append(line)
            labels.append('Predicted Arrival')

    # Agregar las líneas y etiquetas a la leyenda de la gráfica
    ax.legend(lines, labels)

    # Configurar límites y etiquetas
    ax.set_xlim([min(times), max(times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{os.path.basename(mseed_file)}', fontweight='bold')

    # Guardar la figura
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def realizar_deteccion(model, source_image, save_dir):
    """Realiza la detección usando YOLO y retorna las coordenadas de detección."""
    os.makedirs(save_dir, exist_ok=True)
    results = model.predict(
        source=source_image,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,  
        name='',         
        exist_ok=True     
    )
    return results

def guardar_resultados_deteccion(results, tr_data_len, sampling_rate, output_path):
    """Guarda los resultados de la detección y calcula las coordenadas escaladas."""
    x_coords, probs, x_coords_scaled = [], [], []

    for result in results:
        for box in result.boxes:
            x1 = int(box.xyxy[0][0])
            x_coords.append(x1)
            probs.append(float(box.conf[0]))

    x_coords_scaled = [(x * tr_data_len) / (640 * sampling_rate) for x in x_coords]

    df = pd.DataFrame({'x_coords': x_coords, 'probability': probs, 'x_coords_scaled': x_coords_scaled})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    df.to_csv(output_path, index=False)
    return x_coords_scaled

def obtener_ultima_carpeta_predict(base_dir="static/images/detect"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "predict")
    
    predict_dirs = [d for d in os.listdir(base_dir) if re.match(r'predict\d+', d)]
    if not predict_dirs:
        return os.path.join(base_dir, "predict")
    
    max_predict = max([int(re.findall(r'\d+', d)[0]) for d in predict_dirs])
    return os.path.join(base_dir, f"predict{max_predict}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')