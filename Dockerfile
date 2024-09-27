# Utiliza una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo el archivo de dependencias para instalar los módulos
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r  requirements.txt

# Copiar el resto del código de la aplicación
COPY . /app

# Expone el puerto en el que Flask se ejecutará
EXPOSE 5000

# Comando que se ejecutará según la variable FLASK_ENV
CMD ["sh", "-c", "if [ '$FLASK_ENV' = 'development' ]; then flask run --host=0.0.0.0; else gunicorn -w 4 -b 0.0.0.0:5000 app:app; fi"]
