# Usar una imagen oficial de Python como base
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r  requirements.txt

# Copiar los archivos de la aplicación a la imagen del contenedor
COPY . /app

# Exponer el puerto 5000 en el contenedor
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]
