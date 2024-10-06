document.getElementById('selectControls').addEventListener('submit', function (event) {
    // Ocultar el contenedor de resultados anteriores
    
    /*document.getElementById('resultsContainer').style.display = 'none';*/

    // Mostrar mensaje de cargando
    /*document.getElementById('modelOutput').innerText = 'Cargando Data...';*/

    // Realizar una solicitud AJAX a Flask para ejecutar el modelo
    fetch('/load_data', { method: 'POST', body: new FormData(this) } )
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Recargar la página para ver las imágenes actualizadas
                document.getElementById('resultText').innerText = "Resultado: " + data.result;
                                       
            } else {
                document.getElementById('modelOutput').innerText = 'Error al ejecutar el modelo: ' + data.error;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('modelOutput').innerText = 'Error al ejecutar el modelo.';
        });
});

