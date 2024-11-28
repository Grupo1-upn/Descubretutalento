const { PythonShell } = require('python-shell');
const path = require('path'); // Para trabajar con rutas absolutas

// Función para procesar las respuestas usando un modelo (ejemplo: ANN)
async function procesarRespuestas(intereses, aptitudes, modelo) {
    return new Promise((resolve, reject) => {
        const respuestas = { intereses, aptitudes };
        let scriptPath = '';

        // Definir qué modelo usar
        switch (modelo) {
            case 'ann':
                scriptPath = path.join(__dirname, '../algoritmos/predict_ann.py');
                break;
            case 'svm':
                scriptPath = path.join(__dirname, '../algoritmos/predict_svm.py');
                break;
            case 'randomforest':
                scriptPath = path.join(__dirname, '../algoritmos/predict_randomforest.py');
                break;
            default:
                return reject('Modelo no válido');
        }

        // Opciones para ejecutar el script de Python
        const options = {
            mode: 'text',
            pythonOptions: ['-u'],
            args: [JSON.stringify(respuestas)]
        };

        // Ejecutar el script de Python
        PythonShell.run(scriptPath, options, (err, results) => {
            if (err) {
                return reject(`Error ejecutando el modelo ${modelo}: ${err}`);
            }

            const orientacionVocacional = results ? results[0].trim() : null;
            if (!orientacionVocacional) {
                return reject('No se pudo obtener una predicción válida');
            }

            resolve(orientacionVocacional); // Devolver la predicción
        });
    });
}

module.exports = procesarRespuestas; // Exportar la función directamente
