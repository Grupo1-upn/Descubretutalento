const mongoose = require('mongoose');

// Definir el esquema para las respuestas del test
const ResultadosTestSchema = new mongoose.Schema({
    correo: {
        type: String,
        required: true,
    },
    respuestas: {
        intereses: {
            type: Array,
            required: true,
        },
        aptitudes: {
            type: Array,
            required: true,
        },
    },
    orientacion_vocacional: {
        type: String, // Almacena el resultado del modelo
        required: true,
    },
    fecha: {
        type: Date,
        default: Date.now,
    },
});

// Crear y exportar el modelo
module.exports = mongoose.model('Resultados', ResultadosTestSchema);
