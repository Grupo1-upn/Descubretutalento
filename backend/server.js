const express = require('express');
const connectDB = require('./config/basedatos');
const Resultados = require('./modelos/ResultadosTest'); // Modelo de Test
const nodemailer = require('nodemailer');
const procesarRespuestas = require('./controllers/procesarRespuestas');
require('dotenv').config();


console.log('Email:', process.env.EMAIL);
console.log('Password:', process.env.PASSWORD);

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(express.json()); // Para analizar solicitudes JSON

// Conectar a la base de datos
connectDB();

// Ruta para recibir los resultados del test
app.post('/api/enviar_respuestas', async (req, res) => {
    try {
        const { intereses, aptitudes, correo } = req.body; // Obtener datos del cuerpo de la solicitud

        // Llamar a la función para procesar las respuestas con el modelo ANN
        const orientacionVocacional = await procesarRespuestas(intereses, aptitudes, 'svm'); // Colocar algoritmo

        // Guardar las respuestas y la predicción en la base de datos
        const nuevoResultado = new Resultados({
            correo: correo,
            respuestas: { intereses, aptitudes },
            orientacion_vocacional: orientacionVocacional, // Resultado de la predicción
            fecha: new Date(),
        });

        await nuevoResultado.save(); // Guardar en la base de datos

        // Enviar resultados por correo
        await enviarCorreo(correo, orientacionVocacional);

        res.status(200).json({ message: 'Respuestas procesadas y correo enviado' });
    } catch (error) {
        console.error('Error procesando los resultados:', error);
        res.status(500).json({ error: 'Error procesando los resultados' });
    }
});

// Función para enviar el correo
async function enviarCorreo(correo, resultado) {
  const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
          user: process.env.EMAIL, // Tu correo
          pass: process.env.PASSWORD, // Tu contraseña
      },
  });

  const mailOptions = {
      from: process.env.EMAIL,
      to: correo,
      subject: 'Resultados de tu Test Vocacional',
      text: `¡Hola! Estos son los resultados de tu test vocacional:\n\n${resultado}`,
  };

  await transporter.sendMail(mailOptions);
  console.log('Correo enviado a:', correo);
}

app.use(express.static('public')); // Asegúrate de que tus archivos HTML, CSS, JS estén en la carpeta 'public'



// Iniciar el servidor
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});


