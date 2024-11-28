const mongoose = require('mongoose');
require('dotenv').config();

const connectDB = async () => {
  try {
    console.log('MONGODB_URI:', process.env.MONGODB_URI); // Verifica si se imprime el URI correctamente

    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost/TestVocacional');
    
    console.log('MongoDB connected');
  } catch (error) {
    console.error('Error connecting to MongoDB', error);
    process.exit(1);
  }
};

module.exports = connectDB;
