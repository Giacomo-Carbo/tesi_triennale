import tensorflowjs as tfjs
from tensorflow import keras

# carico il modello 
model = keras.models.load_model("final_lstm_encoder.h5", compile=False)

# esporta in formato TensorFlow.js
tfjs.converters.save_keras_model(model, "modeljs")