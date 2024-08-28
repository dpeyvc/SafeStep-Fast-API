import tensorflowjs as tfjs
from tensorflow import keras

def save():
    model=keras.models.load_model('cnn_har_model.keras')
    tf.loadGraphModel(model,'cnn_har_model')