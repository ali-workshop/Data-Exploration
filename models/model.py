from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    def __init__(self):
        pass
    
    def build_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        return model
    
    def compile_model(self, model):
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def fit_model(self, model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        return history
