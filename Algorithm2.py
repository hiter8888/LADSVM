import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from keras.layers import Dense

def get_train():
    for i in range(28, 411):
        csvfile = 'logs/Training_csv/' + str(i) + '.csv'  # Log location
        df = pd.read_csv(csvfile)
        if i == 28:
            resultnp = convert_to_serial(df)
        else:
            resultnp = np.concatenate((resultnp, convert_to_serial(df)), axis=0)
    print(resultnp.shape)
    return resultnp

def get_test():
    for i in range(27):  # Adjusted to match the provided range
        csvfile = 'logs/Testing_csv/' + str(i) + '.csv'  # Log location
        df = pd.read_csv(csvfile)
        if i == 0:
            resultnp = convert_to_serial(df)
        else:
            resultnp = np.concatenate((resultnp, convert_to_serial(df)), axis=0)
    print(resultnp.shape)
    return resultnp

def convert_to_serial(dataframeobj):
    scaler = StandardScaler()
    new_data_frame = dataframeobj['Number'].to_numpy()
    new_data_frame = new_data_frame.reshape(-1, 1)
    data_scaled = scaler.fit_transform(new_data_frame)

    window_size = 10
    timesteps = 5

    X = []
    for i in range(len(data_scaled) - window_size - timesteps + 1):
        X.append(data_scaled[i:i+window_size])
    X = np.array(X)
    print(X.shape)
    return X

def build_lstm_autoencoder(window_size):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(window_size, 1), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(RepeatVector(window_size))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder_model(model, train_x, epochs=10, batch_size=64):
    model.fit(train_x, train_x, epochs=epochs, batch_size=batch_size)
    return Sequential(model.layers[:3])  # Encoder model

def train_svm_on_latent_representation(encoder, train_data):
    latent_representation = encoder.predict(train_data)
    data_reshaped = latent_representation.reshape(latent_representation.shape[0], -1)

    svm_model = OneClassSVM(kernel='rbf', nu=0.0000001)
    svm_model.fit(data_reshaped)
    return svm_model

def predict_and_evaluate(encoder, svm_model, test_data):
    new_latent_representation = encoder.predict(test_data)
    new_data_reshaped = new_latent_representation.reshape(new_latent_representation.shape[0], -1)
    predictions = svm_model.predict(new_data_reshaped)
    return predictions

def main():
    window_size = 10
    
    print("Preparing training data...")
    train_x = get_train()
    
    print("Building and training LSTM autoencoder...")
    autoencoder_model = build_lstm_autoencoder(window_size)
    encoder = train_autoencoder_model(autoencoder_model, train_x)
    
    print("Training SVM on latent representations...")
    svm_model = train_svm_on_latent_representation(encoder, train_x)
    
    print("Preparing test data...")
    test_x = get_test()
    
    print("Evaluating model on test data...")
    predictions = predict_and_evaluate(encoder, svm_model, test_x)
    
    print("train_x shape is:", train_x.shape)
    print("test_x shape is:", test_x.shape)
    print("Predictions:", predictions)

# Run the main function
if __name__ == "__main__":
    main()
