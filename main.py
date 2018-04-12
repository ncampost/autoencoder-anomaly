
import numpy as np
import math

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def main():
    # params
    periods = 16
    n_per_period = 60

    # Create base cosine wave.
    x = np.linspace(0, periods * 2*math.pi, periods * n_per_period)
    cosx = np.cos(x)

    # Add Gaussian noise to cosx.
    cosx += np.random.normal(scale=0.1, size=cosx.shape)

    # Make injected-signal wave
    x_signal = np.linspace(0, int(5/2. * 2)*math.pi, int(5/2. * n_per_period))

    # Senseless noise: exp. of higher-frequency wave, and one unusual high value
    signal = np.exp(np.sin(x_signal*6))
    cosx_signal = np.copy(cosx)
    cosx_signal[int(15/2.*n_per_period): 20/2*n_per_period] = signal
    cosx_signal[4*n_per_period] += 5

    # Define window size on cosx (How many numbers will the autoencoder see at a time to learn the shape?)
    window_size = 6

    # Format window data for the autoencoder
    cosx_windowed = np.empty((cosx.shape[0] - window_size + 1, window_size))
    cosx_signal_windowed = np.empty((cosx.shape[0] - window_size + 1, window_size))
    for i in range(cosx.shape[0] - window_size + 1):
        cosx_windowed[i] = cosx[i:i + window_size]
        cosx_signal_windowed[i] = cosx_signal[i:i + window_size]

    # Build autoencoder.
    input_shape = (window_size,)
    autoencoder = Sequential()
    autoencoder.add(Dense(window_size, activation='tanh', input_shape=input_shape))
    autoencoder.add(Dense(int(math.ceil(window_size/2.)), activation='tanh'))
    autoencoder.add(Dense(int(math.ceil(window_size/4.)), activation='tanh'))
    autoencoder.add(Dense(int(math.ceil(window_size/2.)), activation='tanh'))
    autoencoder.add(Dense(window_size, activation='tanh'))

    # Compile model and train.
    adadelta = optimizers.Adadelta()
    autoencoder.compile(optimizer=adadelta, loss='mean_squared_error')
    autoencoder.fit(cosx_windowed, cosx_windowed, epochs=250, batch_size=32)

    # Create signal reconstruction
    pred_windowed = autoencoder.predict(cosx_signal_windowed)

    # Dewindow the prediction
    cosx_signal_pred = np.copy(cosx)
    for i in range(cosx_signal_pred.shape[0]):
        if i % window_size == 0:
            cosx_signal_pred[i: i + window_size] = pred_windowed[i]

    # Plot noisy wave
    plt.figure(1)
    plt.plot(x, cosx, 'b-', linewidth=2.0, label="Training signal (noisy cos(x))")
    plt.savefig("figure1.png")

    # Plot signal injection and prediction
    plt.figure(2)
    plt.plot(x, cosx_signal_pred, 'r-', label="Autoencoder signal prediction")
    plt.plot(x, cosx_signal, 'g--', label="Signal-injected wave")
    plt.legend(loc='upper right')
    plt.savefig("figure2.png")

    # Calculate reconstruction error and time-average it over time_avg_size observations
    rec_error = np.sqrt(np.power(cosx_signal,2) + np.power(cosx_signal_pred,2))
    time_avg_size = 12
    time_rec_error = np.copy(rec_error)
    for i in range(time_rec_error.shape[0]):
        if i < time_avg_size:
            time_rec_error[i] = 0
        else:
            time_rec_error[i] = np.sum(rec_error[i - time_avg_size:i]) / time_avg_size
    
    # Plot reconstruction errors
    plt.figure(3)
    plt.plot(x, rec_error, 'b--', label="Reconstruction error")
    plt.plot(x, time_rec_error, 'r-', label="Time-averaged reconstruction error")
    plt.plot(x, np.full(x.shape, 1.35) , 'k-', label='Anomaly detection cutoff')
    plt.legend(loc='upper right')
    plt.savefig("figure3.png")

    plt.show()



if __name__ == '__main__':
    main()