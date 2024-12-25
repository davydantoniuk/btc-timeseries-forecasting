import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14) 
  plt.grid(True)

def evaluate_preds(y_true, y_pred):
    def mean_absolute_scaled_error(y_true, y_pred):
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) 
        return mae / mae_naive_no_season
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    mae = tf.keras.metrics.MeanAbsoluteError()
    mae.update_state(y_true, y_pred)
    mse = tf.keras.metrics.MeanSquaredError()
    mse.update_state(y_true, y_pred)
    rmse = tf.sqrt(mse.result())
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
    mape.update_state(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    return {"mae": mae.result().numpy(),
            "mse": mse.result().numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.result().numpy(),
            "mase": mase.numpy()}

def make_windows(x, window_size=7, horizon=1):
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
    windowed_array = x[window_indexes]

    def get_labelled_windows(x, horizon=1):
        return x[:, :-horizon], x[:, -horizon:]
    
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels

def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1 - test_split))
    train_windows, test_windows = windows[:split_size], windows[split_size:]
    train_labels, test_labels = labels[:split_size], labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels

def crteate_model_checkpoint(model_name, save_path="models"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name + ".keras"),
                                              verbose=0,
                                              save_best_only=True)

def make_preds(model, input_data):
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)