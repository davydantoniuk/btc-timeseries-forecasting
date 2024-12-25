import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

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

def adf_test(series,title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') 
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

def kpss_test(series, title=''):
    print(f'Kwiatkowski-Phillips-Schmidt-Shin Test: {title}')
    result = kpss(series.dropna(), regression='c', nlags="auto")
    
    labels = ['KPSS test statistic', 'p-value', '# lags used']
    out = pd.Series(result[0:3], index=labels)

    for key, val in result[3].items():
        out[f'critical value ({key})'] = val
        
    print(out.to_string())        
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has no unit root and is stationary")

from tensorflow.keras import layers
def get_ensemble_models(horizon, 
                        train_data,
                        test_data,
                        num_iter=10, 
                        num_epochs=100, 
                        loss_fns=["mae", "mse", "mape"]):

  ensemble_models = []

  for i in range(num_iter):
    for loss_function in loss_fns:
      print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")
      model = tf.keras.Sequential([
        layers.Dense(128, kernel_initializer="he_normal", activation="relu"), 
        layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
        layers.Dense(horizon)                                 
      ])

      model.compile(loss=loss_function,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["mae", "mse"])

      model.fit(train_data,
                epochs=num_epochs,
                verbose=0,
                validation_data=test_data,
                # Add callbacks to prevent training from going/stalling for too long
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=200,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                patience=100,
                                                                verbose=1)])

      ensemble_models.append(model)

  return ensemble_models 

# Function which uses a list of trained models to make and return a list of predictions
def make_ensemble_preds(ensemble_models, data):
  ensemble_preds = []
  for model in ensemble_models:
    preds = model.predict(data, verbose=0) 
    ensemble_preds.append(preds)
  return tf.constant(tf.squeeze(ensemble_preds))

# Find upper and lower bounds of ensemble predictions
def get_upper_lower(preds): 
  # 1. Measure the standard deviation of the predictions
  std = tf.math.reduce_std(preds, axis=0)
  # 2. Multiply the standard deviation by 1.96
  interval = 1.96 * std 
  # 3. Get the prediction interval upper and lower bounds
  preds_mean = tf.reduce_mean(preds, axis=0)
  lower, upper = preds_mean - interval, preds_mean + interval
  return lower, upper