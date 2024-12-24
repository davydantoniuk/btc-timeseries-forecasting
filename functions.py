import matplotlib.pyplot as plt
import tensorflow as tf
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