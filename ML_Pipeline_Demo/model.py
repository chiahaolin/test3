# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Train data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Test data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Prediction")
  # Show the legend
  plt.legend(shadow='True')
  # Set grids
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis', family='Arial', fontsize=11)
  plt.ylabel('Y axis', family='Arial', fontsize=11)
  # Show
  plt.savefig('model_results.png', dpi=120)



def MeanAbsoluteError(b_test, b_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(b_test, b_pred)
  

def MeanSquareError(b_test, b_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.metrics.mean_squared_error(b_test, b_pred)


# Check Tensorflow version
print(tf.__version__)


# Create features
a = np.arange(1, 60, 4)

# Create labels
b = np.arange(1, 60, 4)


# Split data into train and test sets
a_train = a[:10] # first 40 examples (80% of data)
b_train = b[:10]

a_test = a[10:] # last 10 examples (20% of data)
b_test = b[10:]

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
    ])

# Compile the model
model.compile(loss =MeanAbsoluteError,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ['MeanAbsoluteError'])

# Fit the model
#model.fit(tf.expand_dims(a_train, axis=1), b_train, epochs=100)
model.fit(a_train, b_train, epochs=100)


# Make and plot predictions for model_1
b_preds = model.predict(a_test)
plot_predictions(train_data=a_train, train_labels=b_train,  test_data=a_test, test_labels=b_test,  predictions=b_preds)


# Calculate model_1 metrics
MeanAbsoluteError_1 = np.round(float(MeanAbsoluteError(b_test, b_preds.squeeze()).numpy()), 2)
MeanSquareError_1 = np.round(float(MeanSquareError(b_test, b_preds.squeeze()).numpy()), 2)
print(f'\nMean Absolute Error = {MeanAbsoluteError_1}, Mean Squared Error = {MeanSquareError_1}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {MeanAbsoluteError_1}, Mean Squared Error = {MeanSquareError_1}.')
