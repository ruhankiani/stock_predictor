from model import build_model, training_model
from prepare_data import x_train, y_train, x_test, y_test, scaler, update
import tensorflow as tf
import matplotlib.pyplot as plt
import os

if os.path.exists('predicting_model.keras'):
    train_model = input("Choose whether to use saved model or train a new model (1/2)\n"\
                        "1. Use saved model \n"\
                        "2. Train new model \n")
else:
    train_model = 2

if train_model == 2:
    model = build_model()
    training_model(model, x_train, y_train)
    model.save('predicting_model.keras')
    print("Model saved")
else:
    model = tf.keras.models.load_model('predicting_model.keras')


# test
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# plot
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='originial price')
plt.plot(y_predicted, 'r', label = 'predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
