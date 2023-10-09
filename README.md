# Backpropogation.
import numpy as np 
 
# Generate a synthetic dataset for age classification 
num_samples = 1000 
num_features = 5  # Assuming 5 features 
num_classes = 3  # Assuming 3 age classes 
X = np.random.rand(num_samples, num_features)  # Random feature values between 0 and 1 
y = np.random.randint(0, num_classes, size=num_samples)  # Random age classes 
# Manually split the data into training, validation, and test sets 
train_ratio = 0.7 val_ratio = 0.15 
test_ratio = 0.15 
num_train = int(train_ratio * num_samples) 
num_val = int(val_ratio * num_samples) 
X_train = X[:num_train] 
y_train = y[:num_train] 
X_val = X[num_train:num_train+num_val] 
y_val = y[num_train:num_train+num_val] X_test = X[num_train+num_val:] 
y_test = y[num_train+num_val:] 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
# Create a simple neural network model for classification 
model = Sequential([ 
    Dense(64, activation='relu', input_shape=(num_features,)),     Dense(32, activation='relu'), 
    Dense(num_classes, activation='softmax')  # Use softmax activation for classification 
]) 
 
14 
 
# Compile the model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
# Train the model using backpropagation 
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val)) # Evaluate the model 
loss, accuracy = model.evaluate(X_test, y_test) 
# Use the trained model for predictions 
predictions = model.predict(X_test) 
# ... (previous code remains the same) 
# Train the model using backpropagation 
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val)) 
# Evaluate the model loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Accuracy on Test Set: {accuracy}') 
# Use the trained model for predictions 
predictions = model.predict(X_test) 
# Convert predictions to class labels 
predicted_classes = np.argmax(predictions, axis=1) 
# Print some example predictions and their corresponding true labels 
for i in range(5):  # Print predictions for the first 5 samples  print(f"Predicted Class: {predicted_classes[i]}, True Class: {y_test[i]}") 
