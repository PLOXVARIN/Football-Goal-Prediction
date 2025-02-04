import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# โหลดและเตรียมข้อมูล
df = pd.read_csv('data/dataset.csv')
X = df[['distance', 'angle', 'speed', 'skill']].values
y = df['goal'].values

# Normalize ข้อมูล (0-1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งข้อมูลฝึกและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# สร้างโมเดล Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ฝึกโมเดล
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_split=0.2,
                    verbose=0)

# บันทึกโมเดลและ Scaler
model.save('model.h5')
np.save('scaler.npy', scaler.scale_)
np.save('min.npy', scaler.min_)