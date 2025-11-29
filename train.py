from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

train_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory(
    "data/train",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_directory(
    "data/validation",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=5, validation_data=val_data)
model.save("cat_dog_model.h5")


