import os

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CatVsDogs:
    def __init__(self, data_dir: str = "data"):
        self.test_dir = f"{data_dir}/test"
        self.valid_dir = f"{data_dir}/valid"
        self.train_dir = f"{data_dir}/train"

        self.cats_test_dir = os.path.join(self.test_dir, 'cats')
        self.cats_valid_dir = os.path.join(self.valid_dir, 'cats')
        self.cats_train_dir = os.path.join(self.train_dir, 'cats')
        self.dogs_test_dir = os.path.join(self.test_dir, 'dogs')
        self.dogs_valid_dir = os.path.join(self.valid_dir, 'dogs')
        self.dogs_train_dir = os.path.join(self.train_dir, 'dogs')

        # Image generators
        # (provide iterators over image files and
        #  permits modifications on images, i.e. color in range 0-1 instead of 0-255)
        test_gen = ImageDataGenerator(rescale=1. / 255)
        train_gen = ImageDataGenerator(rescale=1. / 255,
                                       zoom_range=0.2,
                                       shear_range=0.2,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True)

        self.train_img_gen = train_gen.flow_from_directory(self.train_dir,
                                                           target_size=(150, 150),
                                                           batch_size=32,
                                                           class_mode='binary')

        self.valid_img_gen = test_gen.flow_from_directory(self.valid_dir,
                                                          target_size=(150, 150),
                                                          batch_size=32,
                                                          class_mode='binary')
        self.model = None
        self.history = None

    def load_model(self, model_path: str):
        self.model = models.load_model(model_path)
        return

    def save_model(self, model_path: str):
        self.model.save(model_path)

    def build_model(self):
        # Model building
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['acc'])
        self.model.summary()

    def start_training(self, epochs=30):
        if self.model is None:
            self.build_model()

        self.history = self.model.fit(self.train_img_gen,
                                      epochs=epochs,
                                      validation_data=self.valid_img_gen)

    def predict(self, img_data):
        self.model.predict(img_data)
