import sys

from models.cat_vs_dog import CatVsDogs

data_dir = "data"
epochs = 30

if len(sys.argv) > 1:
    data_dir = sys.argv[1]

if len(sys.argv) > 2:
    epochs = int(sys.argv[2])

instance = CatVsDogs(data_dir=data_dir)
instance.build_model()
instance.start_training(epochs=epochs)
instance.save_model("cat_vs_dogs.h5")
