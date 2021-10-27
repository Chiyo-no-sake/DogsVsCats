import sys

from models.cat_vs_dog import CatVsDogs

data_dir = "data"
epochs = 30
batch = 32

if len(sys.argv) > 1:
    data_dir = sys.argv[1]

if len(sys.argv) > 2:
    epochs = int(sys.argv[2])

if len(sys.argv) > 3:
    batch = int(sys.argv[3])

instance = CatVsDogs(data_dir=data_dir, batch_size=batch)
instance.build_model()
instance.start_training(epochs=epochs)
instance.save_model("cat_vs_dogs.h5")
