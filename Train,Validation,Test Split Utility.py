import tensorflow as tf
import os


# ================================
# Local Dataset Path
# ================================
DATA_DIR = r"K:\Tensorflow\datasets\cats_dogs\PetImages"


# ================================
# Dataset Parameters
# ================================
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 123


# ================================
# Create Train / Val / Test Split
# ================================
def create_datasets(data_dir):

    # 70% Training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Remaining 30%
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Split 30% → 15% Val + 15% Test
    val_batches = int(0.5 * len(val_test_ds))

    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    return train_ds, val_ds, test_ds


# ================================
# Load Datasets
# ================================
train_ds, val_ds, test_ds = create_datasets(DATA_DIR)


# ================================
# Print Sizes
# ================================
print("Train batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Test batches:", len(test_ds))
