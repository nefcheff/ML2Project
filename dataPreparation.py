import tensorflow as tf

def load_and_prepare_datasets(train_dir, test_dir, image_size=(224, 224), batch_size=32, seed=999):
    # Load the images from the directories
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=0.25,
        subset="training"
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=0.25,
        subset="validation"
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    # Add standardization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Normalize the data records
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Prefetching und Caching for efficiency
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset

def get_data_augmentation():
    # Define the data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    return data_augmentation