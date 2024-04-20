import gc
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import (
    read_csv,
    get_column,
    get_data,
    prepare_data, ISIZE,
)


def main():

    # ========================= PROCESSING
    train_df = read_csv('trainx.csv')
    test_df = read_csv('valx.csv')

    train_paths = get_column(train_df, 0)
    test_paths = get_column(test_df, 0)

    train_images_dict = get_data(train_paths)
    test_images_dict = get_data(test_paths)

    gc.collect()

    train_images, train_labels = prepare_data(train_images_dict)
    test_images, test_labels = prepare_data(test_images_dict)

    del train_df, test_df, train_paths, test_paths, train_images_dict, test_images_dict

    # ========================= MODEL
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(ISIZE, ISIZE, 1), kernel_regularizer=regularizers.l1(0.02)),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (4, 4), activation='relu', kernel_regularizer=regularizers.l1(0.03)),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(1024, (4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.03)),
        Dropout(0.6),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.6),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('./models/model-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True,
                                 mode='auto')

    model.fit(
        x=train_images,
        y=train_labels,
        batch_size=4,
        epochs=56,
        shuffle=True,
        validation_data=(test_images, test_labels),
        callbacks=[checkpoint]
    )

    model.save('models/beans.h5')


if __name__ == "__main__":
    main()
