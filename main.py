import tensorflow as tf
import argparse
import os
import data_loader as dl


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', dest='train_data_dir', required=True)
    parser.add_argument('--valid_data_dir', dest='valid_data_dir', required=True)
    parser.add_argument('--log_dir', dest='log_dir', required=True)
    parser.add_argument('--model_dir', dest='model_dir', required=True)

    parser.add_argument('--B', dest='B', type=int, required=True)
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=True)
    parser.add_argument('--epoch', dest='epoch', type=int, required=True)
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = argument_parsing()

    train_ls_ds = tf.data.Dataset.list_files(os.path.join(args.train_data_dir, "*.tfrecord"))
    validation_ls_ds = tf.data.Dataset.list_files(os.path.join(args.valid_data_dir, "*.tfrecord"))

    train_dataset = dl.load_data(train_ls_ds, args.batch_size)
    validation_dataset = dl.load_data(validation_ls_ds, batch_size=1)

    callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.log_dir,
        histogram_freq=10,  # How often to log histogram visualizations
        embeddings_freq=10,  # How often to log embedding visualizations
        write_images=True,
        update_freq="batch"
    )  # How often to write logs (default: once per epoch)

    if args.B == 7:
        a = tf.keras.applications.EfficientNetB7(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    elif args.B == 6:
        a = tf.keras.applications.EfficientNetB6(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    elif args.B == 5:
        a = tf.keras.applications.EfficientNetB5(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    elif args.B == 4:
        a = tf.keras.applications.EfficientNetB4(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    elif args.B == 3:
        a = tf.keras.applications.EfficientNetB3(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    elif args.B == 2:
        a = tf.keras.applications.EfficientNetB2(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    elif args.B == 1:
        a = tf.keras.applications.EfficientNetB1(include_top=True, weights=None, classes=2, classifier_activation='softmax')
    else:
        a = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=2, classifier_activation='softmax')

    a.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.BinaryCrossentropy()])
    a.fit(train_dataset, validation_data=validation_dataset, steps_per_epoch=10, validation_steps=10, callbacks=callback) #, epochs=args.epoch, shuffle=True)
    a.save(filepath=args.model_dir)
