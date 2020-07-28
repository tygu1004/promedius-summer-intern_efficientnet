'''
print(parsed_image_dataset)
i = 0
for image_feat in parsed_image_dataset:
    img = tf.io.parse_tensor(image_feat['image'], tf.uint8).numpy()
    img = tf.image.encode_png(img, compression=1)

    tf.io.write_file("./data/Domain_P/tfr/co" + str(i) + ".png", img)
    i += 1
'''

import tensorflow as tf
import argparse
import os
import data_loader as dl


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--log_dir', dest='log_dir', required=True)
    parser.add_argument('--model_dir', dest='model_dir', required=True)
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = argument_parsing()

    ls_ds = tf.data.Dataset.list_files(os.path.join(args.data_dir, "*.tfrecord"))

    dataset = dl.load_data(ls_ds)

    callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.log_dir,
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=0,  # How often to log embedding visualizations
        update_freq="epoch",
    )  # How often to write logs (default: once per epoch)

    #    for i in dataset:
        # img = tf.io.parse_tensor(i['image'], tf.uint8).numpy().shape
        # img = tf.image.encode_png(img, compression=1)
#        print(i)
    a = tf.keras.applications.EfficientNetB0(weights=None, classes=2)
    # a.summary()
    a.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    a.fit(dataset, callbacks=callback, steps_per_epoch=20)
