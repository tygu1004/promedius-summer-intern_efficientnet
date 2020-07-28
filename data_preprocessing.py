import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os


def read_csv(csv_file, ext, data_dir):
    print("Read CSV File...")
    df = pd.read_csv(csv_file)
    df = df.to_numpy()
    file_name, label = np.stack(df, axis=-1)

    for i in range(len(file_name)):
        file_name[i] = os.path.join(data_dir, file_name[i] + '.' + ext)

    file_name = tf.data.Dataset.from_tensor_slices(file_name)
    file_name = file_name.map(load_png)
    label = tf.data.Dataset.from_tensor_slices(label.astype(np.float32))

    return tf.data.Dataset.zip((file_name, label))


def load_png(fn):
    return tf.io.decode_png(contents=tf.io.read_file(filename=fn), channels=1, dtype=tf.uint8)


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--ext', dest='ext', default='png')
    parser.add_argument('--csv_file', dest='csv_file', required=True)
    parser.add_argument('--output_dir', dest='output_dir', required=True)
    parser.add_argument('--output_filename', dest='output_filename', default='png_label')
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = argument_parsing()
    tmp = read_csv(args.csv_file, args.ext, args.data_dir)

    COUNT = 100
    extension = ".tfrecord"
    file_number = 0
    file_name = os.path.join(args.output_dir, args.output_filename + str(file_number) + extension)
    writer = tf.io.TFRecordWriter(file_name)
    print("START WRITING FILE : " + str(file_name))
    for i, (png, label) in enumerate(tmp):
        png = tf.io.serialize_tensor(png)
        png = png.numpy()
        feature = {
              'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[png])),
              'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        if (i+1) % COUNT == 0:
            writer.close()
            file_number += 1
            file_name = os.path.join(args.output_dir, args.output_filename + str(file_number) + extension)
            writer = tf.io.TFRecordWriter(file_name)
            print('Create New TFRecord file : ' + str(file_name))
        writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    print("DONE")
