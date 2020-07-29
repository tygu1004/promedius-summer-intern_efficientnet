import tensorflow as tf

description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }


def load_data(file_path, batch_size):
    dataset = tf.data.TFRecordDataset(file_path)

    def _parse_function(proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(proto, description)

    def _png_to_numpy(it):
        it['image'] = tf.io.parse_tensor(it['image'], tf.uint8)
        return it['image'], it['label']

    dataset = dataset.map(_parse_function)
    dataset = dataset.map(_png_to_numpy)
    dataset = dataset.batch(batch_size)
    return dataset
