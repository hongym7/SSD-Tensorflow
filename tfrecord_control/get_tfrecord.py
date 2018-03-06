import tensorflow as tf


def readRecord(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    keys_to_features = {

        'image/height': tf.FixedLenFeature((), tf.int64, 1),

        'image/width': tf.FixedLenFeature((), tf.int64, 1),

        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),

        # 'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),

        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),

        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),

        'image/object/bbox/xmin': tf.FixedLenFeature((), tf.float32, 1.0),
    }

    features = tf.parse_single_example(serialized_example, features=keys_to_features)

    #print('features ', features)

    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    filename = tf.cast(features['image/filename'], tf.string)
    source_id = tf.cast(features['image/source_id'], tf.string)
    encoded = tf.cast(features['image/encoded'], tf.string)
    xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)

    return height





filename_queue = tf.train.string_input_producer(['../tfrecords/voc_2007_test_000.tfrecord'])
result = readRecord(filename_queue)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print('...')
    print(sess.run(result))
    print('...')

