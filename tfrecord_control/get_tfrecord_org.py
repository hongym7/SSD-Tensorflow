import tensorflow as tf
from PIL import Image
import io

tfrecord_filename = '/mnt/disk2/tmp_non_bbox_process/train-000-of-01024'
#tfrecord_filename = '/mnt/disk2/imagenet-data/train-00272-of-01024'
#tfrecord_filename = '../tfrecords/voc_2007_test_003.tfrecord'


def readRecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)



    print('sum : ', sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_filename)))


    # '''
    keys_to_features = {
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        # 'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        # Object boxes and classes.
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/area': tf.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.VarLenFeature(tf.int64),
        'image/object/difficult': tf.VarLenFeature(tf.int64),

        'image/class/label': tf.VarLenFeature(tf.int64),
        'image/object/bbox/label': tf.VarLenFeature(tf.int64),
        'image/object/bbox/label_text': tf.VarLenFeature(tf.string),




        # Instance masks and classes.
        # 'image/segmentation/object': tf.VarLenFeature(tf.int64),
        # 'image/segmentation/object/class': tf.VarLenFeature(tf.int64)
    }

    features = tf.parse_single_example(serialized_example, features=keys_to_features)

    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    filename = tf.cast(features['image/filename'], tf.string)
    source_id = tf.cast(features['image/source_id'], tf.string)
    encoded = tf.cast(features['image/encoded'], tf.string)
    image_format = tf.cast(features['image/format'], tf.string)
    xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
    label = tf.cast(features['image/class/label'], tf.int64)
    label_voc = tf.cast(features['image/object/bbox/label'], tf.int64)
    label_voc_text = tf.cast(features['image/object/bbox/label_text'], tf.string)







    # '''
    # height = tf.Variable(1.0)
    return height, width, filename, source_id, encoded, image_format, xmin, xmax, ymin, ymax, \
           label, label_voc, label_voc_text


def main():
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    height, width, filename, source_id, encoded, image_format, \
    x_min, x_max, y_min, y_max, label, label_voc, label_voc_text = readRecord(filename_queue)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        vheight, vwidth, vfilename, vsource_id, vencoded, vimage_format, vx_min, vx_max, vy_min, vy_max, vlabel, vlabel_voc, vlabel_voc_text = sess.run(
            [height, width, filename, source_id, encoded, image_format, x_min, x_max, y_min, y_max, label, label_voc, label_voc_text])
        print( vheight, vwidth, vfilename, vsource_id, vimage_format, vx_min, vx_max, vy_min, vy_max, vlabel, vlabel_voc, vlabel_voc_text)
        image = Image.open(io.BytesIO(vencoded))
        image.show()

        coord.request_stop()
        coord.join(threads)




main()