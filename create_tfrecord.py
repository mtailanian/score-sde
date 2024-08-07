import glob
import tensorflow as tf
from pathlib import Path

category = "grid"
data_dir = Path("/home/data/tai/data/mvtec")
out_tfrecord_path = str(data_dir / "tfrecords" / f"{category}.tfrecord")


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [256, 256])
  image /= 255.0  # normalize to [0,1] range
  img = image * 2. - 1.  # normalize to [-1, 1] range
  return dict(image=img, label=None)

# 2) load a directory of images without label 
all_image_paths = list((data_dir / category).glob('test/*/*.png'))
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter(out_tfrecord_path)
dataset_builder = tf.data.TFRecordDataset(out_tfrecord_path)
train_split_name = eval_split_name = 'train'

# 3) output 
# ds = dataset_builder.with_options(dataset_options)
# ds = ds.repeat(count=num_epochs)
# ds = ds.shuffle(shuffle_buffer_size)
# ds = ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds = ds.batch(batch_size, drop_remainder=True)
#
# return ds.prefetch(prefetch_size)