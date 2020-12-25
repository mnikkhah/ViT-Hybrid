import tensorflow as tf
import tensorflow_datasets as tfds



class Dataset:
    def __init__(self, dataset, train_batch_size, test_batch_size, size):
        (self.train_ds, self.test_ds), self.ds_info = tfds.load(dataset,
                    split=['train', 'test'], shuffle_files=False,
                    with_info=True, as_supervised=True, data_dir="./")
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.size = size
        # print(self.ds_info)
        # print(self.ds_info.features["label"].num_classes)

        self.prepare_train_set()
        self.prepare_test_set()

    def prepare_train_set(self):
        self.train_ds = self.train_ds.map(
            self.normalize_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_ds = self.train_ds.cache()
        self.train_ds = self.train_ds.shuffle(self.ds_info.splits['train'].num_examples)
        self.train_ds = self.train_ds.batch(self.train_batch_size)
        self.train_ds = self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)


    def prepare_test_set(self):
        self.test_ds = self.test_ds.map(
            self.normalize_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_ds = self.test_ds.batch(self.test_batch_size)
        self.test_ds = self.test_ds.cache()
        # self.test_ds = self.test_ds.shuffle(self.ds_info.splits['test'].num_examples)
        self.test_ds = self.test_ds.prefetch(tf.data.experimental.AUTOTUNE)


    def normalize_resize(self, image, label):
        """ Normalize Images uint8 --> float32
            Between [0, 1]
            resized to (self.size, self.size, 3)"""
        image = tf.cast(image, tf.float32)#/255.
        label = tf.one_hot(label, self.ds_info.features["label"].num_classes)
        # label = tf.cast(label, tf.float32)
        image = tf.image.resize(
            image, self.size, method='bilinear', preserve_aspect_ratio=False,
            antialias=False)
        return image, label
