import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.layers.experimental import preprocessing



import numpy as np
import random
# import matplotlib.pyplot as plt
# import tensorboard
import argparse

import dataset



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'y', 'true', 't', '1']:
        return True
    elif v.lower() in ['no', 'n', 'false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Input Arguments
parser = argparse.ArgumentParser(description='ClassifierHybrid')
parser.add_argument('--train_batch_size', type=int, default=512, metavar='N',
                    help='Input batch size for training (default: 10)')
parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                    help='Input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Number of epochs to train (default: 1000)')
parser.add_argument('--num_iterations', type=int, default=100000,
                    help='Number of iterations per epoch (default: 5000)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='Learning rate (default: 1e-3)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='Gamma',
                    help='Learning rate drop rate (default: 0.1)')
parser.add_argument('--lr_step', type=int, default=100, metavar='LRStep',
                    help='Change learning rate step (default: 20000)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Learning momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0, metavar='WD',
                    help='Weight decay (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dataset', type=str, default='cars196',
                    help='Folder containing data')
parser.add_argument('--img_size', type=int, default=260,
                    help='Image size as input (default: 260)')
parser.add_argument('--d_model', type=int, default=16,
                    help='Dimension of embeddings (default: 768)')
parser.add_argument('--d_mlp', type=int, default=256,#3072,
                    help='Dimension of MLP inside transformer (default: 3072)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of heads (default: 12)')
parser.add_argument('--num_layers', type=int, default=8,
                    help='Number of layers (default: 12)')
parser.add_argument('--num_classes', type=int, default=196,
                    help='Number of classes (default: 196)')
parser.add_argument('--num_patches', type=int, default=9,
                    help='Number of patches (default: 9)')
parser.add_argument('--patch_size', type=int, default=1,
                    help='Patche sizes (default: 1)')
parser.add_argument('--num_channels', type=int, default=1408,
                    help='Number of channels after backbone (default: 1408)')
parser.add_argument('--model', type=str2bool, default=True,
                    help='The name of the model to load')

########################
######## Config ########
########################

args = parser.parse_args()



class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        kernel_initializer=tf.keras.initializers.GlorotUniform()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension {0}\
                should be divisible by number of heads {1}".format(embed_dim, num_heads))
        self.proj_dim = embed_dim // num_heads
        self.query_mat = layers.Dense(embed_dim, kernel_initializer=kernel_initializer)
        self.key_mat = layers.Dense(embed_dim, kernel_initializer=kernel_initializer)
        self.value_mat = layers.Dense(embed_dim, kernel_initializer=kernel_initializer)
        self.combine_mat = layers.Dense(embed_dim, kernel_initializer=kernel_initializer)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        d_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(d_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, inputs):
        # inputs.shape = [batch_size, seq_len, embed_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_mat(inputs) # (batch_size, seq_len, embed_dim)
        key = self.key_mat(inputs) # (batch_size, seq_len, embed_dim)
        value = self.value_mat(inputs) # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(query, batch_size) # (batch_size, num_heads, seq_len, proj_dim)
        key = self.separate_heads(key, batch_size) # (batch_size, num_heads, seq_len, proj_dim)
        value = self.separate_heads(value, batch_size) # (batch_size, num_heads, seq_len, proj_dim)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0,2,1,3]) # (batch_size, seq_len, num_heads, proj_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim)) # (batch_size, seq_len, embed_dim)
        output = self.combine_mat(concat_attention) # (batch_size, seq_len, embed_dim)

        return output




class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim,
                        activation=tf.keras.activations.gelu,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6)),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        bias_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6)),
            layers.Dropout(dropout_rate)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x = self.attn(x)
        x = self.dropout1(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlp(x)
        out = x + y

        return out


class ViT(tf.keras.Model):
    def __init__(self, img_size, channels, patch_size, num_layers,
        num_classes, d_model, num_heads, d_mlp, dropout_rate=0.1):
        super(ViT, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        initializer_pos = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)

        self.pos_embedding = initializer_pos( shape=(1, num_patches + 1, d_model))
        # self.add_weight("pos_embed", shape=(1, num_patches + 1, d_model))
        self.dropout = layers.Dropout(dropout_rate)
        initializer_cls = tf.keras.initializers.Zeros()
        self.class_embedding = initializer_cls(shape=(1, 1, d_model))
        # self.add_weight("cls_embed", shape=(1, 1, d_model))
        self.patch_projection = layers.Dense(d_model)
        self.encoder_layers = [TransformerBlock(d_model, num_heads, d_mlp, dropout_rate)
                for _ in range(num_layers)]
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_head = tf.keras.Sequential([
            # layers.Dense(d_mlp, activation='relu', kernel_initializer=tf.keras.initializers.Zeros()),
            # layers.Dropout(dropout_rate),
            layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.Zeros())
        ])

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images,
                                            sizes=[1, self.patch_size, self.patch_size, 1],
                                            strides=[1, self.patch_size, self.patch_size, 1],
                                            rates=[1, 1, 1, 1],
                                            padding="VALID")
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        patches = self.extract_patches(x)
        x = self.patch_projection(patches)

        class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.d_model])
        x = tf.concat([class_embedding, x], axis=1)

        x = x + self.pos_embedding
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, training)
        x = self.layernorm(x)

        output = self.mlp_head(x[:, 0])
        return output

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=400):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



class ClassifierHybrid(tf.keras.Model):

    def __init__(self):
        super(ClassifierHybrid, self).__init__()
        self.global_step = 0
        self.backbone = self.get_backbone()
        self.backbone.trainable = False
        trainable_count = np.sum([K.count_params(w) for w in self.backbone.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.backbone.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
        # self.head = tf.keras.Sequential([
        #     layers.Flatten(),
        #     layers.Dense(256, activation='relu'),
        #     layers.Dense(196)
        # ])

        # self.vision_transformer = ViT(img_size=9, channels=1408, patch_size=1, num_layers=8,
        #                  num_classes=196, d_model=512, num_heads=8, d_mlp=512)

        self.vision_transformer = ViT(img_size=args.num_patches,
                                      channels=args.num_channels,
                                      patch_size=args.patch_size,
                                      num_layers=args.num_layers,
                                      num_classes=args.num_classes,
                                      d_model=args.d_model,
                                      num_heads=args.num_heads,
                                      d_mlp=args.d_mlp)
        self.prepare_datasets()
        self.flag = True
        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(260, 260, 3)),
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(),
                preprocessing.RandomContrast(factor=0.1),
            ],
            name="augmentation",
        )
        # self.augmentation.add(tf.keras.Input(shape=(260,260,3)))


    def get_backbone(self):
        EfficientNet_B2 = tf.keras.applications.EfficientNetB2(input_shape=(args.img_size,args.img_size,3),
                        include_top=False, weights="imagenet")
        return EfficientNet_B2

    def prepare_datasets(self):
        cars_ds = dataset.Dataset(args.dataset, args.train_batch_size,
                                    args.test_batch_size, size=(args.img_size,args.img_size))
        self.train_ds = cars_ds.train_ds
        self.test_ds = cars_ds.test_ds
        self.ds_info = cars_ds.ds_info


    def get_optimizer(self):
        # cosine_decay = tf.compat.v1.train.cosine_decay
        # decay_steps = args.lr_step
        # lr_decayed_fn = tf.keras.experimental.CosineDecay(
        #     args.lr, decay_steps)
        # lr_decayed_fn = cosine_decay(args.lr, self.global_step, args.lr_step)
        lr_decayed_fn = CustomSchedule(args.d_model)
        return tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=args.momentum)
        # return tf.keras.optimizers.Nadam(learning_rate=lr_decayed_fn)

    def setup(self):

        self.optimizer = self.get_optimizer()
        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer._decayed_lr(tf.float32)
            return lr
        lr_metric = get_lr_metric(self.optimizer)
        self.compile(optimizer=self.optimizer,
                     loss=tf.keras.losses.CategoricalCrossentropy(),   #tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy', lr_metric])#, tf.keras.metrics.SparseCategoricalAccuracy()])


    def call(self, x, training):
        self.global_step += 1
        if training and random.random()<0.5:
            x = self.augmentation(x)
        features = self.backbone(x)
        # out = self.head(features)
        out = self.vision_transformer(features)
        out = tf.nn.softmax(out, axis=-1)

        if self.flag:
            trainable_count = np.sum([K.count_params(w) for w in self.vision_transformer.trainable_weights])
            non_trainable_count = np.sum([K.count_params(w) for w in self.vision_transformer.non_trainable_weights])

            print('Total params: {:,}'.format(trainable_count + non_trainable_count))
            print('Trainable params: {:,}'.format(trainable_count))
            print('Non-trainable params: {:,}'.format(non_trainable_count))
            self.flag = False


        return out




def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


    my_callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
        #                       patience=4, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath = './models3/weights.epoch{epoch:03d}-val_accuracy{val_accuracy:.3f}',
            monitor="val_accuracy",
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
            options=None,
        ),
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    # myModel = tf.keras.models.load_model('./checkpoint')
    myModel = ClassifierHybrid()
    myModel.setup()
    myModel.flag = False
    x = tf.zeros((1,260,260,3))
    y = myModel(x)
    myModel.flag = True
    # if args.model:
    #     myModel.load_weights('')#weights.044-0.780')

    myModel.fit(x=myModel.train_ds,
                batch_size=args.train_batch_size,
                epochs=args.epochs,
                steps_per_epoch=None,
                validation_steps=None,
                verbose=1,
                validation_data=myModel.test_ds,
                shuffle=True,
                callbacks=my_callbacks
                )


if __name__ == "__main__":
    main()
