from .preprocessing import SparseFeat

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.layers import Input, Embedding, Layer, Flatten, Dropout, Activation, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from collections import OrderedDict

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer as glorot_normal
try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization

def create_input_layers(named_tuple):
    inputs_dict = OrderedDict()
    for fc in named_tuple:
        if isinstance(fc, SparseFeat):
            inputs_dict[fc.name] = Input(shape=(1,), name=fc.name)
        else:
            raise TypeError("Invalid feature column type,got", type(fc))
    return inputs_dict, list(inputs_dict.values())

class Linear(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(int(input_shape[-1]), self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        linear_input = inputs
        linear_logit = tf.matmul(linear_input, self.kernel)
        linear_logit = tf.reduce_sum(linear_logit, axis=-1, keepdims=True)
        return linear_logit
    
    def get_config(self, ):
        config = {'output_dim': self.output_dim}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def create_embedding_list(named_tuple):
    sparse_embedding = {}
    for feat in named_tuple:
        vocab_size = feat.vocabulary_size
        emb_size = feat.embedding_dim
        emb = Embedding(input_dim=vocab_size, output_dim=emb_size,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer='l2',
                        name='emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def embedding_lookup(sparse_embedding, inputs_dict, named_tuple):
    group_embedding_list = []
    for fc in named_tuple:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        lookup_idx = inputs_dict[feature_name]
        group_embedding_list.append(sparse_embedding[embedding_name](lookup_idx))
    return group_embedding_list

class FM(Layer):

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))
        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(concated_embeds_value), axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        return cross_term
    
    def get_config(self):
        config = super(FM, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DNN(Layer):

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [Activation(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, trainings=None, **kwargs):
        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=trainings)
            try:
                fc = self.activation_layers[i](fc, training=trainings)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=trainings)
            deep_input = fc

        return deep_input

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Final_Dense(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        super(Final_Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(int(input_shape[-1]), self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        super(Final_Dense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        output = tf.matmul(inputs, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super(Final_Dense, self).get_config()
        config.update({'output_dim': self.output_dim, 'activation': self.activation})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def DeepFM(feature_named_tuple, l2_reg_dnn, dnn_use_bn, seed, dnn_hidden_units, dnn_activation, dnn_dropout):
    inputs_dict, inputs_list = create_input_layers(feature_named_tuple)
    linear_input = tf.concat(inputs_list, axis= -1)
    linear_logit = Linear(output_dim=1)(linear_input)
    sparse_embedding = create_embedding_list(feature_named_tuple)
    group_embedding_list = embedding_lookup(sparse_embedding, inputs_dict, feature_named_tuple)
    emb_concat = tf.concat(group_embedding_list, axis=1)
    fm_logit = FM()(emb_concat)
    wide_output = keras.layers.Concatenate(axis=1, name='wide_output')([linear_logit, fm_logit])
    dnn_input = Flatten()(emb_concat)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=True)(dnn_output)
    outputs_concat = keras.layers.Concatenate(axis=1)([wide_output, dnn_logit])
    output = Final_Dense(output_dim=1, activation='sigmoid')(outputs_concat)
    model = Model(inputs=inputs_list, outputs=output)

    return model