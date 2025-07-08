import tensorflow as tf
from tensorflow.keras.layers import (Layer, MultiHeadAttention, LayerNormalization, Dense, Dropout, 
                                     Conv1D, Add, BatchNormalization, LeakyReLU, Activation)
import numpy as np

def positional_encoding(length, depth_target):
    if depth_target % 2 != 0:
        raise ValueError(f"Depth target (d_model={depth_target}) must be even for sin/cos positional encoding pairing.")
    
    depth_per_part = depth_target // 2
    positions = np.arange(length)[:, np.newaxis]
    div_term_rates = np.arange(depth_per_part)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * div_term_rates) / np.float32(depth_target))
    angle_rads = positions * angle_rates
    sin_encoding = np.sin(angle_rads)
    cos_encoding = np.cos(angle_rads)
    pos_encoding = np.concatenate([sin_encoding, cos_encoding], axis=-1)
        
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(Layer):
    def __init__(self, fixed_max_sequence_length, d_model, is_vocab=False, name="positional_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.is_vocab = is_vocab
        self.fixed_max_sequence_length = fixed_max_sequence_length
        if self.is_vocab:
            self.embedding_layer = tf.keras.layers.Embedding(self.fixed_max_sequence_length, d_model, mask_zero=False)
        else:
            self.embedding_layer = None 
        self.pos_encoding_const = positional_encoding(length=self.fixed_max_sequence_length, depth_target=self.d_model)

    def call(self, x):
        current_seq_len = tf.shape(x)[1]
        if self.embedding_layer:
            x = self.embedding_layer(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pe_to_add = self.pos_encoding_const[tf.newaxis, :current_seq_len, :]
        x = x + pe_to_add
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "fixed_max_sequence_length": self.fixed_max_sequence_length,
            "d_model": self.d_model,
            "is_vocab": self.is_vocab,
        })
        return config

class BaseAttention(Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0, name="base_attention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout 
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.add = Add()

    def call(self, x, training=False): 
        attn_output, attn_scores = self.mha(query=x, value=x, key=x, return_attention_scores=True, training=training)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate,
        })
        return config

class GlobalSelfAttention(BaseAttention):
    def __init__(self, num_heads, key_dim, dropout=0.0, name="global_self_attention", **kwargs):
        super().__init__(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=name, **kwargs)

class FeedForward(Layer):
    def __init__(self, d_model, dff_multiplier, dropout_rate, name="feed_forward", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.dff_multiplier = dff_multiplier
        self.dropout_rate = dropout_rate
        self.seq = tf.keras.Sequential([
            Dense(dff_multiplier * d_model, activation='gelu', name=f"{name}_dense1"),
            Dense(d_model, name=f"{name}_dense2"),
            Dropout(dropout_rate, name=f"{name}_dropout")
        ], name=f"{name}_sequential_block")
        self.add = Add(name=f"{name}_add")
        self.layernorm = LayerNormalization(epsilon=1e-6, name=f"{name}_layernorm")

    def call(self, x, training=False): 
        x_ff = self.seq(x, training=training)
        x = self.add([x, x_ff])
        x = self.layernorm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dff_multiplier": self.dff_multiplier,
            "dropout_rate": self.dropout_rate,
        })
        return config

class EncoderLayer(Layer):
    def __init__(self, *, d_model, num_heads, dff_multiplier, dropout_rate, name="encoder_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff_multiplier = dff_multiplier
        self.dropout_rate = dropout_rate
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"{name}_mha"
        )
        self.ffn = FeedForward(d_model, dff_multiplier, dropout_rate, name=f"{name}_ffn")

    def call(self, x, training=False): 
        x = self.self_attention(x, training=training)
        x = self.ffn(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff_multiplier": self.dff_multiplier,
            "dropout_rate": self.dropout_rate,
        })
        return config

class TransformerEncoder(Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff_multiplier, dropout_rate, seq_len_for_pe, name="transformer_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff_multiplier = dff_multiplier
        self.dropout_rate = dropout_rate
        self.seq_len_for_pe = seq_len_for_pe
        self.input_projection = Dense(d_model, activation='relu', name=f"{name}_input_projection")
        self.pos_embedding_layer_instance = PositionalEmbedding(
            fixed_max_sequence_length=seq_len_for_pe, 
            d_model=d_model, 
            is_vocab=False, 
            name=f"{name}_pos_embedding"
        )
        self.encoder_layers_list = [
            EncoderLayer(d_model=d_model, num_heads=num_heads,
                         dff_multiplier=dff_multiplier, dropout_rate=dropout_rate, name=f"{name}_enc_layer_{i}")
            for i in range(num_layers)
        ]
        self.dropout_layer_main = Dropout(dropout_rate, name=f"{name}_main_dropout")

    def call(self, x, training=False):
        projected_x = self.input_projection(x)
        x_pe = self.pos_embedding_layer_instance(projected_x)
        x_dropped = self.dropout_layer_main(x_pe, training=training)
        current_x = x_dropped
        for i in range(self.num_layers):
            current_x = self.encoder_layers_list[i](current_x, training=training)
        return current_x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff_multiplier": self.dff_multiplier,
            "dropout_rate": self.dropout_rate,
            "seq_len_for_pe": self.seq_len_for_pe,
        })
        return config


def cnn_residual_block(input_tensor, filters, kernel_size, dropout_rate, name_prefix=""):
    """
    Creates a CNN residual block with a shortcut connection.
    Handles dimension changes in the shortcut path.
    """
    x = input_tensor
    
    # Main path
    x_main = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', name=f"{name_prefix}conv1")(x)
    x_main = BatchNormalization(name=f"{name_prefix}bn1")(x_main)
    x_main = LeakyReLU(negative_slope=0.01, name=f"{name_prefix}leakyrelu1")(x_main)
    x_main = Dropout(dropout_rate, name=f"{name_prefix}dropout1")(x_main)

    x_main = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal', name=f"{name_prefix}conv2")(x_main)
    x_main = BatchNormalization(name=f"{name_prefix}bn2")(x_main)

    
    
    
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', kernel_initializer='he_normal', name=f"{name_prefix}shortcut_conv")(x)
        shortcut = BatchNormalization(name=f"{name_prefix}shortcut_bn")(shortcut)
    else:
        shortcut = x

    
    x_add = Add(name=f"{name_prefix}add")([shortcut, x_main])
    
    # Final activation
    output_tensor = LeakyReLU(negative_slope=0.01, name=f"{name_prefix}final_leakyrelu")(x_add)
    
    return output_tensor

