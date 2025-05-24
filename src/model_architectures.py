import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout, Add, Activation,
    Flatten, Concatenate, LayerNormalization
)

from src.custom_layers import TransformerEncoder, cnn_residual_block
from src import config

def build_cnn_transformer_hybrid(input_shape, num_classes, hp=None):
    print(f"DEBUG build_cnn_transformer_hybrid: input_shape={input_shape}, num_classes={num_classes}, hp_is_None={hp is None}")

    def get_hp(name, default_value_key_in_config, default_value_literal=None):
        if hp:
            if name == 'conv_filters_base': return hp.suggest_categorical('conv_filters_base', [32, 64, 128])
            if name == 'conv_kernel_size': return hp.suggest_categorical('conv_kernel_size', [3, 5, 7])
            if name == 'conv_layers': return hp.suggest_int('conv_layers', 1, 3)
            if name == 'conv_dropout': return hp.suggest_float('conv_dropout', 0.1, 0.4, step=0.05)
            if name == 'transformer_layers': return hp.suggest_int('transformer_layers', 1, 4)
            d_model_val = hp.suggest_categorical('d_model', [64, 128])
            possible_heads = [h for h in [2, 4, 8] if d_model_val % h == 0]
            if not possible_heads: possible_heads = [min([2,4,8], key=lambda x: abs(x - d_model_val/4) if d_model_val/4 >= x else float('inf'))]
            if not possible_heads: possible_heads = [2]
            num_heads_val = hp.suggest_categorical(f'num_heads_for_d_model_{d_model_val}', possible_heads)
            if name == 'd_model': return d_model_val
            if name == 'num_heads': return num_heads_val
            if name == 'dff_multiplier': return hp.suggest_int('dff_multiplier', 2, 4)
            if name == 'transformer_dropout': return hp.suggest_float('transformer_dropout', 0.1, 0.4, step=0.05)
            if name == 'dense_units_1': return hp.suggest_categorical('dense_units_1', [128, 256, 512])
            if name == 'dense_units_2': return hp.suggest_categorical('dense_units_2', [64, 128, 256])
            if name == 'head_dropout': return hp.suggest_float('head_dropout', 0.3, 0.7, step=0.05)
            print(f"Warning: HP '{name}' not explicitly defined for Optuna suggestion in get_hp.")
            return getattr(config, default_value_key_in_config.upper(), default_value_literal)
        return getattr(config, default_value_key_in_config.upper(), default_value_literal)

    inputs = Input(shape=input_shape, name="input_layer")

    cfg_conv_filters_base = get_hp('conv_filters_base', 'CONV_FILTERS_BASE', config.CONV_FILTERS_BASE)
    cfg_conv_kernel_size = get_hp('conv_kernel_size', 'CONV_KERNEL_SIZE', config.CONV_KERNEL_SIZE)
    cfg_conv_dropout = get_hp('conv_dropout', 'CONV_DROPOUT', config.CONV_DROPOUT)
    cfg_conv_layers = get_hp('conv_layers', 'CONV_LAYERS', config.CONV_LAYERS)

    current_filters = cfg_conv_filters_base
    x = Conv1D(current_filters, kernel_size=cfg_conv_kernel_size, padding='same', kernel_initializer='he_normal', name="initial_conv")(inputs)
    x = BatchNormalization(name="initial_bn")(x)
    x = LeakyReLU(negative_slope=0.01, name="initial_leakyrelu")(x)
    x = Dropout(cfg_conv_dropout, name="initial_dropout")(x)

    for i in range(cfg_conv_layers):
        current_filters = max(32, current_filters * 2)
        x = cnn_residual_block(x, current_filters, kernel_size=cfg_conv_kernel_size, dropout_rate=cfg_conv_dropout, name_prefix=f"resblock_{i}_")
        if i < cfg_conv_layers - 1:
            x = MaxPooling1D(pool_size=2, strides=1, padding='same', name=f"maxpool_res_{i}")(x)

    cfg_transformer_layers = get_hp('transformer_layers', 'TRANSFORMER_LAYERS', config.TRANSFORMER_LAYERS)
    cfg_d_model = get_hp('d_model', 'D_MODEL', config.D_MODEL)
    cfg_num_heads = get_hp('num_heads', 'NUM_HEADS', config.NUM_HEADS)
    cfg_dff_multiplier = get_hp('dff_multiplier', 'DFF_MULTIPLIER', config.DFF_MULTIPLIER)
    cfg_transformer_dropout = get_hp('transformer_dropout', 'TRANSFORMER_DROPOUT', config.TRANSFORMER_DROPOUT)

    seq_len_for_pe = x.shape[1]

    transformer_output = TransformerEncoder(
        num_layers=cfg_transformer_layers,
        d_model=cfg_d_model,
        num_heads=cfg_num_heads,
        dff_multiplier=cfg_dff_multiplier,
        dropout_rate=cfg_transformer_dropout,
        seq_len_for_pe=seq_len_for_pe,
        name="transformer_encoder"
    )(x)

    avg_pool = GlobalAveragePooling1D(name="global_avg_pool")(transformer_output)
    max_pool = GlobalMaxPooling1D(name="global_max_pool")(transformer_output)
    pooled_output = Concatenate(name="concatenate_pools")([avg_pool, max_pool])
    pooled_output = LayerNormalization(name="pool_layernorm")(pooled_output)

    cfg_dense_units_1 = get_hp('dense_units_1', 'DENSE_UNITS_1', config.DENSE_UNITS_1)
    cfg_head_dropout = get_hp('head_dropout', 'HEAD_DROPOUT', config.HEAD_DROPOUT)
    cfg_dense_units_2 = get_hp('dense_units_2', 'DENSE_UNITS_2', config.DENSE_UNITS_2)

    head = Dense(cfg_dense_units_1, kernel_initializer='he_normal', name="head_dense_1")(pooled_output)
    head = BatchNormalization(name="head_bn_1")(head)
    head = LeakyReLU(negative_slope=0.01, name="head_leakyrelu_1")(head)
    head = Dropout(cfg_head_dropout, name="head_dropout_1")(head)

    head = Dense(cfg_dense_units_2, kernel_initializer='he_normal', name="head_dense_2")(head)
    head = BatchNormalization(name="head_bn_2")(head)
    head = LeakyReLU(negative_slope=0.01, name="head_leakyrelu_2")(head)
    head = Dropout(cfg_head_dropout, name="head_dropout_2")(head)

    outputs = Dense(num_classes, activation='softmax', name="output_softmax")(head)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_Transformer_Hybrid")
    print("DEBUG build_cnn_transformer_hybrid: Model constructed.")
    return model


def build_simple_1d_cnn(input_shape, num_classes, hp=None):
    print(f"DEBUG build_simple_1d_cnn: input_shape={input_shape}, num_classes={num_classes}, hp_is_None={hp is None}")

    filters1_default = 64
    kernel_size1_default = 5
    dropout_rate_conv_default = 0.2
    num_conv_blocks_default = 2
    dense_units_default = 128
    dropout_rate_dense_default = 0.5

    if hp:
        filters1 = hp.suggest_categorical('simple_cnn_filters1', [32, 64, 128])
        kernel_size1 = hp.suggest_categorical('simple_cnn_kernel_size1', [3, 5, 7])
        dropout_rate_conv = hp.suggest_float('simple_cnn_dropout_conv', 0.1, 0.4, step=0.05)
        num_conv_blocks = hp.suggest_int('simple_cnn_conv_blocks', 1, 2)
        dense_units = hp.suggest_categorical('simple_cnn_dense_units', [64, 128, 256])
        dropout_rate_dense = hp.suggest_float('simple_cnn_dropout_dense', 0.3, 0.6, step=0.05)
