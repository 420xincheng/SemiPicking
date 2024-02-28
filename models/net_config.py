import ml_collections

def get_b16_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 512
    config.patches = ml_collections.ConfigDict({'size': 16})
    config.transformer = ml_collections.ConfigDict()
    config.transformer.dropout_rate = 0.2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.mlp_dim = 2048
    config.decoder_channels = (256, 128, 64, 16)  # 上采样四次，最前面的那个通道数
    config.n_skip = 0
    config.n_classes = 1
    config.classifier = 'seg'
    config.figure_size = [20, 6]
    return config
# patch = 16
def get_b16_tr12_config():
    """baseline"""
    config = get_b16_config()
    config.transformer.num_layers = 8  # transformer的层数
    config.transformer.num_heads = 8
    return config


def get_r50_b16_config():
    """returns the resnest50 + b16 config """
    config = get_b16_config()
    config.patches.grid = 128
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 3, 12)  # 可变参数！
    config.resnet.width_factor = 1
    config.classifier = 'seg'

    # config.pretrained_path = '../models/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [1024, 512, 256, 64]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config



def get_test_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 512
    config.patches = ml_collections.ConfigDict({'size': 16})
    config.transformer = ml_collections.ConfigDict()
    config.transformer.dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.mlp_dim = 2048
    config.decoder_channels = (256, 128, 64, 16)  # 上采样四次，最前面的那个通道数
    config.n_skip = 0
    config.n_classes = 1
    config.classifier = 'token'
    config.figure_size = [20, 6]
    return config

def get_r50_test_config():
    config = get_test_config()
    config.patches.grid = 128
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 3, 12)  # 可变参数！
    config.resnet.width_factor = 1
    config.classifier = 'seg'

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [1024, 512, 256, 64]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'token'
    return config
