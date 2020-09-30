def define_net(**kwargs):
    arch = kwargs['arch']

    if arch == 'r3d':
        from models.r3d_model import Network
        return Network(backbone=kwargs['backbone'], 
            n_class=kwargs['n_class'], pretrained=kwargs['pretrained'])
    elif arch == 'tsn_conv':
        from models.tsn_conv_model import Network
        return Network(backbone=kwargs['backbone'], pooling=kwargs['pooling'],
            n_class=kwargs['n_class'], pretrained=kwargs['pretrained'])
    elif arch == 'tsn_avg':
        from models.tsn_resnet_model import Network
        return Network(backbone=kwargs['backbone'], pooling=kwargs['pooling'],
            n_class=kwargs['n_class'], pretrained=kwargs['pretrained'])
    elif arch == 'tsm_conv':
        from models.tsm_conv_model import Network
        return Network(backbone=kwargs['backbone'], 
            n_class=kwargs['n_class'], pretrained=kwargs['pretrained'])
    elif arch == 'tsm_avg':
        from models.tsm_author_model import Network
        return Network(backbone=kwargs['backbone'],
            n_class=kwargs['n_class'], pretrained=kwargs['pretrained'])