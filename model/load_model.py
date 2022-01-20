from monai.networks.nets import UNet

def load_model(name=''):
    if name =='':
        model = UNet(
            dimensions=2,
            in_channels=3,
            out_channels=3,
            channels=(8, 16, 32,64, 128),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = UNet(
            dimensions=2,
            in_channels=3,
            out_channels=2,
            channels=(8, 16, 32, 64, 128),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    return model


