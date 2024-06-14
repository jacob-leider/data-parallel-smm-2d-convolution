def run():
    print('MODELS')
    from . import vgg16, vgg16_swap_conv2d
    vgg16.run()
    vgg16_swap_conv2d.run()
