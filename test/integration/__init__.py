
def run():
    from . import vgg16, created_conv2d, vgg16_swap_conv2d, created_conv2d_swap_conv2d
    print("INTEGRATION")
    vgg16.run()
    created_conv2d.run()
    vgg16_swap_conv2d.run()
    created_conv2d_swap_conv2d.run()
