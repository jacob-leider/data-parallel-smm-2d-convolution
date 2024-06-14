def run():
    print('LAYERS')
    from . import conv2d, linear, avgpool2d, maxpool2d, adaptiveavgpool2d, relu, flatten
    conv2d.run()
    avgpool2d.run()
    linear.run()
    maxpool2d.run()
    adaptiveavgpool2d.run()
    relu.run()
    flatten.run()
