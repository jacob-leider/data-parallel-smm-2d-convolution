from . import kn2rowconv2d, maxpool2d, linear, relu, avgpool2d, adaptiveavgpool2d, flatten

def run():
    print('UNIT')
    kn2rowconv2d.run()
    maxpool2d.run()
    linear.run()
    relu.run()
    avgpool2d.run()
    adaptiveavgpool2d.run()
    flatten.run()

if __name__ == "__main__":
    run()
