from . import kn2rowconv2d, maxpool2d, linear, relu

def run():
    print('UNIT')
    kn2rowconv2d.run()
    maxpool2d.run()
    linear.run()
    relu.run()

if __name__ == "__main__":
    run()
