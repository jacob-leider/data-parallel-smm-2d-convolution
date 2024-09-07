from bench import layer, swap_conv2d, swap_backend
import model_zoo

layer.run()
model_zoo.run_on(swap_conv2d.runner)
model_zoo.run_on(swap_backend.runner)
