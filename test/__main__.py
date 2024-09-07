from test import unit, swap_backend, swap_conv2d, ops
import model_zoo

unit.run()
model_zoo.run_on(swap_conv2d.runner)
model_zoo.run_on(swap_backend.runner)
ops.run()
