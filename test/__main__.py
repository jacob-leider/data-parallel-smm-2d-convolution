from test import unit, swap_backend, swap_conv2d, ops
import models

unit.run()
models.run_on(swap_conv2d.runner)
models.run_on(swap_backend.runner)
ops.run()
