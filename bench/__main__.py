from bench import layer, swap_conv2d, swap_backend
import models

layer.run()
models.run_on(swap_conv2d.runner)
models.run_on(swap_backend.runner)
