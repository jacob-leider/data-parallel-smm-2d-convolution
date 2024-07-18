from bench import layers, swap_conv2d, swap_backend
import runners

layers.run()
runners.run_on(swap_conv2d.runner)
runners.run_on(swap_backend.runner)
