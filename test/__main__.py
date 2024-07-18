from test import unit, swap_backend, swap_conv2d
import runners

unit.run()
runners.run_on(swap_conv2d.runner)
runners.run_on(swap_backend.runner)
