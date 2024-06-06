import time

import ai3
import torch

def predict_show_time(runner, data, runner_name: str):
        start_time = time.time()
        out = None
        if isinstance(runner, torch.nn.Module):
            with torch.inference_mode():
                out = runner(data)
        elif isinstance(runner, ai3.model.Model):
            out = runner.predict(data)
        else:
            assert(0 and "invalid runner f{type(runner)}")
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"  Time {runner_name}: {inference_time} seconds")
        return out
