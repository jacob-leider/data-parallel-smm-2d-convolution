#include "utils.hpp"
#include <torch/extension.h>

// TODO implement these
// https://lemire.me/blog/2020/06/10/reusing-a-thread-in-c-for-better-performance/
// Split out_h and out_w into ranges that cover the whole thing and give that to
// a thread
//

// TODO try to pass the f created earlier into this function then we can this
// thread worker more generalizable
template <typename dtype> struct worker {
    worker() = default;
    void start(const dtype *input, const int input_channels,
               const int input_height, const int input_width,
               const dtype *kernel, const int kernel_height,
               const int kernel_width, const int kernel_size,
               const std::optional<const dtype *> bias, dtype *output,
               const int output_height, const int output_width,
               const std::vector<int> &padding, const std::vector<int> &stride,
               const std::vector<int> &dilation) {
        thread = std::thread([&]() {
            while (true) {
                while (!has_work.load()) {
                    if (exiting.load()) {
                        return;
                    }
                }
                dtype sum = 0;
                const int out_ch = out_channel.load(); // TODO make these ulongs
                const int out_r = out_row.load();
                const int out_c = out_col.load();
                if ((out_ch < 0) && (out_r < 0) && (out_c < 0)) {
                    std::cout << "less than zero\n";
                }
                if ((out_ch == 0) && (out_r == 0) && (out_c == 0)) {
                    std::cout << "all zeros in start\n";
                }
                for (int in_c = 0; in_c < input_channels; ++in_c) {
                    for (int kern_r = 0; kern_r < kernel_height; ++kern_r) {
                        for (int kern_c = 0; kern_c < kernel_width; ++kern_c) {
                            int h_offset = out_r * stride[0] - padding[0] +
                                           kern_r * dilation[0];
                            int w_offset = out_c * stride[1] - padding[1] +
                                           kern_c * dilation[1];
                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                sum +=
                                    input[in_c * input_height * input_width +
                                          h_offset * input_width + w_offset] *
                                    kernel[out_ch * input_channels *
                                               kernel_size +
                                           in_c * kernel_size +
                                           kern_r * kernel_width + kern_c];
                            }
                        }
                    }
                }
                if (bias.has_value()) {
                    sum += bias.value()[out_ch];
                }
                output[out_ch * output_height * output_width +
                       out_r * output_width + out_c] = sum;

                has_work.store(false);
            }
        });
    }

    inline void stop_thread() {
        exiting.store(true);
        if (thread.joinable()) {
            thread.join();
        }
    }

    inline void work(int out_ch, int out_r, int out_c) {
        if ((out_ch == 0) && (out_r == 0) && (out_c == 0)) {
            std::cout << "all zeros in work\n";
        }
        // TODO we have a case all zeros in work but not all zeros in start
        // no idea how
        out_channel.store(out_ch);
        out_row.store(out_r);
        out_col.store(out_c);
        has_work.store(true);
    }

    inline bool free() { return !has_work.load(); }

  private:
    std::atomic<bool> has_work{false};
    std::atomic<bool> exiting{false};
    std::atomic<int> out_channel;
    std::atomic<int> out_row;
    std::atomic<int> out_col;
    std::thread thread;
};

template <typename dtype>
at::Tensor run(const dtype *input, const std::vector<int> input_shape,
               const dtype *kernel, const std::vector<int> kernel_shape,
               const std::optional<const dtype *> bias,
               const std::vector<int> output_shape,
               const std::vector<int> padding, const std::vector<int> stride,
               const std::vector<int> dilation) {
    const int input_channels = input_shape[1];
    const int input_height = input_shape[2];
    const int input_width = input_shape[3];
    const int kernel_height = kernel_shape[2];
    const int kernel_width = kernel_shape[3];

    const int output_channels = kernel_shape[0];
    const int output_height = output_shape[2];
    const int output_width = output_shape[3];
    const int kernel_size = kernel_height * kernel_width;

    dtype *output = new dtype[output_channels * output_height * output_width];

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<worker<dtype>> workers(num_threads);
    for (worker<dtype> &w : workers) {
        w.start(input, input_channels, input_height, input_width, kernel,
                kernel_height, kernel_width, kernel_size, bias, output,
                output_height, output_width, padding, stride, dilation);
    }

    for (int out_ch = 0; out_ch < output_channels; ++out_ch) {
        for (int out_r = 0; out_r < output_height; ++out_r) {
            for (int out_c = 0; out_c < output_width; ++out_c) {
                bool worker_found = false;
                while (!worker_found) {
                    for (worker<dtype> &w : workers) {
                        if (w.free()) {
                            w.work(out_ch, out_r, out_c);
                            worker_found = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (worker<dtype> &w : workers) {
        w.stop_thread();
    }

    return torch::from_blob(output,
                            {1, output_channels, output_height, output_width});
}

IMPL_ENTRY_FOR_DOUBLE_FLOAT
