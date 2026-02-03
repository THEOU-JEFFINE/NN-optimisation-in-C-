#ifndef MNIST_HPP
#define MNIST_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <string>

#include "tensor.hpp"

#ifdef MNIST_PRE_PAD
#define PRE_PAD 2
#else
#define PRE_PAD 0
#endif

class MNIST {
    public:
        // train-images-idx3-ubyte: training set images
        // train-labels-idx1-ubyte: training set labels
        // t10k-images-idx3-ubyte:  test set images
        // t10k-labels-idx1-ubyte:  test set labels
        MNIST(std::string path) : pad_(PRE_PAD), path_(path) {
            load();
        }

        void load() {
            uint32_t magic_number, num_imgs, num_rows, num_cols;

            std::ifstream is(path_.c_str(), std::ios::in | std::ios::binary);
            is.read(reinterpret_cast<char *>(&magic_number), 4);
            is.read(reinterpret_cast<char *>(&num_imgs), 4);
            is.read(reinterpret_cast<char *>(&num_rows), 4);
            is.read(reinterpret_cast<char *>(&num_cols), 4);

            if (is_little_endian()) {
                std::reverse(reinterpret_cast<char *>(&magic_number), reinterpret_cast<char *>(&magic_number) + sizeof(uint32_t));
                std::reverse(reinterpret_cast<char *>(&num_imgs), reinterpret_cast<char *>(&num_imgs) + sizeof(uint32_t));
                std::reverse(reinterpret_cast<char *>(&num_rows), reinterpret_cast<char *>(&num_rows) + sizeof(uint32_t));
                std::reverse(reinterpret_cast<char *>(&num_cols), reinterpret_cast<char *>(&num_cols) + sizeof(uint32_t));
            }

            assert(magic_number == 0x00000803 && "expected MNIST image file format");
            assert(num_rows == 28 && num_cols == 28 && "expected images of size 28x28");

            imgs_ = Tensor(num_imgs, 1, 28 + 2 * pad_, 28 + 2 * pad_);
            if (pad_)
                imgs_.fill(0);
            for (size_t n = 0; n < num_imgs; ++n) {
                for (size_t h = 0; h < 28; ++h) {
                    for (size_t w = 0; w < 28; ++w) {
                        uint8_t byte;
                        is.read(reinterpret_cast<char *>(&byte), 1);
                        // scale to 0.0 .. 1.0
                        float pixel = byte / 255.f;
                        //// normalize
                        //float mean = 0.1307f, stdev = 0.3081f;
                        //pixel = (pixel - mean) / stdev;
                        imgs_(n, 0, h + pad_, w + pad_) = pixel;
                    }
                }
            }
        }

        Tensor at(size_t idx) {
            return slice(idx, 1);
        }

        Tensor slice(size_t idx, size_t num) {
            assert(idx + num < imgs_.N && "index out of bounds");
            return imgs_.slice(idx, num);
        }

        void print(size_t idx) {
            auto img = at(idx);
            for (size_t h = pad_; h < 28 + pad_; ++h) {
                for (size_t w = pad_; w < 28 + pad_; ++w) {
                    int val = (int)(img(0, 0, h, w) * 255.f);
                    // use ANSI escape codes for colored terminal output
                    // https://en.wikipedia.org/wiki/ANSI_escape_code#Select_Graphic_Rendition_parameters
                    std::cout << "\033[48;5;" << 255 - val / 12 << "m \033[0m";
                    // text only
                    //std::cout << (val > 0 ? "x" : " ");
                }
                std::cout << std::endl;
            }
        }

    private:
        bool is_little_endian() {
            int n = 1;
            return *(char *)&n == 1;
        }
        Tensor imgs_;
        size_t pad_;
        std::string path_;
};

#endif // MNIST_HPP
