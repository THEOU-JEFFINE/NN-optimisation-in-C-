#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <memory>
#include <vector>

class Tensor {
    public:
        Tensor() : Tensor(0, 0, 0, 0) {}
        Tensor(size_t n) : Tensor(n, 1, 1, 1) {}
        Tensor(size_t n, size_t c) : Tensor(n, c, 1, 1) {}
        Tensor(size_t n, size_t c, size_t h) : Tensor(n, c, h, 1) {}
        Tensor(size_t n, size_t c, size_t h, size_t w) :
            N(n), C(c), H(h), W(w), offset_(0), data_(std::make_shared<std::vector<float>>(n * c * h * w)) {}
        Tensor(size_t n, size_t c, size_t h, size_t w, size_t offset, std::shared_ptr<std::vector<float>> data) :
            N(n), C(c), H(h), W(w), offset_(offset), data_(data) {}

        bool empty() {
            return data_->empty();
        }

        float* data() {
            return data_->data();
        }

        void fill(float c) {
            std::fill(data_->begin() + offset_, data_->begin() + offset_ + N * C * H * W, c);
        }

        float& operator()(size_t n, size_t c=0, size_t h=0, size_t w=0) {
            // TODO
            // define the data layout for the Tensor
            // access the Tensor element
            //Using horner's method so that we can reduce the number of multiplications down to 3
            size_t index = offset_ + ((n * C + c) * H + h) * W + w;
            return (*data_)[index];
        }

        Tensor slice(size_t idx, size_t num) {
            size_t offset = offset_ + idx * C * H * W;
            return Tensor(num, C, H, W, offset, data_);
        }

        std::ostream &write(std::ostream &os) const {
            return os << N << "x" << C << "x" << H << "x" << W;
        }

        size_t N, C, H, W;

    private:
        size_t offset_;
        std::shared_ptr<std::vector<float>> data_;
};

std::ostream& operator<<(std::ostream &os, const Tensor& t) {
    return t.write(os);
}

#endif // TENSOR_HPP
