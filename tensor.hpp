#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

class Tensor {
    public:
        using storage_t = int16_t;

        Tensor() : Tensor(0, 0, 0, 0) {}
        Tensor(size_t n) : Tensor(n, 1, 1, 1) {}
        Tensor(size_t n, size_t c) : Tensor(n, c, 1, 1) {}
        Tensor(size_t n, size_t c, size_t h) : Tensor(n, c, h, 1) {}
        Tensor(size_t n, size_t c, size_t h, size_t w) :
            N(n), C(c), H(h), W(w), offset_(0), scale_(1.0f/256.0f), data_(std::make_shared<std::vector<storage_t>>(n * c * h * w)) {}
        Tensor(size_t n, size_t c, size_t h, size_t w, size_t offset, std::shared_ptr<std::vector<storage_t>> data) :
            N(n), C(c), H(h), W(w), offset_(offset), scale_(1.0f/256.0f), data_(data) {}

        bool empty() const {
            return data_->empty();
        }

        storage_t* raw_data() {
            return data_->data();
        }

        void set_scale(float s) { scale_ = s; }
        float scale() const { return scale_; }

        void fill(float c) {
            storage_t v = quantize(c);
            std::fill(data_->begin() + offset_, data_->begin() + offset_ + N * C * H * W, v);
        }

        // Proxy to allow assignment like t(n,c,h,w) = float_value
        class TensorRef {
            public:
                TensorRef(Tensor &t, size_t idx) : t_(t), idx_(idx) {}
                operator float() const { return static_cast<float>(t_.data_->at(idx_)) * t_.scale_; }
                TensorRef& operator=(float v) { t_.data_->at(idx_) = t_.quantize(v); return *this; }
            private:
                Tensor &t_;
                size_t idx_;
        };

        TensorRef operator()(size_t n, size_t c=0, size_t h=0, size_t w=0) {
            size_t index = offset_ + ((n * C + c) * H + h) * W + w;
            return TensorRef(*this, index);
        }

        float get_flat(size_t i) const {
            return static_cast<float>(data_->at(offset_ + i)) * scale_;
        }

        void set_flat(size_t i, float v) {
            data_->at(offset_ + i) = quantize(v);
        }

        Tensor slice(size_t idx, size_t num) const {
            size_t offset = offset_ + idx * C * H * W;
            return Tensor(num, C, H, W, offset, data_);
        }

        std::ostream &write(std::ostream &os) const {
            return os << N << "x" << C << "x" << H << "x" << W;
        }

        // Quantize a buffer of floats into this tensor and set an appropriate scale
        void quantize_from_floats(const std::vector<float> &src) {
            size_t count = N * C * H * W;
            if (src.size() != count) return;

            float max_abs = 0.0f;
            for (float v : src) max_abs = std::max(max_abs, std::fabs(v));
            if (max_abs == 0.0f) {
                scale_ = 1.0f;
            } else {
                scale_ = max_abs / static_cast<float>(std::numeric_limits<storage_t>::max());
            }

            for (size_t i = 0; i < count; ++i) {
                data_->at(offset_ + i) = quantize(src[i]);
            }
        }

        size_t N, C, H, W;

    private:
        size_t offset_;
        float scale_;
        std::shared_ptr<std::vector<storage_t>> data_;

        storage_t quantize(float v) const {
            if (scale_ == 0.0f) return 0;
            float inv = 1.0f / scale_;
            float q = std::round(v * inv);
            if (q > std::numeric_limits<storage_t>::max()) q = std::numeric_limits<storage_t>::max();
            if (q < std::numeric_limits<storage_t>::min()) q = std::numeric_limits<storage_t>::min();
            return static_cast<storage_t>(q);
        }
};

std::ostream& operator<<(std::ostream &os, const Tensor& t) {
    return t.write(os);
}

#endif // TENSOR_HPP
