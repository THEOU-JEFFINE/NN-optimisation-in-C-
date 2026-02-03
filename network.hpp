#include <omp.h>
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <mutex>
#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

static int GEMM_TM = 64;
static int GEMM_TN = 64;
static int GEMM_TK = 64;

inline void get_gemm_tiling(int &tm, int &tn, int &tk) {
    tm = GEMM_TM; tn = GEMM_TN; tk = GEMM_TK;
}

inline int get_omp_threads() {
    int t = 1;
#pragma omp parallel
    {
#pragma omp master
        t = omp_get_num_threads();
    }
    return t;
}
static inline void my_sgemm(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
#ifdef HAVE_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
#else
    static int TM = GEMM_TM;
    static int TN = GEMM_TN;
    static int TK = GEMM_TK;

    int TM_loc = TM, TN_loc = TN, TK_loc = TK;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < M; i0 += TM_loc) {
        for (int j0 = 0; j0 < N; j0 += TN_loc) {
            int i_max = std::min(M, i0 + TM_loc);
            int j_max = std::min(N, j0 + TN_loc);

            for (int i = i0; i < i_max; ++i) {
                float* crow = C + i * ldc + j0;
                for (int j = 0; j < (j_max - j0); ++j) crow[j] = 0.0f;
            }

            for (int k0 = 0; k0 < K; k0 += TK_loc) {
                int k_max = std::min(K, k0 + TK_loc);

                for (int i = i0; i < i_max; ++i) {
                    const float* arow = A + i * lda + k0;
                    float* crow = C + i * ldc + j0;

                    for (int k = k0; k < k_max; ++k) {
                        float a = arow[k - k0];
                        const float* brow = B + k * ldb + j0;
                        #pragma omp simd
                        for (int j = 0; j < (j_max - j0); ++j) {
                            crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
#endif
}


enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<< (std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d:     return os << "Conv2d";
        case LayerType::Linear:     return os << "Linear";
        case LayerType::MaxPool2d:  return os << "MaxPool2d";
        case LayerType::ReLu:       return os << "ReLu";
        case LayerType::SoftMax:    return os << "SoftMax";
        case LayerType::Flatten:    return os << "Flatten";
    };
    return os << static_cast<std::uint8_t>(layer_type);
}

class Layer {
    public:
        Layer(LayerType layer_type) : layer_type_(layer_type), input_(), weights_(), bias_(), output_() {}

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) = 0;

        void print() {
            std::cout << layer_type_ << std::endl;
            if (!input_.empty())   std::cout << "  input: "   << input_   << std::endl;
            if (!weights_.empty()) std::cout << "  weights: " << weights_ << std::endl;
            if (!bias_.empty())    std::cout << "  bias: "    << bias_    << std::endl;
            if (!output_.empty())  std::cout << "  output: "  << output_  << std::endl;
        }
        void set_input(const Tensor& input) {
            input_ = input;
        }

        Tensor output() const {
            return output_;
        }

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;
};


class Conv2d : public Layer {
    public:
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::Conv2d), in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), pad_(pad) {
            
            weights_ = Tensor(out_channels, in_channels, kernel_size, kernel_size);
            
            bias_ = Tensor(out_channels);
        }

    void read_weights_bias(std::ifstream& is) override {
        is.read(reinterpret_cast<char*>(weights_.data()), weights_.N * weights_.C * weights_.H * weights_.W * sizeof(float));
        is.read(reinterpret_cast<char*>(bias_.data()), bias_.N * bias_.C * bias_.H * bias_.W * sizeof(float));
    }

    float* im2col_safe(Tensor& input, size_t n_batch, size_t out_h, size_t out_w) {
        size_t channels = input.C;
        size_t k_sz = kernel_size_;
        size_t col_h = channels * k_sz * k_sz;
        size_t col_w = out_h * out_w;
        
        thread_local std::vector<float> local_col;
        if (local_col.size() != col_h * col_w) local_col.resize(col_h * col_w);
        float* cols_ptr = local_col.data();
        long stride = static_cast<long>(this->stride_);
        long pad = static_cast<long>(this->pad_);
        long in_h = static_cast<long>(input.H);
        long in_w = static_cast<long>(input.W);

        
        for (long y = 0; y < (long)out_h; ++y) {
            for (long x = 0; x < (long)out_w; ++x) {
                long col_idx = y * out_w + x;
                long in_y_origin = y * stride - pad;
                long in_x_origin = x * stride - pad;

                size_t row_idx = 0;
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t ky = 0; ky < k_sz; ++ky) {
                        for (size_t kx = 0; kx < k_sz; ++kx) {
                            long in_y = in_y_origin + ky;
                            long in_x = in_x_origin + kx;

                            float val = 0.0f;
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                val = input(n_batch, c, in_y, in_x);
                            }

                            cols_ptr[row_idx * col_w + col_idx] = val;
                            row_idx++;
                        }
                    }
                }
            }
        }

        return cols_ptr;
    }

    void fwd() override {
        size_t batch_size = input_.N;
        size_t out_h = (input_.H + 2 * pad_ - kernel_size_) / stride_ + 1;
        size_t out_w = (input_.W + 2 * pad_ - kernel_size_) / stride_ + 1;
        size_t num_pixels = out_h * out_w;
        size_t K = in_channels_ * kernel_size_ * kernel_size_;
        
        output_ = Tensor(batch_size, out_channels_, out_h, out_w);
        
        float* w_base = &weights_(0,0,0,0);
        float* b_base = &bias_(0,0,0,0);
        float* out_base = &output_(0,0,0,0);

        for (size_t b = 0; b < batch_size; ++b) {
            
            float* col_ptr = im2col_safe(input_, b, out_h, out_w);

            
            float* C_ptr = out_base + b * (out_channels_ * num_pixels);

            my_sgemm((int)out_channels_, (int)num_pixels, (int)K,
                     w_base, (int)K,
                     col_ptr, (int)num_pixels,
                     C_ptr, (int)num_pixels);

            for (size_t o = 0; o < out_channels_; ++o) {
                float bias_val = b_base[o];
                float* out_row = C_ptr + o * num_pixels;
                #pragma omp simd
                for (size_t p = 0; p < num_pixels; ++p) out_row[p] += bias_val;
            }
        }
    }

private:
    size_t in_channels_, out_channels_, kernel_size_, stride_, pad_;
    
};

class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear), in_features_(in_features), out_features_(out_features) {
            weights_ = Tensor(out_features, in_features);
            bias_ = Tensor(out_features);
        }

    void read_weights_bias(std::ifstream& is) override {
        is.read(reinterpret_cast<char*>(weights_.data()), weights_.N * weights_.C * weights_.H * weights_.W * sizeof(float));
        is.read(reinterpret_cast<char*>(bias_.data()), bias_.N * bias_.C * bias_.H * bias_.W * sizeof(float));
    }

    void fwd() override {
        size_t batch_size = input_.N;
        output_ = Tensor(batch_size, out_features_, 1, 1);

        for(size_t n=0; n<batch_size; ++n){
            float* in_ptr = &input_(n, 0, 0, 0);
            #pragma omp parallel for schedule(static)
            for(long o=0; o<(long)out_features_; ++o){
                float sum = bias_(o, 0, 0, 0);
                float* w_ptr = &weights_(o, 0, 0, 0);

                #pragma omp simd reduction(+:sum)
                for(size_t i=0; i<in_features_; ++i){
                    sum += in_ptr[i] * w_ptr[i];
                }

                output_(n, o, 0, 0) = sum;
            }
        }
    }

    private:
        size_t in_features_, out_features_;

};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d), kernel_size_(kernel_size), stride_(stride), pad_(pad) {}
        
        void read_weights_bias(std::ifstream& is) override {}
        
    void fwd() override {
        size_t batch_size = input_.N;
        size_t channels = input_.C;
        size_t in_h = input_.H;
        size_t in_w = input_.W;

        size_t out_h = (in_h +2 * pad_ -kernel_size_)/stride_ +1;
        size_t out_w = (in_w +2 *pad_ - kernel_size_)/stride_ +1;

        output_ = Tensor(batch_size, channels, out_h, out_w);

        for(size_t n=0; n<batch_size; ++n){
            for(size_t c=0; c<channels; ++c){
                for(size_t y=0; y< out_h; ++y){
                    for(size_t x=0; x<out_w; ++x){
                        float max_val = std::numeric_limits<float>::lowest();

                        for(size_t kh=0; kh<kernel_size_; ++kh){
                            for(size_t kw=0; kw<kernel_size_; ++kw){
                                long in_y = static_cast<long>(y * stride_ +kh) -static_cast<long>(pad_);
                                long in_x = static_cast<long>(x*stride_ +kw) - static_cast<long>(pad_);

                                if(in_y >=0 && in_y <static_cast<long>(in_h) && in_x >=0 && in_x <static_cast<long>(in_w)){
                                    float val = input_(n, c, in_y, in_x);
                                    if(val > max_val){
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        output_(n, c, y, x) = max_val;
                    }
                }
            }
        }
    }

    private:
        size_t kernel_size_, stride_, pad_;

};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {}
    
    void read_weights_bias(std::ifstream& is) override {

    }

    void fwd() override{
        output_ = Tensor(input_.N, input_.C, input_.H, input_.W);

        float* in_ptr = input_.data();
        float* out_ptr = output_.data();

        size_t total_ele = input_.N * input_.C * input_.H * input_.W;

        for(size_t i=0; i<total_ele; ++i){
            if(in_ptr[i]<0.0f){
                out_ptr[i]=0.0f;
            }
            else{
                out_ptr[i] = in_ptr[i];
            }
        }
    }
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    
    void read_weights_bias(std::ifstream& is) override{

    }

    void fwd() override{
        output_ = Tensor(input_.N, input_.C, input_.H, input_.W);

        for(size_t n=0; n<input_.N; ++n){
            float max_val = input_(n, 0, 0, 0);
            for(size_t c=1; c<input_.C; ++c){
                if(input_(n, c, 0, 0)>max_val){
                    max_val = input_(n, c, 0, 0);
                }
            }

            float sum = 0.0f;
            for(size_t c=0; c<input_.C; ++c){
                float val = std::exp(input_(n, c, 0, 0)-max_val);
                output_(n, c, 0, 0)=val;
                sum+=val;
            }

            for(size_t c=0; c<input_.C; ++c){
                output_(n, c, 0, 0)/=sum;
            }

        }
    }
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    
    void read_weights_bias(std::ifstream& is) override{

    }

    void fwd() override{
        size_t flattened_sz = input_.C * input_.H * input_.W;
        output_ =Tensor(input_.N, flattened_sz, 1, 1);
        std::copy(input_.data(), input_.data()+(input_.N *flattened_sz), output_.data());
    }


};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        ~NeuralNetwork(){
            for(auto* layer: layers_){
                delete layer;
            }
            layers_.clear();
        }

        void add(Layer* layer) {
            layers_.push_back(layer);
        }

        void load(std::string file) {
            std::ifstream is(file, std::ios::binary);
            if(!is.is_open()){
                std::cerr << "Error opening File:" <<file <<std::endl;
                return;
            }
            for(auto* layer: layers_){
                layer->read_weights_bias(is);
            }
            is.close();
        }

        Tensor predict(Tensor input) {
            Tensor next_input = input;
            for(auto* layer: layers_){
                layer->set_input(next_input);
                layer->fwd();
                next_input = layer->output();

                if(debug_){
                    layer->print();
                }
            }
            return next_input;
        }

    private:
        bool debug_;
        std::vector<Layer*> layers_;
};

#endif // NETWORK_HPP
