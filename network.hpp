#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>

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
        // TODO: additional required methods
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
            //OC-IC-KH-KW layout
            weights_ = Tensor(out_channels, in_channels, kernel_size, kernel_size);
            //One value for the output channel
            bias_ = Tensor(out_channels);
        }

    // TODO
    void read_weights_bias(std::ifstream& is) override {
        is.read(reinterpret_cast<char*>(weights_.data()), weights_.N * weights_.C * weights_.H * weights_.W * sizeof(float));
        is.read(reinterpret_cast<char*>(bias_.data()), bias_.N * bias_.C * bias_.H * bias_.W * sizeof(float));
    }

    void fwd() override {
        size_t batch_size = input_.N;
        size_t in_h = input_.H;
        size_t in_w = input_.W;

        size_t out_h = (in_h +2 * pad_ - kernel_size_)/stride_ +1;
        size_t out_w = (in_w +2 * pad_ -kernel_size_)/stride_ +1;

        output_ = Tensor(batch_size, out_channels_, out_h, out_w);

        for(size_t n=0; n<batch_size; ++n){
            for(size_t oc=0; oc<out_channels_; ++oc){
                for(size_t y=0; y<out_h; ++y){
                    for(size_t x=0; x<out_w; ++x){
                        float sum = bias_(oc);
                        
                        for(size_t ic=0; ic<in_channels_; ++ic){
                            for(size_t kh=0; kh<kernel_size_; ++kh){
                                for(size_t kw=0; kw<kernel_size_; ++kw){
                                    long in_y = static_cast<long>(y * stride_ +kh)- static_cast<long>(pad_);
                                    long in_x = static_cast<long>(x * stride_ +kw)- static_cast<long>(pad_);

                                    if (in_y >= 0 && in_y <static_cast<long>(in_h) && in_x >=0 && in_x <static_cast<long>(in_w)){
                                        sum+=input_(n, ic, in_y, in_x) * weights_(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }

                        output_(n, oc, y, x) = sum;
                    }
                }
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

    // TODO
    void read_weights_bias(std::ifstream& is) override {
        is.read(reinterpret_cast<char*>(weights_.data()), weights_.N * weights_.C * weights_.H * weights_.W * sizeof(float));
        is.read(reinterpret_cast<char*>(bias_.data()), bias_.N * bias_.C * bias_.H * bias_.W * sizeof(float));
    }

    void fwd() override {
        size_t batch_size = input_.N;
        output_ = Tensor(batch_size, out_features_, 1, 1);

        for(size_t n=0; n<batch_size; ++n){
            for(size_t o=0; o<out_features_; ++o){
                float sum = bias_(o, 0, 0, 0);
                for(size_t i=0; i<in_features_; ++i){
                    sum+= input_(n, i, 0, 0) * weights_(o, i, 0, 0);
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
    // TODO
        
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
    // TODO
    
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
    // TODO

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
    // TODO
    
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

        //Destructor for cleaning up
        ~NeuralNetwork(){
            for(auto* layer: layers_){
                delete layer;
            }
            layers_.clear();
        }

        void add(Layer* layer) {
            // TODO
            layers_.push_back(layer);
        }

        void load(std::string file) {
            // TODO
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
            // TODO
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
        // TODO: storage for layers
        std::vector<Layer*> layers_;
};

#endif // NETWORK_HPP
