#define MNIST_PRE_PAD

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cassert>
#include <random>
#include <iomanip>
#include <chrono>

#include "tensor.hpp"
#include "mnist.hpp"
#include "network.hpp"

std::vector<uint8_t> read_mnist_labels(std::string path) {
    std::ifstream is(path, std::ios::binary);
    if(!is.is_open()){
        std::cerr<<"Error opening the label file:" <<path<<std::endl;
        exit(1);
    }

    uint32_t magic_no, no_itms;
    is.read(reinterpret_cast<char*>(&magic_no), 4);
    is.read(reinterpret_cast<char*>(&no_itms), 4);

    auto swap_endianness = [](uint32_t val){
        return ((val << 24) & 0xFF000000) |
               ((val << 8)  & 0x00FF0000) |
               ((val >> 8)  & 0x0000FF00) |
               ((val >> 24) & 0x000000FF);

    };

    if (magic_no != 2049) {
        magic_no = swap_endianness(magic_no);
        no_itms = swap_endianness(no_itms);
    }
    assert(magic_no == 2049 && "Invalid labels file magic number");

    std::vector<uint8_t> labels(no_itms);
    is.read(reinterpret_cast<char*>(labels.data()), no_itms);   
    return labels;
}

void print_confidence(Tensor output){
    std::cout << "Class Confidence:" <<std::endl;
    float max_val = -1.0f;
    int pred_class = 0;
    for(int c=0; c<10; ++c){
        if (output(0, c, 0, 0) > max_val){
            max_val = output(0, c, 0, 0);
            pred_class = c;
        }
    }

    for(int c=0; c<10; c++){
        float prob = output(0, c, 0, 0);
        int bars = static_cast<int>(prob *20);
        std::cout <<(c == pred_class ? "--> " : "   ") << c <<": [";
        for(int b=0; b<20; ++b) std::cout <<(b < bars ? "#" : " ");
        std::cout <<"] " <<std::fixed <<std::setprecision(2) <<(prob *100) <<"%" <<std::endl;
           

        
    }
}

void print_binary(Tensor input){
    int H=input.H;
    int W=input.W;

    for(int y=0; y<H;++y){
        for(int x=0; x<W; ++x){
            float val = input(0, 0, y, x);
            if(val >0.1f){
                std::cout<<"\033[1;32m1\033[0m";
            } else {
                std::cout<<"\033[1;30m0\033[0m";
            }
        }
        std::cout <<"\n";
    }
}

int main(int argc, char** argv){
    std::string image_file ="data-mnist-t10k-images-idx3-ubyte";
    std::string label_file ="data-mnist-t10k-labels-idx1-ubyte";
    std::string weights_file ="data-mnist-lenet.raw";

    MNIST test_data(image_file);

    std::vector<uint8_t> test_labels;
    bool have_labels = false;
    std::ifstream label_check(label_file);
    if(label_check.good()){
        label_check.close();
        test_labels = read_mnist_labels(label_file);
        have_labels = true;
    } else{
        std::cout <<"Label file not found. Accuracy will not be computed." <<std::endl;
    }

    NeuralNetwork nn;

    //layer 1: conv2d 1x32x32 to 6x28x28 (kernel 5x5)
    nn.add(new Conv2d(1, 6, 5, 1));
    nn.add(new ReLu());


    //layer 2: pool 6x28x28 to  6x14x14 
    nn.add(new MaxPool2d(2, 2));

    //layer 3: conv2d 6x14x14 to 16x10x10
    nn.add(new Conv2d(6, 16, 5, 1));
    nn.add(new ReLu());


    //layer 4: pool 16x10x10 to 16x5x5
    nn.add(new MaxPool2d(2, 2));

    //16x5x5 to 400
    nn.add(new Flatten());

    //layer 5: linear 400 to 120
    nn.add(new Linear(400, 120));
    nn.add(new ReLu());

    //layer 6: linear 120 to 84
    nn.add(new Linear(120, 84));
    nn.add(new ReLu());

    //layer 7: linear 84 to 10
    nn.add(new Linear(84, 10));
    nn.add(new SoftMax());

    nn.load(weights_file);
    std::cout <<"Loaded weights from " << weights_file <<std::endl;

    // Print memory metrics for parameters (weights + biases)
    size_t total_params = nn.total_params();
    size_t bytes_float32 = nn.memory_bytes_for_params(4);
    size_t bytes_int8 = nn.memory_bytes_for_params(1);
    double saved_bytes = static_cast<double>(bytes_float32 - bytes_int8);
    double saved_pct = 100.0 * saved_bytes / static_cast<double>(bytes_float32);

    std::cout << "Model parameters: " << total_params << std::endl;
    std::cout << "Parameter memory (float32): " << bytes_float32 << " bytes" << std::endl;
    std::cout << "Parameter memory (int16):   " << bytes_int8 << " bytes" << std::endl;
    std::cout << "Saved: " << saved_bytes << " bytes (" << saved_pct << "% )" << std::endl;

    std::cout <<"Running inference ... " <<std::endl;

    size_t num_correct =0;
    size_t num_tests = 1000;

    if(have_labels && num_tests > test_labels.size()){
        num_tests = test_labels.size();
    }

    //start timer...
    auto start_time = std::chrono::high_resolution_clock::now();

    for(size_t i=0; i<num_tests; ++i){
        Tensor input=test_data.at(i);
        Tensor output = nn.predict(input);

        int pred_class = 0;
        float max_prob = output(0,0,0,0);

        for(int c=1; c<10; ++c){
            if(output(0, c, 0, 0) > max_prob){
                max_prob = output(0, c, 0, 0);
                pred_class = c;
            }
        }

        if(have_labels && pred_class == static_cast<int>(test_labels[i])){
            num_correct++;
        }

        if ((i+1)%100 ==0){
            std::cout <<"processed "<< (i+1) <<"images.";
            if(have_labels){
                std::cout <<" Accuracy:" <<(100.0f * num_correct/(i+1)) <<"%";
            }

            std::cout <<std::endl;

        }
    }

    //stop the clock
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    if(have_labels){
        std::cout << "Final Accuracy:" <<(100.0f * num_correct/num_tests) << "%" <<std::endl;
    } else {
        std::cout <<"Inference completed on " <<num_tests <<" images." <<std::endl;
    }

    std::cout << " Total Time:     " << duration.count() << " ms" << std::endl;
    std::cout << " Time per Image: " << (duration.count() / num_tests) << " ms" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    



    return 0;
}










