Memory optimisation
The storage data type of all tensors was changed from 32 bit float to int_16.

Used std::vector<int16_t> by replacing std::vector<float>
Used symmetric quantization using pre Tensor scaling factors

Model parameters: 61706
Parameter memory (float32): 246824 bytes
Parameter memory (int16):   61706 bytes
Saved: 185118 bytes (75% )

This was the result when I quantized the float to int_16 data type.
