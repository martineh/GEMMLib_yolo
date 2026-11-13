#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int batch;
    int height;
    int width;
    int channels;
    float* data;
} Tensor4D;

Tensor4D create_tensor(int batch, int height, int width, int channels) {
    Tensor4D tensor;
    tensor.batch = batch;
    tensor.height = height;
    tensor.width = width;
    tensor.channels = channels;
    tensor.data = (float*)calloc(batch * height * width * channels, sizeof(float));
    return tensor;
}

void free_tensor(Tensor4D* tensor) {
    free(tensor->data);
    tensor->data = NULL;
}

void initialize_weights(float* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = (float)rand() / RAND_MAX * 0.2 - 0.1; // Small random values
    }
}

void initialize_bias(float* bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

void initialize_bn_params(float* gamma, float* beta, float* mean, float* var, int size) {
    for (int i = 0; i < size; i++) {
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
        mean[i] = 0.0f;
        var[i] = 1.0f;
    }
}

void conv2d(Tensor4D* input, Tensor4D* output, float* weights, float* bias, 
            int kernel_size, int stride, int padding) {
    int out_h = (input->height + 2*padding - kernel_size) / stride + 1;
    int out_w = (input->width + 2*padding - kernel_size) / stride + 1;
    
    for (int b = 0; b < output->batch; b++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                for (int c = 0; c < output->channels; c++) {
                    int out_idx = b*(out_h*out_w*output->channels) + h*(out_w*output->channels) + w*(output->channels) + c;
                    output->data[out_idx] = bias[c];
                    
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int in_h = h*stride + kh - padding;
                            int in_w = w*stride + kw - padding;
                            
                            if (in_h >= 0 && in_h < input->height && in_w >= 0 && in_w < input->width) {
                                for (int kc = 0; kc < input->channels; kc++) {
                                    int in_idx = b*(input->height*input->width*input->channels) + 
                                               in_h*(input->width*input->channels) + 
                                               in_w*(input->channels) + kc;
                                    int kernel_idx = c*(kernel_size*kernel_size*input->channels) + 
                                                  kh*(kernel_size*input->channels) + 
                                                  kw*(input->channels) + kc;
                                    
                                    output->data[out_idx] += input->data[in_idx] * weights[kernel_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void batch_norm(Tensor4D* input, Tensor4D* output, float* gamma, float* beta, 
                float* mean, float* var, float epsilon) {
    for (int b = 0; b < input->batch; b++) {
        for (int h = 0; h < input->height; h++) {
            for (int w = 0; w < input->width; w++) {
                for (int c = 0; c < input->channels; c++) {
                    int idx = b*(input->height*input->width*input->channels) + 
                            h*(input->width*input->channels) + 
                            w*(input->channels) + c;
                    
                    float normalized = (input->data[idx] - mean[c]) / sqrtf(var[c] + epsilon);
                    output->data[idx] = gamma[c] * normalized + beta[c];
                }
            }
        }
    }
}

void relu(Tensor4D* input, Tensor4D* output) {
    for (int i = 0; i < input->batch * input->height * input->width * input->channels; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }
}

void max_pool2d(Tensor4D* input, Tensor4D* output, int pool_size, int stride) {
    int out_h = (input->height - pool_size) / stride + 1;
    int out_w = (input->width - pool_size) / stride + 1;
    
    for (int b = 0; b < input->batch; b++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                for (int c = 0; c < input->channels; c++) {
                    float max_val = -INFINITY;
                    
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int in_h = h*stride + ph;
                            int in_w = w*stride + pw;
                            int in_idx = b*(input->height*input->width*input->channels) + 
                                       in_h*(input->width*input->channels) + 
                                       in_w*(input->channels) + c;
                            
                            if (input->data[in_idx] > max_val) {
                                max_val = input->data[in_idx];
                            }
                        }
                    }
                    
                    int out_idx = b*(out_h*out_w*input->channels) + 
                                h*(out_w*input->channels) + 
                                w*(input->channels) + c;
                    output->data[out_idx] = max_val;
                }
            }
        }
    }
}

void avg_pool2d(Tensor4D* input, Tensor4D* output, int pool_size, int stride) {
    int out_h = (input->height - pool_size) / stride + 1;
    int out_w = (input->width - pool_size) / stride + 1;
    
    for (int b = 0; b < input->batch; b++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                for (int c = 0; c < input->channels; c++) {
                    float sum = 0;
                    int count = 0;
                    
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int in_h = h*stride + ph;
                            int in_w = w*stride + pw;
                            int in_idx = b*(input->height*input->width*input->channels) + 
                                       in_h*(input->width*input->channels) + 
                                       in_w*(input->channels) + c;
                            
                            sum += input->data[in_idx];
                            count++;
                        }
                    }
                    
                    int out_idx = b*(out_h*out_w*input->channels) + 
                                h*(out_w*input->channels) + 
                                w*(input->channels) + c;
                    output->data[out_idx] = sum / count;
                }
            }
        }
    }
}

void dense(Tensor4D* input, Tensor4D* output, float* weights, float* bias) {
    for (int b = 0; b < input->batch; b++) {
        for (int o = 0; o < output->channels; o++) {
            float sum = bias[o];
            
            for (int i = 0; i < input->height * input->width * input->channels; i++) {
                int in_idx = b*(input->height*input->width*input->channels) + i;
                int w_idx = o*(input->height*input->width*input->channels) + i;
                
                sum += input->data[in_idx] * weights[w_idx];
            }
            
            int out_idx = b*(output->channels) + o;
            output->data[out_idx] = sum;
        }
    }
}

void bottleneck_block(Tensor4D* input, Tensor4D* output, 
                     float conv1_weights[], float conv1_bias[],
                     float bn1_gamma[], float bn1_beta[], float bn1_mean[], float bn1_var[],
                     float conv2_weights[], float conv2_bias[],
                     float bn2_gamma[], float bn2_beta[], float bn2_mean[], float bn2_var[],
                     float conv3_weights[], float conv3_bias[],
                     float bn3_gamma[], float bn3_beta[], float bn3_mean[], float bn3_var[],
                     float conv_shortcut_weights[], float conv_shortcut_bias[],
                     float bn_shortcut_gamma[], float bn_shortcut_beta[], 
                     float bn_shortcut_mean[], float bn_shortcut_var[],
                     int stride) {
    
    Tensor4D conv1_out = create_tensor(input->batch, input->height, input->width, 64);
    Tensor4D bn1_out = create_tensor(input->batch, input->height, input->width, 64);
    Tensor4D relu1_out = create_tensor(input->batch, input->height, input->width, 64);
    
    Tensor4D conv2_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 64);
    Tensor4D bn2_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 64);
    Tensor4D relu2_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 64);
    
    Tensor4D conv3_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 256);
    Tensor4D bn3_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 256);
    
    Tensor4D shortcut_conv_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 256);
    Tensor4D shortcut_bn_out = create_tensor(input->batch, (input->height+1)/stride, (input->width+1)/stride, 256);
    
    // 1x1 convolution
    conv2d(input, &conv1_out, conv1_weights, conv1_bias, 1, stride, 0);
    batch_norm(&conv1_out, &bn1_out, bn1_gamma, bn1_beta, bn1_mean, bn1_var, 1e-5);
    relu(&bn1_out, &relu1_out);
    
    // 3x3 convolution
    conv2d(&relu1_out, &conv2_out, conv2_weights, conv2_bias, 3, 1, 1);
    batch_norm(&conv2_out, &bn2_out, bn2_gamma, bn2_beta, bn2_mean, bn2_var, 1e-5);
    relu(&bn2_out, &relu2_out);
    
    // 1x1 convolution
    conv2d(&relu2_out, &conv3_out, conv3_weights, conv3_bias, 1, 1, 0);
    batch_norm(&conv3_out, &bn3_out, bn3_gamma, bn3_beta, bn3_mean, bn3_var, 1e-5);
    
    // Shortcut connection
    if (stride != 1 || input->channels != 256) {
        conv2d(input, &shortcut_conv_out, conv_shortcut_weights, conv_shortcut_bias, 1, stride, 0);
        batch_norm(&shortcut_conv_out, &shortcut_bn_out, bn_shortcut_gamma, bn_shortcut_beta, 
                 bn_shortcut_mean, bn_shortcut_var, 1e-5);
    } else {
        // Identity shortcut
        shortcut_bn_out = *input;
    }
    
    // Add residual
    for (int i = 0; i < output->batch * output->height * output->width * output->channels; i++) {
        output->data[i] = bn3_out.data[i] + shortcut_bn_out.data[i];
    }
    
    // Final ReLU
    relu(output, output);
    
    free_tensor(&conv1_out);
    free_tensor(&bn1_out);
    free_tensor(&relu1_out);
    free_tensor(&conv2_out);
    free_tensor(&bn2_out);
    free_tensor(&relu2_out);
    free_tensor(&conv3_out);
    free_tensor(&bn3_out);
    if (stride != 1 || input->channels != 256) {
        free_tensor(&shortcut_conv_out);
        free_tensor(&shortcut_bn_out);
    }
}

Tensor4D resnet50v1_5(Tensor4D* input, int num_classes) {
    // Initialize random seed
    srand(42);
    
    // Initialize all weights and parameters
    float conv1_weights[64*7*7*3];
    float conv1_bias[64];
    float bn1_gamma[64], bn1_beta[64], bn1_mean[64], bn1_var[64];
    
    initialize_weights(conv1_weights, 64*7*7*3);
    initialize_bias(conv1_bias, 64);
    initialize_bn_params(bn1_gamma, bn1_beta, bn1_mean, bn1_var, 64);
    
    // Bottleneck blocks parameters
    int channels[4] = {256, 512, 1024, 2048};
    int blocks[4] = {3, 4, 6, 3};
    
    float* conv_weights[4][6][3];
    float* conv_bias[4][6][3];
    float* bn_gamma[4][6][3];
    float* bn_beta[4][6][3];
    float* bn_mean[4][6][3];
    float* bn_var[4][6][3];
    
    float* shortcut_weights[4];
    float* shortcut_bias[4];
    float* shortcut_gamma[4];
    float* shortcut_beta[4];
    float* shortcut_mean[4];
    float* shortcut_var[4];
    
    // Allocate and initialize all parameters
    for (int stage = 0; stage < 4; stage++) {
        int in_channels = (stage == 0) ? 64 : channels[stage-1];
        int out_channels = channels[stage];
        
        shortcut_weights[stage] = (float*)malloc(256*1*1*in_channels * sizeof(float));
        shortcut_bias[stage] = (float*)malloc(256 * sizeof(float));
        shortcut_gamma[stage] = (float*)malloc(256 * sizeof(float));
        shortcut_beta[stage] = (float*)malloc(256 * sizeof(float));
        shortcut_mean[stage] = (float*)malloc(256 * sizeof(float));
        shortcut_var[stage] = (float*)malloc(256 * sizeof(float));
        
        initialize_weights(shortcut_weights[stage], 256*1*1*in_channels);
        initialize_bias(shortcut_bias[stage], 256);
        initialize_bn_params(shortcut_gamma[stage], shortcut_beta[stage], 
                           shortcut_mean[stage], shortcut_var[stage], 256);
        
        for (int block = 0; block < blocks[stage]; block++) {
            for (int conv = 0; conv < 3; conv++) {
                int conv_size = (conv == 1) ? 3 : 1;
                int conv_channels = (conv == 2) ? out_channels : 64;
                
                conv_weights[stage][block][conv] = (float*)malloc(64*conv_size*conv_size*in_channels * sizeof(float));
                conv_bias[stage][block][conv] = (float*)malloc(64 * sizeof(float));
                bn_gamma[stage][block][conv] = (float*)malloc(64 * sizeof(float));
                bn_beta[stage][block][conv] = (float*)malloc(64 * sizeof(float));
                bn_mean[stage][block][conv] = (float*)malloc(64 * sizeof(float));
                bn_var[stage][block][conv] = (float*)malloc(64 * sizeof(float));
                
                initialize_weights(conv_weights[stage][block][conv], 64*conv_size*conv_size*in_channels);
                initialize_bias(conv_bias[stage][block][conv], 64);
                initialize_bn_params(bn_gamma[stage][block][conv], bn_beta[stage][block][conv],
                                   bn_mean[stage][block][conv], bn_var[stage][block][conv], 64);
            }
            in_channels = out_channels;
        }
    }
    
    // Final dense layer
    float fc_weights[1000*2048];
    float fc_bias[1000];
    initialize_weights(fc_weights, 1000*2048);
    initialize_bias(fc_bias, 1000);
    
    // Forward pass
    Tensor4D conv1_out = create_tensor(input->batch, input->height/2, input->width/2, 64);
    Tensor4D bn1_out = create_tensor(input->batch, input->height/2, input->width/2, 64);
    Tensor4D relu1_out = create_tensor(input->batch, input->height/2, input->width/2, 64);
    Tensor4D pool1_out = create_tensor(input->batch, input->height/4, input->width/4, 64);
    
    conv2d(input, &conv1_out, conv1_weights, conv1_bias, 7, 2, 3);
    batch_norm(&conv1_out, &bn1_out, bn1_gamma, bn1_beta, bn1_mean, bn1_var, 1e-5);
    relu(&bn1_out, &relu1_out);
    max_pool2d(&relu1_out, &pool1_out, 3, 2);
    
    Tensor4D current_input = pool1_out;
    
    for (int stage = 0; stage < 4; stage++) {
        for (int block = 0; block < blocks[stage]; block++) {
            int stride = (block == 0 && stage > 0) ? 2 : 1;
            Tensor4D block_out = create_tensor(current_input.batch, 
                                             (stride == 2) ? current_input.height/2 : current_input.height,
                                             (stride == 2) ? current_input.width/2 : current_input.width,
                                             channels[stage]);
            
            bottleneck_block(&current_input, &block_out, 
                           conv_weights[stage][block][0], conv_bias[stage][block][0],
                           bn_gamma[stage][block][0], bn_beta[stage][block][0], 
                           bn_mean[stage][block][0], bn_var[stage][block][0],
                           conv_weights[stage][block][1], conv_bias[stage][block][1],
                           bn_gamma[stage][block][1], bn_beta[stage][block][1], 
                           bn_mean[stage][block][1], bn_var[stage][block][1],
                           conv_weights[stage][block][2], conv_bias[stage][block][2],
                           bn_gamma[stage][block][2], bn_beta[stage][block][2], 
                           bn_mean[stage][block][2], bn_var[stage][block][2],
                           shortcut_weights[stage], shortcut_bias[stage],
                           shortcut_gamma[stage], shortcut_beta[stage],
                           shortcut_mean[stage], shortcut_var[stage],
                           stride);
            
            if (block > 0) free_tensor(&current_input);
            current_input = block_out;
        }
    }
    
    Tensor4D avg_pool_out = create_tensor(current_input.batch, 1, 1, current_input.channels);
    Tensor4D fc_out = create_tensor(current_input.batch, 1, 1, num_classes);
    
    avg_pool2d(&current_input, &avg_pool_out, current_input.height, 1);
    dense(&avg_pool_out, &fc_out, fc_weights, fc_bias);
    
    // Cleanup
    free_tensor(&conv1_out);
    free_tensor(&bn1_out);
    free_tensor(&relu1_out);
    free_tensor(&pool1_out);
    free_tensor(&current_input);
    free_tensor(&avg_pool_out);
    
    for (int stage = 0; stage < 4; stage++) {
        free(shortcut_weights[stage]);
        free(shortcut_bias[stage]);
        free(shortcut_gamma[stage]);
        free(shortcut_beta[stage]);
        free(shortcut_mean[stage]);
        free(shortcut_var[stage]);
        
        for (int block = 0; block < blocks[stage]; block++) {
            for (int conv = 0; conv < 3; conv++) {
                free(conv_weights[stage][block][conv]);
                free(conv_bias[stage][block][conv]);
                free(bn_gamma[stage][block][conv]);
                free(bn_beta[stage][block][conv]);
                free(bn_mean[stage][block][conv]);
                free(bn_var[stage][block][conv]);
            }
        }
    }
    
    return fc_out;
}

int main() {
    int batch_size = 1;
    int height = 224;
    int width = 224;
    int channels = 3;
    int num_classes = 1000;
    
    Tensor4D input = create_tensor(batch_size, height, width, channels);
    
    // Initialize with random values
    for (int i = 0; i < batch_size * height * width * channels; i++) {
        input.data[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
    }
    
    Tensor4D output = resnet50v1_5(&input, num_classes);
    
    printf("Output shape: %d x %d x %d x %d\n", output.batch, output.height, output.width, output.channels);
    printf("First 5 class scores:\n");
    for (int i = 0; i < 5; i++) {
        printf("Class %d: %f\n", i, output.data[i]);
    }
    
    free_tensor(&input);
    free_tensor(&output);
    
    return 0;
}

