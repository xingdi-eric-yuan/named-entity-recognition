#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <mitie/ner_trainer.h>
#include "data_structure.h"
#include "read_data.h"
#include "mitie_ner.h"
#include "helper.h"

// cnn
#include "convolution.h"
#include "string_proc.h"
#include "cost_gradient.h"
#include "gradient_checking.h"
#include "helper.h"
#include "matrix_maths.h"
#include "result_predict.h"
#include "read_config.h"
#include "weights_IO.h"
#include "get_sample.h"
#include "train_network.h"
#include "weight_init.h"

//#define N_GRAM 3
//#define WORD_VEC_LEN 300

// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1 //don't use
#define POOL_STOCHASTIC 2
// get Key type
#define KEY_CONV 0
#define KEY_POOL 1
#define KEY_DELTA 2
#define KEY_UP_DELTA 3
#define KEY_HESSIAN 4
#define KEY_UP_HESSIAN 5
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
// sample
#define SAMPLE_ROWS 0
#define SAMPLE_COLS 1
// hash
#define HASH_DELTA 0
#define HASH_HESSIAN 1

#define ATD at<double>
#define elif else if
#define $$LOG if(use_log && log_iter % 1000 == 0){
#define $$_LOG }

using namespace std;
using namespace cv;
using namespace dlib;
using namespace mitie;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern float training_percent;

// Local Response Normalization
static int lrn_size = 3;
static double lrn_scale = 0.0000125;
static double lrn_beta = 0.75;
static double lrn_k = 2;

extern std::vector<ConvLayerConfig> convConfig;
extern std::vector<FullConnectLayerConfig> fcConfig;
extern SoftmaxLayerConfig softmaxConfig;
extern std::vector<int> sample_vec;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern bool is_gradient_checking;
extern bool use_log;
extern int log_iter;
extern int batch_size;
extern int pooling_method;
extern int non_linearity;
extern int training_epochs;
extern double lrate_w;
extern double lrate_b;
extern int iter_per_epo;
extern int convedWidth;
extern int nGram;
extern int word_vec_len;

