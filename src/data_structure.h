#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

///////////////////////////////////
// mitie Structures
///////////////////////////////////
struct singleWord {
    std::string word;
    int label;
    singleWord(string a, int b) : word(a), label(b){}
};

struct mitie_entity{
	std::string tag;
	int start;
	int length;
	mitie_entity(string a, int b, int c) : tag(a), start(b), length(c){}
};

///////////////////////////////////
// Network Layer Structures
///////////////////////////////////
typedef struct ConvKernel{
    Mat W;
    double b;
    Mat Wgrad;
    double bgrad;
    Mat Wd2;
    double bd2;
    double lr_b;
    double lr_w;
}ConvK;

typedef struct ConvLayer{
    vector<ConvK> layer;
}Cvl;

typedef struct FullConnectLayer{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    Mat Wd2;
    Mat bd2;
    double lr_b;
    double lr_w;
}Fcl;

typedef struct SoftmaxRegession{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    double cost;
    Mat Wd2;
    Mat bd2;
    double lr_b;
    double lr_w;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
struct ConvLayerConfig {
    int KernelWidth;
    int KernelHeight;
    int KernelAmount;
    double WeightDecay;
    int PoolingHori;
    int PoolingVert;
    bool useLRN; //LocalResponseNormalization
    ConvLayerConfig(int a1, int a2, int b, double c, int d1, int d2, bool e) : KernelWidth(a1), KernelHeight(a2), KernelAmount(b), WeightDecay(c), PoolingHori(d1), PoolingVert(d2) , useLRN(e){}
};

struct FullConnectLayerConfig {
    int NumHiddenNeurons;
    double WeightDecay;
    double DropoutRate;
    FullConnectLayerConfig(int a, double b, double c) : NumHiddenNeurons(a), WeightDecay(b), DropoutRate(c) {}
};

struct SoftmaxLayerConfig {
    int NumClasses;
    double WeightDecay;
};