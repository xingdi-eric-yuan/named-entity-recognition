#include "weight_init.h"

using namespace cv;
using namespace std;

void
weightRandomInit(ConvK &convk, int height, int width){
    
    convk.W = Mat::ones(height, width, CV_64FC1);
    randu(convk.W, Scalar(-1.0), Scalar(1.0));
    convk.b = 0.0;
    convk.Wgrad = Mat::zeros(height, width, CV_64FC1);
    convk.bgrad = 0.0;
    convk.Wd2 = Mat::zeros(convk.W.size(), CV_64FC1);
    convk.bd2 = 0.0;
    
    double epsilon = 0.05;
    convk.W = convk.W * epsilon;
    convk.lr_w = lrate_w;
    convk.lr_b = lrate_b;
}

void
weightRandomInit(Fcl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;
    ntw.W = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.W, Scalar(-1.0), Scalar(1.0));
    ntw.W = ntw.W * epsilon;
    ntw.b = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Wgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.bgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Wd2 = Mat::zeros(ntw.W.size(), CV_64FC1);
    ntw.bd2 = Mat::zeros(ntw.b.size(), CV_64FC1);
    ntw.lr_w = lrate_w;
    ntw.lr_b = lrate_b;
}

void 
weightRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    smr.W = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W, Scalar(-1.0), Scalar(1.0));
    smr.W = smr.W * epsilon;
    smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
    smr.cost = 0.0;
    smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.bgrad = Mat::zeros(nclasses, 1, CV_64FC1);
    smr.Wd2 = Mat::zeros(smr.W.size(), CV_64FC1);
    smr.bd2 = Mat::zeros(smr.b.size(), CV_64FC1);
    smr.lr_w = lrate_w;
    smr.lr_b = lrate_b;
}

void
ConvNetInitPrarms(std::vector<Cvl> &ConvLayers, std::vector<Fcl> &HiddenLayers, Smr &smr, int imgHeight, int imgWidth){
    // Init Conv layers
    for(int i = 0; i < convConfig.size(); i++){
        Cvl tpcvl;
        for(int j = 0; j < convConfig[i].KernelAmount; j++){
            ConvK tmpConvK;
            weightRandomInit(tmpConvK, convConfig[i].KernelHeight, convConfig[i].KernelWidth);
            tpcvl.layer.push_back(tmpConvK);
        }
        ConvLayers.push_back(tpcvl);
    }
    // Init Hidden layers
    int outHeight = imgHeight;
    int outWidth = imgWidth;
    for(int i = 0; i < convConfig.size(); i++){
        outHeight = outHeight - convConfig[i].KernelHeight + 1;
        outWidth = outWidth - convConfig[i].KernelWidth + 1;
        outHeight = outHeight / convConfig[i].PoolingVert;
        outWidth = outWidth / convConfig[i].PoolingHori;
    }
    int hiddenfeatures = outHeight * outWidth;
    for(int i = 0; i < ConvLayers.size(); i++){
        hiddenfeatures *= convConfig[i].KernelAmount;
    }
    if(fcConfig.size() > 0){
        Fcl tpntw; 
        weightRandomInit(tpntw, hiddenfeatures, fcConfig[0].NumHiddenNeurons);
        HiddenLayers.push_back(tpntw);
        for(int i = 1; i < fcConfig.size(); i++){
            Fcl tpntw2;
            weightRandomInit(tpntw2, fcConfig[i - 1].NumHiddenNeurons, fcConfig[i].NumHiddenNeurons);
            HiddenLayers.push_back(tpntw2);
        }
    }
    // Init Softmax layer
    if(fcConfig.size() == 0){
        weightRandomInit(smr, softmaxConfig.NumClasses, hiddenfeatures);
    }else{
        weightRandomInit(smr, softmaxConfig.NumClasses, fcConfig[fcConfig.size() - 1].NumHiddenNeurons);
    }
}














