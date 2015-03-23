#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void gradientChecking_ConvLayer(std::vector<Cvl>&, std::vector<Fcl>&, Smr&, std::vector<Mat>&, Mat&);

void gradientChecking_FullConnectLayer(std::vector<Cvl>&, std::vector<Fcl>&, Smr&, std::vector<Mat>&, Mat&);

void gradientChecking_SoftmaxLayer(std::vector<Cvl>&, std::vector<Fcl>&, Smr&, std::vector<Mat>&, Mat&);
