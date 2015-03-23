#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void
//trainNetwork(std::vector<Mat> &, Mat &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &);
//trainNetwork(const std::vector<Mat> &, const Mat &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &, const std::vector<Mat> &, const Mat &);
trainNetwork(const std::vector<std::vector<std::string> > &, const Mat &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &, const std::vector<std::vector<std::string> > &, const Mat &, std::unordered_map<string, Mat> &);

