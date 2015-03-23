#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void getSample(const std::vector<Mat>&, std::vector<Mat>*, const Mat&, Mat*, int, int);

void getSample(const std::vector<std::vector<std::string> > &, std::vector<Mat>* , const Mat &, Mat *, int , std::unordered_map<string, Mat> &);