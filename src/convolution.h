#pragma once
#include "general_settings.h"
#include <unordered_map>

using namespace std;
using namespace cv;

#include "convolution.h"


Point findLoc(const Mat &, int);

Mat Pooling(const Mat &, int , int , int , std::vector<Point > &);

Mat Pooling(const Mat &, int , int , int );


Mat UnPooling(const Mat &, int , int , int , std::vector<Point> &);
Mat localResponseNorm(const unordered_map<string, Mat> &, string );
Mat localResponseNorm(const std::vector<std::vector<Mat> > &, int , int , int , int );
Mat dlocalResponseNorm(const unordered_map<string, Mat> &, string );

void convAndPooling(const std::vector<Mat> &, const std::vector<Cvl> &, 
                unordered_map<string, Mat> &, 
                unordered_map<string, std::vector<Point> >&);

void hashDelta(const Mat &, unordered_map<string, Mat> &, int , int );
void convAndPooling(const std::vector<Mat> &, const std::vector<Cvl> &, std::vector<std::vector<Mat> > &);