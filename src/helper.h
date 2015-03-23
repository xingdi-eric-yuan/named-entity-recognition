#pragma once
#include "general_settings.h"

using namespace std;

string getMajoriryElem(std::vector<string> &);
// label - number look up tables
int label2num(std::string);
std::string num2label(int);

std::vector<mitie_entity> sentence2entities(std::vector<singleWord>&);

Mat vec2Mat(const std::vector<int> &);
Mat vec2Mat(const std::vector<std::string> &, std::unordered_map<std::string, Mat> &);

// int <==> string
string i2str(int);
int str2i(string);

void unconcatenateMat(const std::vector<Mat>&, std::vector<std::vector<Mat> >*, int);
Mat concatenateMat(const std::vector<std::vector<Mat> >&);
Mat concatenateMat(const std::vector<Mat>&, int );
double getLearningRate(const Mat&);