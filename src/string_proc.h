#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void toKey(string &, int);
std::vector<string> getKeys(int, int, int, int);
std::vector<string> getSpecKeys(int, int, int, int, int);
std::vector<string> getLayerKey(int, int, int);
std::vector<string> getLayer(int, int);
int getSampleNum(string);
int getCurrentKernelNum(string);
int getCurrentLayerNum(string);
string getCurrentLayer(string);
string getCurrentKernel(string);
string getPreviousLayerKey(string, int);