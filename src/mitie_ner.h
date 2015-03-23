#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;


void namedEntityRecognitionTrain(std::string, 
                            std::vector<std::vector<singleWord> > &);

void namedEntityRecognitionPredict(std::string,
                            std::vector<std::vector<singleWord> > &);

void namedEntityRecognitionPredict(std::string,
                            std::vector<std::string>&);
