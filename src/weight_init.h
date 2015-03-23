#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void weightRandomInit(ConvK&, int, int);

void weightRandomInit(Fcl&, int, int);

void weightRandomInit(Smr&, int, int);

void ConvNetInitPrarms(std::vector<Cvl>&, std::vector<Fcl>&, Smr&, int, int);