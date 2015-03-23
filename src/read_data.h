#pragma once
#include <fstream>
#include "general_settings.h"

using namespace std;
using namespace cv;

void readWordvec(std::string, std::unordered_map<string, Mat> &);
void readDataset(std::string, std::vector<std::vector<singleWord> >&, 
				 std::vector<std::vector<singleWord> >&);


void resolutioner(const std::vector<std::vector<singleWord> >&, std::vector<std::vector<std::string> > &, std::vector<int> &, std::unordered_map<string, int> &, std::vector<string>&);
void resolutionerTest(const std::vector<std::vector<singleWord> >&, std::vector<std::vector<std::string> > &, std::vector<int> &, std::unordered_map<string, int> &);



void readLine(std::vector<string> &);

void resolutionerTest(const std::vector<std::string> &, std::vector<std::vector<std::string> > &);
void resolutionerTest(const std::vector<singleWord> &, std::vector<std::vector<std::string> > &);