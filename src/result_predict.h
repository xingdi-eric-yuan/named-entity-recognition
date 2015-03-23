#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

Mat resultPredict(const std::vector<Mat> &, const std::vector<Cvl> &, const std::vector<Fcl> &, const Smr &);
Mat getProbMatrix(const std::vector<Mat> &, const std::vector<Cvl> &, const std::vector<Fcl> &, const Smr &);


void labelJudgement(const Mat &, std::vector<string> &, std::vector<string> &);

void sentenceNER(const std::vector<std::string> &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &, std::unordered_map<string, Mat> &, std::vector<string> &, bool);
void sentenceNER(const std::vector<singleWord> &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &, std::unordered_map<string, Mat> &, std::vector<string> &, bool);

void testNetwork(const std::vector<std::vector<singleWord> > &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &, std::unordered_map<string, Mat> &, std::vector<string> &, bool);

//void testNetwork(const std::vector<Mat> &, const Mat&, const std::vector<Cvl> &, const std::vector<Fcl> &, const Smr &);
void testNetwork(const std::vector<std::vector<std::string> > &, const Mat &, std::vector<Cvl> &, std::vector<Fcl> &, Smr &, std::unordered_map<string, Mat> &);