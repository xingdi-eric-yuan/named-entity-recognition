#include "get_sample.h"

using namespace cv;
using namespace std;

void 
getSample(const std::vector<Mat>& src1, std::vector<Mat>* dst1, const Mat& src2, Mat* dst2, int _size, int _type){
    dst1 -> clear();
    if(_type == SAMPLE_ROWS){
        if(src1.size() < _size){
            for(int i = 0; i < src1.size(); i++){
                dst1 -> push_back(src1[i]);
            }
            Rect roi = Rect(0, 0, src2.cols, src2.rows);
            src2(roi).copyTo(*dst2);
            return;
        }
        Mat m = Mat::ones(1, 1, CV_64FC1);
        randu(m, Scalar(0.0), Scalar(1.0));
        m *= (src2.rows - _size - 1);
        int randomNum = int(m.ATD(0, 0));
        for(int i = 0; i < _size; i++){
            dst1 -> push_back(src1[i + randomNum]);
        }
        Rect roi = Rect(0, randomNum, src2.cols, _size);
        src2(roi).copyTo(*dst2);
    }else{
        if(src1.size() < _size){
            for(int i = 0; i < src1.size(); i++){
                dst1 -> push_back(src1[i]);
            }
            Rect roi = Rect(0, 0, src2.cols, src2.rows);
            src2(roi).copyTo(*dst2);
            return;
        }
        random_shuffle(sample_vec.begin(), sample_vec.end());
        for(int i = 0; i < _size; i++){
            dst1 -> push_back(src1[sample_vec[i]]);
            for(int j = 0; j < src2.rows; j++){
                dst2 -> ATD(j, i) = src2.ATD(j, sample_vec[i]);
            }
        }
    }
}

void 
getSample(const std::vector<std::vector<std::string> > &src1, std::vector<Mat>* dst1, const Mat &src2, Mat *dst2, int _size, std::unordered_map<string, Mat> &wordvec){
    dst1 -> clear();
    if(src1.size() < _size){
        for(int i = 0; i < src1.size(); i++){
            dst1 -> push_back(vec2Mat(src1[i], wordvec));
        }
        Rect roi = Rect(0, 0, src2.cols, src2.rows);
        src2(roi).copyTo(*dst2);
        return;
    }
    random_shuffle(sample_vec.begin(), sample_vec.end());
    for(int i = 0; i < _size; i++){
        dst1 -> push_back(vec2Mat(src1[sample_vec[i]], wordvec));
        for(int j = 0; j < src2.rows; j++){
            dst2 -> ATD(j, i) = src2.ATD(j, sample_vec[i]);
        }
    }
}




