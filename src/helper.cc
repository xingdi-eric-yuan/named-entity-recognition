#include "helper.h"

using namespace std;

string getMajoriryElem(std::vector<string> &vec){
    if(vec.size() <= 2) return vec[0];
    unordered_map<string, int> map;
    for(int i = 0; i < vec.size(); i++){
        if(map.find(vec[i]) == map.end()) map[vec[i]] = 1;
        else ++map[vec[i]];
    }
    string res;
    int _max = INT_MIN;
    for(unordered_map<string, int>::iterator it = map.begin(); it != map.end(); it++){
        if(_max == INT_MIN || it -> second > _max){
            res = it -> first;
            _max = it -> second;
        }
    }
    return res;
}

int 
label2num(std::string label){
    int res = 0;
    if(label.compare("O") == 0){
        res = 0;
    }elif(label.compare("B-NEWSTYPE") == 0){
        res = 1;
    }elif(label.compare("B-PROVIDER") == 0){
        res = 2;
    }elif(label.compare("B-KEYWORDS") == 0){
        res = 3;
    }elif(label.compare("B-SECTION") == 0){
        res = 4;
    }elif(label.compare("I-NEWSTYPE") == 0){
        res = 5;
    }elif(label.compare("I-PROVIDER") == 0){
        res = 6;
    }elif(label.compare("I-KEYWORDS") == 0){
        res = 7;
    }elif(label.compare("I-SECTION") == 0){
        res = 8;
    }else{
        res = 9;
    }
    return res;
}

string 
num2label(int num){
    string res = "";
    if(num == 0){
        res = "O";
    }elif(num == 1){
        res = "B-NEWSTYPE";
    }elif(num == 2){
        res = "B-PROVIDER";
    }elif(num == 3){
        res = "B-KEYWORDS";
    }elif(num == 4){
        res = "B-SECTION";
    }elif(num == 5){
        res = "I-NEWSTYPE";
    }elif(num == 6){
        res = "I-PROVIDER";
    }elif(num == 7){
        res = "I-KEYWORDS";
    }elif(num == 8){
        res = "I-SECTION";
    }elif(num == 9){
        res = "ERROR";
    }
    return res;
}

std::vector<mitie_entity> 
sentence2entities(std::vector<singleWord> &sentence){
    std::vector<mitie_entity> res;
    for(int i = 0; i < sentence.size(); ++i){
        //if(sentence[i].label != 0){
            if(res.empty() || sentence[i].label != label2num(res[res.size() - 1].tag)){
                mitie_entity tmpme(num2label(sentence[i].label), i, 1);
                res.push_back(tmpme);
            }else{
                ++res[res.size() - 1].length;
            }
        //}
    }
    return res;
}

Mat 
vec2Mat(const std::vector<int> &labelvec){
    Mat res = Mat::zeros(1, labelvec.size(), CV_64FC1);
    for(int i = 0; i < labelvec.size(); i++){
        res.ATD(0, i) = (double)(labelvec[i]);
    }
    return res;
}

Mat 
vec2Mat(const std::vector<std::string> &resol, std::unordered_map<std::string, Mat> &wordvec){
    Mat res = Mat::zeros(nGram, word_vec_len, CV_64FC1);
    for(int i = 0; i < resol.size(); i++){
        Mat roi = res(Rect(0, i, word_vec_len, 1));
        if(wordvec.find(resol[i]) != wordvec.end()) wordvec[resol[i]].copyTo(roi);
    }
    return res;
}

// int to string
string i2str(int num){
    stringstream ss;
    ss<<num;
    string s = ss.str();
    return s;
}

// string to int
int str2i(string str){
    return atoi(str.c_str());
}

void
unconcatenateMat(const std::vector<Mat> &src, std::vector<std::vector<Mat> > *dst, int vsize){
    for(int i = 0; i < src.size() / vsize; i++){
        std::vector<Mat> tmp;
        for(int j = 0; j< vsize; j++){
            Mat img;
            src[i * vsize + j].copyTo(img);
            tmp.push_back(img);
        }
        dst -> push_back(tmp);
    }
}

void
unconcatenateMat(std::vector<Mat> &src, std::vector<std::vector<Mat> > &dst, int vsize){
    for(int i=0; i<src.size() / vsize; i++){
        std::vector<Mat> tmp;
        for(int j=0; j<vsize; j++){
            tmp.push_back(src[i * vsize + j]);
        }
        dst.push_back(tmp);
    }
}


Mat 
concatenateMat(const std::vector<std::vector<Mat> > &vec){
    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    int height = vec[0].size() * subFeatures;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);
    for(int i = 0; i < vec.size(); i++){
        for(int j = 0; j < vec[i].size(); j++){
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

Mat 
concatenateMat(const std::vector<Mat> &vec, int matcols){
    std::vector<std::vector<Mat> > temp;
    unconcatenateMat(vec, &temp, vec.size() / matcols);
    return concatenateMat(temp);
}

double 
getLearningRate(const Mat &data){
    // see Yann LeCun's Efficient BackProp, 5.1 A Little Theory
    int nfeatures = data.rows;
    int nsamples = data.cols;
    //covariance matrix = x * x' ./ nfeatures;
    Mat Sigma = data * data.t() / nsamples;
    SVD uwvT = SVD(Sigma);
    Sigma.release();
    return 0.9 / uwvT.w.ATD(0, 0);
}





