#include "read_data.h"

using namespace std;
using namespace cv;


void
readWordvec(std::string path, unordered_map<string, Mat> &wordvec){

    ifstream infile(path);
    string line;
    while (getline(infile, line)){
        istringstream iss(line);
        string tmpstr;
        iss >> tmpstr;
        Mat tmpmat = Mat::zeros(1, word_vec_len, CV_64FC1);
        double tmpdouble = 0.0;
        for(int i = 0; i < word_vec_len; ++i){
            iss >> tmpdouble;
            tmpmat.ATD(0, i) = tmpdouble;
        }
        wordvec[tmpstr] = tmpmat;
    }
}

void
readDataset(std::string path, 
    std::vector<std::vector<singleWord> >& trainData, 
    std::vector<std::vector<singleWord> >& testData){
    std::vector<std::vector<singleWord> > data;
    ifstream infile(path);
    string line;
    std::vector<singleWord> sentence;

    while (getline(infile, line)){
        if(line.empty() || line[0] == ' '){
            if(!sentence.empty()){
                data.push_back(sentence);
                sentence.clear();
            }
        }else{
            istringstream iss(line);
            string tmpword;
            string tmplabel;
            iss >> tmpword >> tmplabel;
            int tmp = label2num(tmplabel);
            singleWord tmpsw(tmpword, tmp);
            sentence.push_back(tmpsw);
        }
    }
    // random shuffle
    random_shuffle(data.begin(), data.end());
    // cross validation
    int trainSize = (int)((float)data.size() * training_percent);
    for(int i = 0; i < data.size(); ++i){
        if(i < trainSize) trainData.push_back(data[i]);
        else testData.push_back(data[i]);
    }
    data.clear();
    std::vector<std::vector<singleWord> >().swap(data);
}

void 
resolutioner(const std::vector<std::vector<singleWord> >& data, std::vector<std::vector<std::string> > &resol, std::vector<int> &labels, std::unordered_map<string, int> &resolmap, std::vector<string> &re_resolmap){
    std::vector<string> tmpvec;
    std::string tmpstr;
    int labelNum = 0;
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size() - (nGram - 1); j++){
            tmpstr = "";
            for(int k = 0; k < nGram; k++){
                tmpvec.push_back(data[i][j + k].word);
                tmpstr += num2label(data[i][j + k].label);
                if(k < nGram - 1) tmpstr += ",";
            }
            resol.push_back(tmpvec);
            if(resolmap.find(tmpstr) == resolmap.end()){
                resolmap[tmpstr] = labelNum;
                ++ labelNum;
                re_resolmap.push_back(tmpstr);
            }
            labels.push_back(resolmap[tmpstr]);
            tmpstr = "";
            tmpvec.clear();
        }
    }
}

void 
resolutionerTest(const std::vector<std::vector<singleWord> >& data, std::vector<std::vector<std::string> > &resol, std::vector<int> &labels, std::unordered_map<string, int> &resolmap){
    std::vector<string> tmpvec;
    std::string tmpstr;
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size() - (nGram - 1); j++){
            tmpstr = "";
            for(int k = 0; k < nGram; k++){
                tmpvec.push_back(data[i][j + k].word);
                tmpstr += num2label(data[i][j + k].label);
                if(k < nGram - 1) tmpstr += ",";
            }
            resol.push_back(tmpvec);
            if(resolmap.find(tmpstr) == resolmap.end()){
                labels.push_back(-1);
            }else labels.push_back(resolmap[tmpstr]);
            tmpstr = "";
            tmpvec.clear();
        }
    }
}



// read data from stdin
void 
readLine(std::vector<string> &str){
    string line;
    getline(cin, line);

    istringstream stm(line);
    string word;
    while(stm >> word) {
        str.push_back(word);
    }
}


void 
resolutionerTest(const std::vector<std::string> &data, std::vector<std::vector<std::string> > &resol){
    if(data.size() < nGram) return;
    std::vector<string> tmpvec;
    for(int j = 0; j < data.size() - (nGram - 1); j++){
        for(int k = 0; k < nGram; k++){
            tmpvec.push_back(data[j + k]);
        }
        resol.push_back(tmpvec);
        tmpvec.clear();
    }
}

void 
resolutionerTest(const std::vector<singleWord> &data, std::vector<std::vector<std::string> > &resol){
    if(data.size() < nGram) return;
    std::vector<string> tmpvec;
    for(int j = 0; j < data.size() - (nGram - 1); j++){
        for(int k = 0; k < nGram; k++){
            tmpvec.push_back(data[j + k].word);
        }
        resol.push_back(tmpvec);
        tmpvec.clear();
    }
}



