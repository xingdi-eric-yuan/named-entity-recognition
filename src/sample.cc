#include "general_settings.h"
using namespace std;
using namespace cv;


std::vector<ConvLayerConfig> convConfig;
std::vector<FullConnectLayerConfig> fcConfig;
SoftmaxLayerConfig softmaxConfig;
std::vector<int> sample_vec;
///////////////////////////////////
// General parameters
///////////////////////////////////
bool is_gradient_checking = false;
bool use_log = false;
int log_iter = 0;
int batch_size = 1;
int pooling_method = 0;
int non_linearity = 2;
int training_epochs = 0;
double lrate_w = 0.0;
double lrate_b = 0.0;
int iter_per_epo = 0;
int convedWidth = 1;
int nGram = 3;
int word_vec_len = 300;
float training_percent = 0.8;

void
train_mitie(){
    long start, end;
    start = clock();

    readConfigFile("config.txt", false);
    std::vector<std::vector<singleWord> > trainData;
    std::vector<std::vector<singleWord> > testData;
    readDataset("dataset/news_tagged_data.txt", trainData, testData);
    cout<<"Successfully read dataset, training data size is "<<trainData.size()<<", test data size is "<<testData.size()<<endl;

    namedEntityRecognitionTrain("network/total_word_feature_extractor.dat", trainData);
    namedEntityRecognitionPredict("network/new_ner_model_800.dat", testData);

    trainData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    testData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
}

void 
train_cnn(){
    long start, end;
    start = clock();

    readConfigFile("config.txt", true);
    unordered_map<string, Mat> wordvec;
    readWordvec("dataset/wordvecs.txt", wordvec);
    cout<<"Successfully read wordvecs, map size is "<<wordvec.size()<<endl;

    std::vector<std::vector<singleWord> > trainData;
    std::vector<std::vector<singleWord> > testData;
    readDataset("dataset/news_tagged_data.txt", trainData, testData);
    cout<<"Successfully read dataset, training data size is "<<trainData.size()<<", test data size is "<<testData.size()<<endl;

    std::vector<std::vector<std::string> > trainX;
    std::vector<int> labels;
    std::unordered_map<string, int> resolmap;
    std::vector<string> re_resolmap;
    resolutioner(trainData, trainX, labels, resolmap, re_resolmap);
    cout<<"there are "<<trainX.size()<<" training data..."<<endl;
    cout<<"there are "<<resolmap.size()<<" kind of labels..."<<endl;
    Mat trainY = vec2Mat(labels);

    std::vector<std::vector<std::string> > testX;
    std::vector<int> labelsT;
    resolutionerTest(testData, testX, labelsT, resolmap);
    cout<<"there are "<<testX.size()<<" test data..."<<endl;
    Mat testY = vec2Mat(labelsT);

    int imgHeight = nGram;
    int imgWidth = word_vec_len;
    int nsamples = trainX.size();
    for(int i = 0; i < nsamples; i++){
        sample_vec.push_back(i);
    }
    std::vector<Cvl> ConvLayers;
    std::vector<Fcl> HiddenLayers;
    Smr smr;
    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgHeight, imgWidth);
    // Train network using Back Propogation
    trainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, testX, testY, wordvec);
    testNetwork(testData, ConvLayers, HiddenLayers, smr, wordvec, re_resolmap, true);

    string tmpstr = "kernel";
    mkdir(tmpstr.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    saveConvLayer(ConvLayers, "kernel/");
    save2XML("network/", "info_80", ConvLayers, HiddenLayers, smr, re_resolmap);
    ConvLayers.clear();
    HiddenLayers.clear();
    trainData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    testData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
}

void 
NER_cnn(){
    long start, end;
    start = clock();

    std::vector<std::string> sentence;
    cout<<"Type a query... (end with <Enter Key>)"<<endl;;
    readLine(sentence);

    string path = "network/info_80.xml";
    fstream _file;
    _file.open(path.c_str(), ios::in);
    if(!_file){
        cout<<"Can not find the model file..."<<endl;
        return;
    }else _file.close();

    readConfigFile("config.txt", false);
    unordered_map<string, Mat> wordvec;
    readWordvec("dataset/wordvecs.txt", wordvec);
    int imgHeight = nGram;
    int imgWidth = word_vec_len;
    std::vector<Cvl> ConvLayers;
    std::vector<Fcl> HiddenLayers;
    Smr smr;
    std::vector<string> re_resolmap;
    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgHeight, imgWidth);
    readFromXML("network/info_80.xml", ConvLayers, HiddenLayers, smr, re_resolmap);
    sentenceNER(sentence, ConvLayers, HiddenLayers, smr, wordvec, re_resolmap, true);
    ConvLayers.clear();
    HiddenLayers.clear();

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
}

void 
NER_mitie(){

    long start, end;
    start = clock();

    std::vector<std::string> sentence;
    cout<<"Type a query... (end with <Enter Key>)"<<endl;
    readLine(sentence);
    string path = "network/new_ner_model_800.dat";
    fstream _file;
    _file.open(path.c_str(), ios::in);
    if(!_file){
        cout<<"Can not find the model file..."<<endl;
        return;
    }else _file.close();
    namedEntityRecognitionPredict(path, sentence);

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
}
/*
void fun_for_debug(){

    readConfigFile("config.txt", false);
    unordered_map<string, Mat> wordvec;
    readWordvec("dataset/wordvecs.txt", wordvec);
    cout<<"Successfully read wordvecs, map size is "<<wordvec.size()<<endl;

    std::vector<std::vector<singleWord> > trainData;
    std::vector<std::vector<singleWord> > testData;
    readDataset("dataset/news_tagged_data.txt", trainData, testData);
    cout<<"Successfully read dataset, training data size is "<<trainData.size()<<", test data size is "<<testData.size()<<endl;

    int imgHeight = nGram;
    int imgWidth = word_vec_len;
    std::vector<Cvl> ConvLayers;
    std::vector<Fcl> HiddenLayers;
    Smr smr;
    std::vector<string> re_resolmap;
    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgHeight, imgWidth);
    readFromXML("network/info_80.xml", ConvLayers, HiddenLayers, smr, re_resolmap);
    
    testNetwork(testData, ConvLayers, HiddenLayers, smr, wordvec, re_resolmap, true);
    testNetwork(testData, ConvLayers, HiddenLayers, smr, wordvec, re_resolmap, false);
    ConvLayers.clear();
    HiddenLayers.clear();
    trainData.clear();
}
*/


int 
main(int argc, char** argv){

    if(argc != 2){
        cout<<"You must choose the run mode as the first command line argument"<<endl;
        cout<<" 1 : Train mitie"<<endl;
        cout<<" 2 : Train convolutional neural networks"<<endl;
        cout<<" 3 : Do named entity recognition using mitie"<<endl;
        cout<<" 4 : Do named entity recognition using convolutional neural networks"<<endl;
        return 0;
    }else{
        if(*(argv[1]) == '1') train_mitie();
        elif(*(argv[1]) == '2') train_cnn();
        elif(*(argv[1]) == '3') NER_mitie();
        else NER_cnn();
    }
    return 0;
}

