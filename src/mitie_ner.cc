#include "mitie_ner.h"

using namespace std;
using namespace cv;
using namespace dlib;
using namespace mitie;

void namedEntityRecognitionTrain(std::string trainer_path, 
                            std::vector<std::vector<singleWord> > &trainData){
    
    ner_trainer trainer(trainer_path);
    std::vector<std::string> sentence;
    std::vector<mitie_entity> tmpme;
    for(int i = 0; i < trainData.size(); i++){
        for(int j = 0; j < trainData[i].size(); j++){
            sentence.push_back(trainData[i][j].word);
        }
        ner_training_instance sample(sentence);
        tmpme = sentence2entities(trainData[i]);
        for(int j = 0; j < tmpme.size(); j++){
            sample.add_entity(tmpme[j].start, tmpme[j].length, tmpme[j].tag.c_str());
        }
        trainer.add(sample);
        sentence.clear();
        tmpme.clear();
    }
    // The trainer can take advantage of a multi-core CPU.  So set the number of threads
    // equal to the number of processing cores for maximum training speed.
    trainer.set_num_threads(4);
    // This function does the work of training.  Note that it can take a long time to run
    // when using larger training datasets.  So be patient.
    named_entity_extractor ner = trainer.train();
    // Now that training is done we can save the ner object to disk like so.  This will
    // allow you to load the model back in using mitie_load_named_entity_extractor("new_ner_model.dat").
    serialize("network/new_ner_model_800.dat") << "mitie::named_entity_extractor" << ner;

}

void namedEntityRecognitionPredict(std::string extractor_path,
                            std::vector<std::vector<singleWord> > &testData){
    bool flag = true;
    int correct = 0;
    int word_correct = 0;
    int total = 0;
    string classname;
    named_entity_extractor ner;
    dlib::deserialize(extractor_path) >> classname >> ner;

    // Print out what kind of tags this tagger can predict.
    const std::vector<string> tagstr = ner.get_tag_name_strings();
    cout << "The tagger supports "<< tagstr.size() <<" tags:" << endl;
    for (unsigned int i = 0; i < tagstr.size(); ++i){
        cout << "   " << tagstr[i] << endl;
    }

    std::vector<std::string> sentence;
    std::vector<pair<unsigned long, unsigned long> > chunks;
    std::vector<unsigned long> chunk_tags;
    std::vector<double> chunk_scores;
    for(int i = 0; i < testData.size(); ++i){
        for(int j = 0; j < testData[i].size(); j++){
            sentence.push_back(testData[i][j].word);
        }
        ner.predict(sentence, chunks, chunk_tags, chunk_scores);
        // If a confidence score is not necessary for your application you can detect entities
        // using the operator() method as shown in the following line.
        //ner(sentence, chunks, chunk_tags);
        flag = true;
        cout<<endl<<"******************  predicting test data number "<<i<<"..."<<endl;
        cout << "\nNumber of named entities detected: " << chunks.size() << endl;
        for (unsigned int k = 0; k < chunks.size(); ++k){
            cout << "   Tag " << chunk_tags[k] << ": ";
            cout << "Score: " << fixed << setprecision(3) << chunk_scores[k] << ": ";
            cout << tagstr[chunk_tags[k]] << ": ";
            // chunks[k] defines a half open range in sentence that contains the entity.
            for (unsigned long j = chunks[k].first; j < chunks[k].second; ++j){
                cout << sentence[j] << " ";
                if(testData[i][j].label != label2num(tagstr[chunk_tags[k]])) flag = false;
                else ++word_correct;
                ++ total;
            }
            cout << endl;
        }
        cout<<"----- ";
        if(flag){
            ++correct;
            cout<<"correct!";
        }else cout<<"incorrect...";
        cout<<endl;
        sentence.clear();
        chunks.clear();
        chunk_tags.clear();
        chunk_scores.clear();
    }

    cout<<"######################################"<<endl;
    cout<<"## mitie - Sentence test result. "<<correct<<" correct of "<<testData.size()<<" total."<<endl;
    cout<<"## Accuracy is "<<(double)correct / (double)testData.size()<<endl;
    cout<<"######################################"<<endl<<endl;

    cout<<"######################################"<<endl;
    cout<<"## mitie - Single word test result. "<<word_correct<<" correct of "<<total<<" total."<<endl;
    cout<<"## Accuracy is "<<(double)word_correct / (double)total<<endl;
    cout<<"######################################"<<endl<<endl;
}


void namedEntityRecognitionPredict(std::string extractor_path,
                            std::vector<std::string> &sentence){

    string classname;
    named_entity_extractor ner;
    dlib::deserialize(extractor_path) >> classname >> ner;

    // Print out what kind of tags this tagger can predict.
    const std::vector<string> tagstr = ner.get_tag_name_strings();
   
    std::vector<pair<unsigned long, unsigned long> > chunks;
    std::vector<unsigned long> chunk_tags;
    std::vector<double> chunk_scores;
    ner.predict(sentence, chunks, chunk_tags, chunk_scores);

    cout << "\nNumber of named entities detected: " << chunks.size() << endl;
    for (unsigned int k = 0; k < chunks.size(); ++k){
        cout << "   Tag " << chunk_tags[k] << ": ";
        cout << "Score: " << fixed << setprecision(3) << chunk_scores[k] << ": ";
        cout << tagstr[chunk_tags[k]] << ": ";
        // chunks[k] defines a half open range in sentence that contains the entity.
        for (unsigned long j = chunks[k].first; j < chunks[k].second; ++j){
            cout << sentence[j] << " ";
        }
        cout << endl;
    }

}
