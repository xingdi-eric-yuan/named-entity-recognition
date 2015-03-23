#include "convolution.h"


Point
findLoc(const Mat &prob, int m){
    Mat temp, idx;
    Point res = Point(0, 0);
    prob.reshape(0, 1).copyTo(temp); 
    sortIdx(temp, idx, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
    int i = idx.at<int>(0, m);
    res.x = i % prob.rows;
    res.y = i / prob.rows;
    temp.release();
    idx.release();
    return res;
}

Mat
Pooling(const Mat &M, int pVert, int pHori, int poolingMethod, std::vector<Point> &locat){
    if(pVert == 1 && pHori == 1){
        locat.push_back(Point(0, 0));
        Mat res;
        M.copyTo(res);
        return res;
    }
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX / 2, remY / 2, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            double val = 0.0;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                double minVal = 0.0;
                double maxVal = 0.0;
                Point minLoc; 
                Point maxLoc;
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
                val = maxVal;
                locat.push_back(Point(maxLoc.x + j * pHori, maxLoc.y + i * pVert));
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp)[0] / (pVert * pHori);
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                double sumval = sum(temp)[0];
                Mat prob = temp.mul(Reciprocal(sumval));
                int ran = std::rand() % (temp.rows * temp.cols);
                Point loc = findLoc(prob, ran);
                val = temp.ATD(loc.y, loc.x);
                locat.push_back(Point(loc.x + j * pHori, loc.y + i * pVert));            
                prob.release();
            }
            res.ATD(i, j) = val;
            temp.release();
        }
    }
    newM.release();
    return res;
}


Mat
Pooling(const Mat &M, int pVert, int pHori, int poolingMethod){
    if(pVert == 1 && pHori == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX / 2, remY / 2, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            double val = 0.0;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                double minVal = 0.0;
                double maxVal = 0.0;
                Point minLoc; 
                Point maxLoc;
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
                val = maxVal;
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp)[0] / (pVert * pHori);
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                double sumval = sum(temp)[0];
                Mat prob = temp.mul(Reciprocal(sumval));
                val = sum(prob.mul(temp))[0];
                prob.release();
            }
            res.ATD(i, j) = val;
            temp.release();
        }
    }
    newM.release();
    return res;
}


Mat 
UnPooling(const Mat &M, int pVert, int pHori, int poolingMethod, std::vector<Point> &locat){
    if(pVert == 1 && pHori == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    Mat res;
    if(POOL_MEAN == poolingMethod){
        Mat one = Mat::ones(pVert, pHori, CV_64FC1);
        res = kron(M, one) / (pVert * pHori);
        one.release();
    }elif(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod){
        res = Mat::zeros(M.rows * pVert, M.cols * pHori, CV_64FC1);
        for(int i = 0; i < M.rows; i++){
            for(int j = 0; j < M.cols; j++){
                res.ATD(locat[i * M.cols + j].y, locat[i * M.cols + j].x) = M.ATD(i, j);
            }
        }
    }
    return res;
}

Mat
localResponseNorm(const unordered_map<string, Mat> &map, string str){

    int current_kernel_num = getCurrentKernelNum(str);
    int current_layer_num = getCurrentLayerNum(str);
    string current_layer = getCurrentLayer(str);

    Mat res;
    map.at(current_layer + "K" + i2str(current_kernel_num)).copyTo(res);
    Mat sum = Mat::zeros(res.rows, res.cols, CV_64FC1);

    int from, to;
    if(convConfig[current_layer_num].KernelAmount < lrn_size){
        from = 0;
        to = convConfig[current_layer_num].KernelAmount - 1;
    }else{
        from = (current_kernel_num - lrn_size / 2) >= 0 ? (current_kernel_num - lrn_size / 2) : 0;
        to = (current_kernel_num + lrn_size / 2) <= (convConfig[current_layer_num].KernelAmount - 1) ? 
             (current_kernel_num + lrn_size / 2) : (convConfig[current_layer_num].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        string tmpstr = current_layer + "K" + i2str(j);
        Mat tmpmat;
        map.at(tmpstr).copyTo(tmpmat);
        sum += pow(tmpmat, 2.0);
        tmpmat.release();
    }
    sum = sum * lrn_scale + lrn_k;
    divide(res, pow(sum, lrn_beta), res);
    sum.release();
    return res;
}

Mat
localResponseNorm(const std::vector<std::vector<Mat> > &vec, int cl, int k, int s, int m){
    Mat res;
    //(tpvec, cl, k, s, m);
    vec[s][m * convConfig[cl].KernelAmount + k].copyTo(res);
    Mat sum = Mat::zeros(res.rows, res.cols, CV_64FC1);
    int from, to;
    if(convConfig[cl].KernelAmount < lrn_size){
        from = 0;
        to = convConfig[cl].KernelAmount - 1;
    }else{
        from = (k - lrn_size / 2) >= 0 ? (k - lrn_size / 2) : 0;
        to = (k + lrn_size / 2) <= (convConfig[cl].KernelAmount - 1) ? 
             (k + lrn_size / 2) : (convConfig[cl].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        Mat tmpmat;
        vec[s][m * convConfig[cl].KernelAmount + j].copyTo(tmpmat);
        sum += pow(tmpmat, 2.0);
        tmpmat.release();
    }
    sum = sum * lrn_scale + lrn_k;
    divide(res, pow(sum, lrn_beta), res);
    sum.release();
    return res;
}

Mat
dlocalResponseNorm(const unordered_map<string, Mat> &map, string str){

    int current_kernel_num = getCurrentKernelNum(str);
    int current_layer_num = getCurrentLayerNum(str);
    string current_layer = getCurrentLayer(str);

    Mat a, da, tmp, res;
    map.at(current_layer + "K" + i2str(current_kernel_num)).copyTo(a);
    map.at(str).copyTo(da);
    Mat sum = Mat::zeros(a.rows, a.cols, CV_64FC1);
    int from, to;
    if(convConfig[current_layer_num].KernelAmount < lrn_size){
        from = 0;
        to = convConfig[current_layer_num].KernelAmount - 1;
    }else{
        from = (current_kernel_num - lrn_size / 2) >= 0 ? (current_kernel_num - lrn_size / 2) : 0;
        to = (current_kernel_num + lrn_size / 2) <= (convConfig[current_layer_num].KernelAmount - 1) ? 
             (current_kernel_num + lrn_size / 2) : (convConfig[current_layer_num].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        string tmpstr = current_layer + "K" + i2str(j);
        Mat tmpmat;
        map.at(tmpstr).copyTo(tmpmat);
        sum += pow(tmpmat, 2.0);
        tmpmat.release();
    }
    sum = sum * lrn_scale + lrn_k;

    res = da.mul(pow(sum, lrn_beta));
    tmp = a.mul(a).mul(da);
    tmp = tmp.mul(pow(sum, lrn_beta - 1)) * lrn_scale * lrn_beta * 2.0;
    res -= tmp;

    pow(sum, lrn_beta * 2, tmp);
    res = divide(res, pow(sum, lrn_beta * 2));
    a.release();
    da.release();
    sum.release();
    tmp.release();
    return res;
}

void 
convAndPooling(const std::vector<Mat> &x, const std::vector<Cvl> &CLayers, 
                unordered_map<string, Mat> &map, 
                unordered_map<string, std::vector<Point> > &loc){
    // Conv & Pooling
    int nsamples = x.size();
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        std::vector<string> vec;
        for(int cl = 0; cl < CLayers.size(); cl ++){
            int pHori = convConfig[cl].PoolingHori;
            int pVert = convConfig[cl].PoolingVert;
            if(cl == 0){
                // Convolution
                for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                    string s2 = s1 + "C0K" + i2str(k);
                    Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                    Mat tmpconv = convCalc(x[m], temp, CONV_VALID);
                    tmpconv += CLayers[cl].layer[k].b;
                    tmpconv = nonLinearity(tmpconv);
                    map[s2] = tmpconv;
                    temp.release();
                    tmpconv.release();
                }
                // Local response normalization & Pooling
                for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                    string s2 = s1 + "C0K" + i2str(k);
                    // Local response normalization
                    Mat tmpconv;
                    map.at(s2).copyTo(tmpconv);
                    //if(convConfig[cl].useLRN) tmpconv = localResponseNorm(map, s2);
                    std::vector<Point> PoolingLoc;
                    tmpconv = Pooling(tmpconv, pVert, pHori, pooling_method, PoolingLoc);
                    string s3 = s2 + "P";
                    map[s3] = tmpconv;
                    loc[s3] = PoolingLoc;
                    vec.push_back(s3);
                    tmpconv.release();
                    PoolingLoc.clear();
                }
            }else{
                std::vector<string> tmpvec;
                for(int tp = 0; tp < vec.size(); tp ++){
                    // Convolution
                    for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                        string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
                        Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                        Mat tmpconv = convCalc(map.at(vec[tp]), temp, CONV_VALID);
                        tmpconv += CLayers[cl].layer[k].b;
                        tmpconv = nonLinearity(tmpconv);
                        map[s2] = tmpconv;
                        temp.release();
                        tmpconv.release();
                    }
                    // Local response normalization & Pooling
                    for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                        string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
                        Mat tmpconv;
                        map.at(s2).copyTo(tmpconv);
                        //if(convConfig[cl].useLRN) tmpconv = localResponseNorm(map, s2);
                        std::vector<Point> PoolingLoc;
                        tmpconv = Pooling(tmpconv, pVert, pHori, pooling_method, PoolingLoc);
                        string s3 = s2 + "P";
                        map[s3] = tmpconv;
                        loc[s3] = PoolingLoc;
                        tmpvec.push_back(s3);
                        tmpconv.release();
                        PoolingLoc.clear();
                    }
                }
                swap(vec, tmpvec);
                tmpvec.clear();
            }
        }    
        vec.clear();   
    }
}

void
hashDelta(const Mat &src, unordered_map<string, Mat> &map, int layersize, int type){
    int nsamples = src.cols;
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        std::vector<string> vecstr;
        for(int i = 0; i < layersize; i ++){
            if(i == 0){
                string s2 = s1 + "C0";
                for(int k = 0; k < convConfig[i].KernelAmount; k ++){
                    string s3 = s2 + "K" + i2str(k) + "P";
                    if(i == layersize - 1){
                        if(type == HASH_DELTA) s3 += "D";
                        elif(type == HASH_HESSIAN) s3 += "H";
                    }
                    vecstr.push_back(s3);
                }
            }else{
                std::vector<string> vec2;
                for(int j = 0; j < vecstr.size(); j ++){
                    string s2 = vecstr[j] + "C" + i2str(i);
                    for(int k = 0; k < convConfig[i].KernelAmount; k ++){
                        string s3 = s2 + "K" + i2str(k) + "P";
                        if(i == layersize - 1){
                            if(type == HASH_DELTA) s3 += "D";
                            elif(type == HASH_HESSIAN) s3 += "H";
                        }
                        vec2.push_back(s3);
                    }
                }
                swap(vecstr, vec2);
                vec2.clear();
            }
        }
        int sqDim = src.rows / vecstr.size();
        int tmpWidth = convedWidth;
        int tmpHeight = sqDim / tmpWidth;
        for(int i = 0; i < vecstr.size(); i++){
            Rect roi = Rect(m, i * sqDim, 1, sqDim);
            Mat temp;
            src(roi).copyTo(temp);
            Mat img = temp.reshape(0, tmpHeight);
            map[vecstr[i]] = img;
        }  
    }
}

void 
convAndPooling(const std::vector<Mat> &x, const std::vector<Cvl> &CLayers, std::vector<std::vector<Mat> > &res){

    int nsamples = x.size();
    res.clear();
    for(int i = 0; i < nsamples; i++){
        std::vector<Mat> tmp;
        tmp.push_back(x[i]);
        res.push_back(tmp);
    }
    std::vector<std::vector<Mat> > tpvec(nsamples);
    for(int cl = 0; cl < convConfig.size(); cl++){
        for(int i = 0; i < tpvec.size(); i++){
            tpvec[i].clear();
        }
        int pHori = convConfig[cl].PoolingHori;
        int pVert = convConfig[cl].PoolingVert;
        for(int s = 0; s < nsamples; s++){
            for(int m = 0; m < res[s].size(); m++){
                for(int k = 0; k < convConfig[cl].KernelAmount; k++){
                    Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                    Mat tmpconv = convCalc(res[s][m], temp, CONV_VALID);
                    tmpconv += CLayers[cl].layer[k].b;
                    tmpconv = nonLinearity(tmpconv);
                    tpvec[s].push_back(tmpconv);
                }
                if(convConfig[cl].useLRN) {
                    std::vector<Mat>tmp;
                    for(int k = 0; k < convConfig[cl].KernelAmount; k++){
                        Mat temp = tpvec[s][m * convConfig[cl].KernelAmount + k];
                        temp = localResponseNorm(tpvec, cl, k, s, m);
                        tmp.push_back(temp);
                    }
                    for(int k = 0; k < convConfig[cl].KernelAmount; k++){
                        Mat temp = tmp[k];
                        tpvec[s][m * convConfig[cl].KernelAmount + k] = Pooling(tmp[k], pVert, pHori, pooling_method);
                    }
                }else{
                    for(int k = 0; k < convConfig[cl].KernelAmount; k++){
                        Mat temp = tpvec[s][m * convConfig[cl].KernelAmount + k];
                        tpvec[s][m * convConfig[cl].KernelAmount + k] = Pooling(temp, pVert, pHori, pooling_method);
                    }
                }
            }
        }
        swap(res, tpvec);
    }   
    for(int i = 0; i < tpvec.size(); i++){
        tpvec[i].clear();
    }  
    tpvec.clear();
}
