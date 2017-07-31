
#pragma once
#include "../data/DataPoint.h"
#include "../data/DataSet.h"
#include "../loss/LossFunction.h"
#include "../common/init_param.h"
#include "../common/util.h"
#include "../kernel/kernel.h"

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <math.h>

namespace SOL
{

/**
*  namespace: Kernel Online Learning
*/

template <typename FeatType, typename LabelType> class Kernel_optim
{

protected:
    unsigned int curIterNum;    //iteration number
    float gamma; //disaccount parameter
    DataSet<FeatType, LabelType> &dataSet;

protected:
    string id_str;

public:

    Kernel_optim(const Params &param,DataSet<FeatType, LabelType> &dataset);

    virtual ~Kernel_optim()
    {
    }
    //single classifiers
    Gaussian< FeatType, LabelType > gau[13];
    Poly< FeatType, LabelType > pol[3];
    float Train();
protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) = 0;
};


template <typename FeatType, typename LabelType>
Kernel_optim<FeatType, LabelType>::Kernel_optim(const Params &param, DataSet<FeatType, LabelType> &dataset) : dataSet(dataset)
{
    this->curIterNum = 0;

    this->gamma = param.gamma;
    //Init the classifier
    gau[0] = Gaussian< FeatType, LabelType >(pow(2, -6), param.eta, param.alpha, param.beta, param.weight_start);
    gau[1] = Gaussian< FeatType, LabelType >(pow(2, -5), param.eta, param.alpha, param.beta, param.weight_start);
    gau[2] = Gaussian< FeatType, LabelType >(pow(2, -4), param.eta, param.alpha, param.beta, param.weight_start);
    gau[3] = Gaussian< FeatType, LabelType >(pow(2, -3), param.eta, param.alpha, param.beta, param.weight_start);
    gau[4] = Gaussian< FeatType, LabelType >(pow(2, -2), param.eta, param.alpha, param.beta, param.weight_start);
    gau[5] = Gaussian< FeatType, LabelType >(pow(2, -1), param.eta, param.alpha, param.beta, param.weight_start);
    gau[6] = Gaussian< FeatType, LabelType >(pow(2, 0), param.eta, param.alpha, param.beta, param.weight_start);
    gau[7] = Gaussian< FeatType, LabelType >(pow(2, 1), param.eta, param.alpha, param.beta, param.weight_start);
    gau[8] = Gaussian< FeatType, LabelType >(pow(2, 2), param.eta, param.alpha, param.beta, param.weight_start);
    gau[9] = Gaussian< FeatType, LabelType >(pow(2, 3), param.eta, param.alpha, param.beta, param.weight_start);
    gau[10] = Gaussian< FeatType, LabelType >(pow(2, 4), param.eta, param.alpha, param.beta, param.weight_start);
    gau[11] = Gaussian< FeatType, LabelType >(pow(2, 5), param.eta, param.alpha, param.beta, param.weight_start);
    gau[12] = Gaussian< FeatType, LabelType >(pow(2, 6), param.eta, param.alpha, param.beta, param.weight_start);

    pol[0] = Poly< FeatType, LabelType >(1, param.eta, param.alpha, param.beta, param.weight_start);
    pol[1] = Poly< FeatType, LabelType >(2, param.eta, param.alpha, param.beta, param.weight_start);
    pol[2] = Poly< FeatType, LabelType >(3, param.eta, param.alpha, param.beta, param.weight_start);
}


template <typename FeatType, typename LabelType>
float  Kernel_optim<FeatType, LabelType>::Train()
{
    FeatType errorNum=0;
    if(dataSet.Rewind() == false)
        return 1.f;
    //reset
    while(1)
    {
        const DataChunk<FeatType,LabelType> &chunk = dataSet.GetChunk();
        //all the data has been processed!
        if(chunk.dataNum  == 0)
            break;

        for (size_t i = 0; i < chunk.dataNum; i++)
        {
            if(curIterNum%10000==0)
                cout<<curIterNum<<"\t"<<flush;

            this->curIterNum++;
            const DataPoint<FeatType, LabelType> &data = chunk.data[i];

            FeatType y = this->UpdateWeightVec(data); //key step of an algorithm
            //loss
            if (data.label*y<=0)
            {
                errorNum++;
            }
        }
        dataSet.FinishRead();
    }
    cout<<"\n#Training Instances:"<<curIterNum<<endl;

    //print out SV
    int SV_sum = 0;

    cout << "\n Gaussian kernel #SV,weight" << endl;
    for (int i = 0; i < 13; i++)
    {
        SV_sum = SV_sum + gau[i].size_SV;
        cout << gau[i].size_SV<<"\t"<<gau[i].weight_classifier<<"\terr\t"<<gau[i].err_num<<endl;
    }

    cout << "\n Polynomial kernel #SV,weight"  << endl;
    for (int i = 0; i < 3; i++)
    {
        SV_sum = SV_sum + pol[i].size_SV;
		cout << pol[i].size_SV << "\t" << pol[i].weight_classifier << "\terr\t" << pol[i].err_num << endl;
    }

    cout << "\n" << "#SV: \t" << SV_sum << endl;

    return float(errorNum / dataSet.size());
}

}
