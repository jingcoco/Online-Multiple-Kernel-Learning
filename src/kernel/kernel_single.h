

#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_single : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_single(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_single();
    classifier<FeatType, LabelType> * classifier_index;

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
};

template <typename FeatType, typename LabelType>
kernel_single<FeatType, LabelType>::kernel_single(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    switch (param.single_classifier)
    {
    case 0:
        this->classifier_index = gau + 0;
        break;
    case 1:
        this->classifier_index = gau + 1;
        break;
    case 2:
        this->classifier_index = gau + 2;
        break;
    case 3:
        this->classifier_index = gau + 3;
        break;
    case 4:
        this->classifier_index = gau + 4;
        break;
    case 5:
        this->classifier_index = gau + 5;
        break;
    case 6:
        this->classifier_index = gau + 6;
        break;
    case 7:
        this->classifier_index = gau + 7;
        break;
    case 8:
        this->classifier_index = gau + 8;
        break;
    case 9:
        this->classifier_index = gau + 9;
        break;
    case 10:
        this->classifier_index = gau + 10;
        break;
    case 11:
        this->classifier_index = gau + 11;
        break;
    case 12:
        this->classifier_index = gau + 12;
        break;
    case 13:
        this->classifier_index = pol + 0;
        break;
    case 14:
        this->classifier_index = pol + 1;
        break;
    case 15:
        this->classifier_index = pol + 2;
        break;
    default:
        cout << "err_classifier" << endl;
        exit(0);
    }

}

template <typename FeatType, typename LabelType>
kernel_single<FeatType, LabelType>::~kernel_single()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_single<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;


    y = classifier_index->Predict(x);

    if (x.label*y<= 0)
    {
        SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
        classifier_index->add_SV(support);
    }
    return y;
}

}













































