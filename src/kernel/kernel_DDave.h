

#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_DDave : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_DDave(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_DDave();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
};

template <typename FeatType, typename LabelType>
kernel_DDave<FeatType, LabelType>::kernel_DDave(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{

}

template <typename FeatType, typename LabelType>
kernel_DDave<FeatType, LabelType>::~kernel_DDave()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_DDave<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_prediction = 0;

    //gaussian
    for (int i = 0; i < 13; i++)
    {
        y = gau[i].Predict(x);
        if (y >= 0)
            sum_prediction = sum_prediction + gau[i].weight_classifier;
        else
            sum_prediction = sum_prediction - gau[i].weight_classifier;

        //record the err num of each classifier
        if (y * x.label <=0)
        {
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
            gau[i].add_SV(support);
        }
    }

    //pol
    for (int i = 0; i <3; i++)
    {
        y = pol[i].Predict(x);
        if (y >= 0)
            sum_prediction = sum_prediction + pol[i].weight_classifier;
        else
            sum_prediction = sum_prediction - pol[i].weight_classifier;

        //record the err num of each classifier
        if (y * x.label <= 0)
        {
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
            pol[i].add_SV(support);
        }
    }
    return sum_prediction;
}

}
