

#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_DDpa : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_DDpa(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_DDpa();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
};

template <typename FeatType, typename LabelType>
kernel_DDpa<FeatType, LabelType>::kernel_DDpa(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{

}

template <typename FeatType, typename LabelType>
kernel_DDpa<FeatType, LabelType>::~kernel_DDpa()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_DDpa<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_weight = 0;
    FeatType sum_prediction = 0;

    //gaussian
    for (int i = 0; i < 13; i++)
    {
        y = gau[i].Predict(x);

        if (y>=0)
            sum_prediction = sum_prediction + gau[i].weight_classifier;
        else
            sum_prediction = sum_prediction - gau[i].weight_classifier;
        //record the err num of each classifier
		if (y * x.label <= 0)
		{
			gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);//only 0 or 1 considered
		}
		if (y*x.label<1)
		{
			SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(min(gau[i].eta, 1 - y*x.label)*x.label , x);
            gau[i].add_SV(support);
        }
        sum_weight = sum_weight + gau[i].weight_classifier;
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
			pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);
		}
		if (y*x.label<1)
		{
			SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(min(pol[i].eta, 1 - y*x.label)*x.label, x);
            pol[i].add_SV(support);
        }
        sum_weight = sum_weight + pol[i].weight_classifier;
    }


    //scale the classifier weights to sum 1
    for (int i = 0; i < 13; i++)
    {
        gau[i].weight_classifier = gau[i].weight_classifier / sum_weight;
    }

    for (int i = 0; i < 3; i++)
    {
        pol[i].weight_classifier = pol[i].weight_classifier / sum_weight;
    }
    return sum_prediction;
}

}
