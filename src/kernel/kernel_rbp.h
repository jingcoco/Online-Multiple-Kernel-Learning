

#pragma once

#include "kernel_optim.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_rbp : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_rbp(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_rbp();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    int Budget;
};

template <typename FeatType, typename LabelType>
kernel_rbp<FeatType, LabelType>::kernel_rbp(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->Budget = param.budget_size;
}

template <typename FeatType, typename LabelType>
kernel_rbp<FeatType, LabelType>::~kernel_rbp()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_rbp<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_weight = 0;
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
        if (y * x.label <= 0)
        {
            gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
            gau[i].add_SV(support);

			if (gau[i].size_SV == this->Budget + 1)
			{
				srand((unsigned)time(NULL));
				int SV_to_delete = rand() % (Budget);//from 0 to Budget-1
				gau[i].delete_SV(SV_to_delete);
			}
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
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
            pol[i].add_SV(support);


			if (pol[i].size_SV == this->Budget + 1)
			{
				srand((unsigned)time(NULL));
				int SV_to_delete = rand() % (Budget);//from 0 to Budget-1
				pol[i].delete_SV(SV_to_delete);
			}
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
