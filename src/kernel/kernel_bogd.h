

#pragma once

#include "kernel_optim.h"

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_bogd : public Kernel_optim<FeatType, LabelType>
{
public:
	kernel_bogd(const Params &param, DataSet<FeatType, LabelType> &dataset);
	virtual ~kernel_bogd();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    int Budget;
	FeatType lambda;
	FeatType eta;
};

template <typename FeatType, typename LabelType>
kernel_bogd<FeatType, LabelType>::kernel_bogd(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->Budget = param.budget_size;
	this->lambda = param.lambda;
	this->eta = param.eta;
}

template <typename FeatType, typename LabelType>
kernel_bogd<FeatType, LabelType>::~kernel_bogd()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_bogd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
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

		SV<FeatType, LabelType>* p_alpha = gau[i].SV_begin;
		while (p_alpha != NULL)
		{
			p_alpha->SV_alpha = p_alpha->SV_alpha*(1 - this->eta*this->lambda);
			p_alpha = p_alpha->next;
		}


        //record the err num of each classifier
		if (y * x.label <= 0)
		{
			gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);
		}

		if (y*x.label<=1)
		{
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta, x);
            gau[i].add_SV(support);

            if (gau[i].size_SV == this->Budget + 1)
                gau[i].delete_SV(0);
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

		SV<FeatType, LabelType>* p_alpha = pol[i].SV_begin;
		while (p_alpha != NULL)
		{
			p_alpha->SV_alpha = p_alpha->SV_alpha*(1 - this->eta*this->lambda);
			p_alpha = p_alpha->next;
		}


        //record the err num of each classifier
        if (y * x.label <= 0)
        {
            pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);
		}

		if (y*x.label <= 1)
		{
			SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta, x);
			pol[i].add_SV(support);

			if (pol[i].size_SV == this->Budget + 1)
				pol[i].delete_SV(0);
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
