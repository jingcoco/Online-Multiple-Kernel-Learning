
#pragma once

#include "kernel_optim.h"
#include <random>
#include <iostream>
#include <time.h>

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_sd : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_sd(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_sd();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    FeatType max_weight;
    FeatType delta;// the smoothing parameter
    std::default_random_engine generator;
};

template <typename FeatType, typename LabelType>
kernel_sd<FeatType, LabelType>::kernel_sd(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    max_weight = param.weight_start;
	delta = param.delta_stoch;
    this->generator = default_random_engine((unsigned)time(NULL));
}

template <typename FeatType, typename LabelType>
kernel_sd<FeatType, LabelType>::~kernel_sd()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_sd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_weight = 0;
    FeatType sum_prediction = 0;
    FeatType p_t = 0;

    //gaussian
    for (int i = 0; i < 13; i++)
    {
		//predict
		y = gau[i].Predict(x);

		if (y >= 0)
			sum_prediction = sum_prediction + gau[i].weight_classifier;
		else
			sum_prediction = sum_prediction - gau[i].weight_classifier;

		p_t = (1 - delta)*gau[i].weight_classifier / max_weight + delta;
		std::bernoulli_distribution distribution(p_t);
		bool Z_t = distribution(generator);
		
		if ((y * x.label <= 0)&&Z_t)
        {

				gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);
				SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
                gau[i].add_SV(support);
        }
        sum_weight = sum_weight + gau[i].weight_classifier;
    }

    //pol
    for (int i = 0; i < 3; i++)
    {
		//predict
		y = pol[i].Predict(x);

		if (y >= 0)
			sum_prediction = sum_prediction + pol[i].weight_classifier;
		else
			sum_prediction = sum_prediction - pol[i].weight_classifier;
		
		p_t = (1 - delta)*pol[i].weight_classifier / max_weight + delta;
		std::bernoulli_distribution distribution(p_t);
		bool Z_t = distribution(generator);

		if ((y * x.label <= 0)&&Z_t)
		{
				pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);
				SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
				gau[i].add_SV(support);
		}
        sum_weight = sum_weight + pol[i].weight_classifier;
    }

    max_weight = 0;
    //scale the classifier weights to sum 1
    for (int i = 0; i < 13; i++)
    {
        gau[i].weight_classifier = gau[i].weight_classifier / sum_weight;
        if (gau[i].weight_classifier > max_weight)
            max_weight = gau[i].weight_classifier;
    }

    for (int i = 0; i < 3; i++)
    {
        pol[i].weight_classifier = pol[i].weight_classifier / sum_weight;
        if (pol[i].weight_classifier > max_weight)
            max_weight = pol[i].weight_classifier;
    }
    return sum_prediction;
}
}
