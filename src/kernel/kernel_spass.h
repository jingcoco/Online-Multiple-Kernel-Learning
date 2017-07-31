
#pragma once

#include "kernel_optim.h"
#include <random>
#include <iostream>
#include <time.h>
namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_spass : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_spass(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_spass();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    std::default_random_engine generator;
    FeatType p_t;
	FeatType delta;
	FeatType max_weight;
};

template <typename FeatType, typename LabelType>
kernel_spass<FeatType, LabelType>::kernel_spass(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->generator = default_random_engine((unsigned)time(NULL));
    this->p_t = 0;
	this->delta = param.delta_stoch;
	this->max_weight = 1 / 16;
}

template <typename FeatType, typename LabelType>
kernel_spass<FeatType, LabelType>::~kernel_spass()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_spass<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_weight = 0;
    FeatType sum_prediction = 0;
    FeatType loss = 0;
    FeatType tao=0;

    //gaussian
    for (int i = 0; i < 13; i++)
    {
		FeatType p_t1 = (1 - delta)*gau[i].weight_classifier / max_weight + delta;
		std::bernoulli_distribution distribution(p_t1);
		bool Z_t1 = distribution(generator);
		
		if (Z_t1)
		{
			y = gau[i].Predict(x);
			loss = 1 - y * x.label;
			sum_prediction = sum_prediction + y * gau[i].weight_classifier;

			if (loss > 0)
			{
				gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, loss);

				p_t = min(gau[i].alpha, loss) / gau[i].beta;
				std::bernoulli_distribution distribution(p_t);
				bool Z_t = distribution(generator);
				if (Z_t)
				{
					tao = min(gau[i].eta / p_t, loss);
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao, x);
					gau[i].add_SV(support);
				}
			}
        }
        sum_weight = sum_weight + gau[i].weight_classifier;
    }

 
	//pol
	for (int i = 0; i < 3; i++)
	{
		FeatType p_t1 = (1 - delta)*pol[i].weight_classifier / max_weight + delta;
		std::bernoulli_distribution distribution(p_t1);
		bool Z_t1 = distribution(generator);

		if (Z_t1)
		{
			y = pol[i].Predict(x);
			loss = 1 - y * x.label;
			sum_prediction = sum_prediction + y * pol[i].weight_classifier;

			if (loss > 0)
			{
				pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, loss);

				p_t = min(pol[i].alpha, loss) / pol[i].beta;
				std::bernoulli_distribution distribution(p_t);
				bool Z_t = distribution(generator);
				if (Z_t)
				{
					tao = min(pol[i].eta / p_t, loss);
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao, x);
					pol[i].add_SV(support);
				}
			}
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
