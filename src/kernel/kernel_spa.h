#define EXIT 1E-20
#pragma once

#include "kernel_optim.h"
#include <random>
#include <iostream>
#include <time.h>
namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_spa : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_spa(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_spa();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    std::default_random_engine generator;
    FeatType p_t;
};

template <typename FeatType, typename LabelType>
kernel_spa<FeatType, LabelType>::kernel_spa(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->generator = default_random_engine((unsigned)time(NULL));
    this->p_t = 0;
}

template <typename FeatType, typename LabelType>
kernel_spa<FeatType, LabelType>::~kernel_spa()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_spa<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_weight = 0;
    FeatType sum_prediction = 0;
    FeatType loss = 0;
    FeatType tao=0;

    //gaussian
    for (int i = 0; i < 13; i++)
    {
		if (gau[i].weight_classifier > EXIT)
		{
			y = gau[i].Predict(x);
			loss = 1 - y * x.label;
			sum_prediction = sum_prediction + y * gau[i].weight_classifier;

			//record the err num of each classifier
			if (y * x.label <= 0)
			{
			}


			if (loss > 0)
			{
				gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, loss);
				p_t = min(gau[i].alpha, loss) / gau[i].beta;
				// bernoulli_distribution

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
		if (pol[i].weight_classifier > EXIT)
		{
			y = pol[i].Predict(x);
			loss = 1 - y * x.label;
			sum_prediction = sum_prediction + y * pol[i].weight_classifier;

			//record the err num of each classifier
			if (y * x.label <= 0)
			{
			}


			if (loss > 0)
			{
				pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, loss);
				p_t = min(pol[i].alpha, loss) / pol[i].beta;
				// bernoulli_distribution

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


    //scale the classifier weights to sum 1
    for (int i = 0; i < 13; i++)
	{
		if (gau[i].weight_classifier > EXIT)
        gau[i].weight_classifier = gau[i].weight_classifier / sum_weight;
    }

    for (int i = 0; i < 3; i++)
    {
		if (pol[i].weight_classifier > EXIT)
        pol[i].weight_classifier = pol[i].weight_classifier / sum_weight;
    }


    return sum_prediction;
}

}
