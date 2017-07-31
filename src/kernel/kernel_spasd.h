
#pragma once
#define EXIT 1E-20

#include "kernel_optim.h"
#include <random>
#include <iostream>
#include <time.h>
namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_spasd : public Kernel_optim<FeatType, LabelType>
{
public:
    kernel_spasd(const Params &param, DataSet<FeatType, LabelType> &dataset);
    virtual ~kernel_spasd();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    std::default_random_engine generator;
    FeatType p_t;
    FeatType max_weight;
    FeatType delta;
};

template <typename FeatType, typename LabelType>
kernel_spasd<FeatType, LabelType>::kernel_spasd(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->generator = default_random_engine((unsigned)time(NULL));
    this->p_t = 0;
    this->max_weight = param.weight_start;
    this->delta = param.delta_stoch;
}

template <typename FeatType, typename LabelType>
kernel_spasd<FeatType, LabelType>::~kernel_spasd()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_spasd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{
    FeatType y;
    FeatType sum_weight = 0;
    FeatType sum_prediction = 0;
    FeatType loss = 0;
    FeatType tao=0;

    //gaussian
    for (int i = 0; i < 13; i++)
    {
        y = gau[i].Predict(x);
        loss = 1 - y * x.label;
		if (y>0)
			sum_prediction = sum_prediction + gau[i].weight_classifier;
		else
			sum_prediction = sum_prediction - gau[i].weight_classifier;

		if (y * x.label <= 0)
			gau[i].weight_classifier = gau[i].weight_classifier*this->gamma;//only 0 or 1 considered

        if (loss > 0)
        {
            FeatType p_t1 = (1 - delta)*gau[i].weight_classifier / max_weight + delta;
            
            p_t = min(gau[i].alpha, loss) / gau[i].beta;

            // bernoulli_distribution

            std::bernoulli_distribution distribution(p_t*p_t1);
            bool Z_t = distribution(generator);
            if (Z_t)
            {
                tao = min(gau[i].eta / p_t, loss);
                SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao, x);
                gau[i].add_SV(support);
            }
        }
        sum_weight = sum_weight + gau[i].weight_classifier;
    }

    //pol
    for (int i = 0; i < 3; i++)
    {

        y = pol[i].Predict(x);
        loss = 1 - y * x.label;
		if (y>0)
			sum_prediction = sum_prediction + pol[i].weight_classifier;
		else
			sum_prediction = sum_prediction - pol[i].weight_classifier;

		if (y * x.label <= 0)
			pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);//only 0 or 1 considered

        if (loss > 0)
        {
            FeatType p_t1 = (1 - delta)*pol[i].weight_classifier / max_weight + delta;
            p_t = min(pol[i].alpha, loss) / pol[i].beta;
            // bernoulli_distribution

            std::bernoulli_distribution distribution(p_t*p_t1);
            bool Z_t = distribution(generator);
            if (Z_t)
            {
                tao = min(pol[i].eta / p_t, loss);
                SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao, x);
                pol[i].add_SV(support);
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
