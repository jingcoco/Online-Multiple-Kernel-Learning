
#pragma once

#include "kernel_optim.h"
#include <math.h>
#include <cmath>

namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_forgetron : public Kernel_optim<FeatType, LabelType>
{
public:
	kernel_forgetron(const Params &param, DataSet<FeatType, LabelType> &dataset);
	virtual ~kernel_forgetron();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    int Budget;
	int err_until_now [16];
	FeatType Q[16];
};

template <typename FeatType, typename LabelType>
kernel_forgetron<FeatType, LabelType>::kernel_forgetron(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->Budget = param.budget_size;
	for (int i = 0; i < 16; i++)
	{
		this->err_until_now[i] = 0;
		this->Q[i] = 0;
	}
}

template <typename FeatType, typename LabelType>
kernel_forgetron<FeatType, LabelType>::~kernel_forgetron()
{
}

//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_forgetron<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
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
			err_until_now[i]++;

			gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);
			SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
			gau[i].add_SV(support);
		}

		if (gau[i].size_SV == this->Budget + 1)
		{
			FeatType predict = gau[i].Predict(gau[i].SV_begin->SV_data);

			FeatType mu = gau[i].SV_begin->SV_data.label*predict;
			FeatType delta = gau[i].SV_begin->SV_alpha / gau[i].SV_begin->SV_data.label;

			FeatType coeA = delta*delta - 2 * delta*mu;
			FeatType coeB = 2 * delta;
			FeatType coeC = Q[i] - (15.0 / 32.0)*err_until_now[i];

			FeatType phi = 0;
			if (coeA == 0)
				phi = (std::max)(0.0, (std::min)(1.0, -coeC / coeB));
			else if (coeA>0)
			{
				if (coeA + coeB + coeC <= 0)
					phi = 1;
				else
					phi = (-coeB + sqrt(coeB*coeB - 4 * coeA*coeC)) / (2 * coeA);
			}
			else if (coeA<0)
			{
				if (coeA + coeB + coeC <= 0)
					phi = 1;
				else
					phi = (-coeB - sqrt(coeB*coeB - 4 * coeA*coeC)) / (2 * coeA);
			}

			//alpha=phi*alpha_t;
			SV<FeatType, LabelType>* p_change_alpha = gau[i].SV_begin;
			while (p_change_alpha != NULL)
			{
				p_change_alpha->SV_alpha = (float)(p_change_alpha->SV_alpha*phi);
				p_change_alpha = p_change_alpha->next;
			}

			Q[i] = Q[i] + (delta*phi)*(delta*phi) + 2 * delta*phi*(1 - phi*mu);
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


        //record the err num of each classifier
        if (y * x.label <= 0)
        {
			err_until_now[13+i]++;

            pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);
            SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
            pol[i].add_SV(support);

        }


		if (pol[i].size_SV == this->Budget + 1)
		{
			FeatType predict = pol[i].Predict(pol[i].SV_begin->SV_data);

			FeatType mu = pol[i].SV_begin->SV_data.label*predict;
			FeatType delta = pol[i].SV_begin->SV_alpha / pol[i].SV_begin->SV_data.label;

			FeatType coeA = delta*delta - 2 * delta*mu;
			FeatType coeB = 2 * delta;
			FeatType coeC = Q[i + 13] - (15.0 / 32.0)*err_until_now[i + 13];

			FeatType phi = 0;
			if (coeA == 0)
				phi = (std::max)(0.0, (std::min)(1.0, -coeC / coeB));
			else if (coeA>0)
			{
				if (coeA + coeB + coeC <= 0)
					phi = 1;
				else
					phi = (-coeB + sqrt(coeB*coeB - 4 * coeA*coeC)) / (2 * coeA);
			}
			else if (coeA<0)
			{
				if (coeA + coeB + coeC <= 0)
					phi = 1;
				else
					phi = (-coeB - sqrt(coeB*coeB - 4 * coeA*coeC)) / (2 * coeA);
			}

			//alpha=phi*alpha_t;
			SV<FeatType, LabelType>* p_change_alpha = pol[i].SV_begin;
			while (p_change_alpha != NULL)
			{
				p_change_alpha->SV_alpha = (float)(p_change_alpha->SV_alpha*phi);
				p_change_alpha = p_change_alpha->next;
			}

			Q[i+13] = Q[i+13] + (delta*phi)*(delta*phi) + 2 * delta*phi*(1 - phi*mu);
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
