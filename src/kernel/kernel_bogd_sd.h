
#pragma once

#include "kernel_optim.h"
#include <random>
#include <iostream>
#include <time.h>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class kernel_bogd_sd : public Kernel_optim<FeatType, LabelType>
	{
	public:
		kernel_bogd_sd(const Params &param, DataSet<FeatType, LabelType> &dataset);
		virtual ~kernel_bogd_sd();

	protected:
		virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		FeatType max_weight;
		FeatType delta;// the smoothing parameter
		std::default_random_engine generator;
		int total;
		int Budget;
		FeatType lambda;
		FeatType eta;
	};

	template <typename FeatType, typename LabelType>
	kernel_bogd_sd<FeatType, LabelType>::kernel_bogd_sd(const Params &param,
		DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
	{
		max_weight = param.weight_start;
		delta = param.delta_stoch;
		this->generator = default_random_engine((unsigned)time(NULL));
		this-> total = 0;
		this->Budget = param.budget_size;
		this->lambda = param.lambda;
		this->eta = param.eta;
	}

	template <typename FeatType, typename LabelType>
	kernel_bogd_sd<FeatType, LabelType>::~kernel_bogd_sd()
	{
		cout << total;
	}


	//Deterministic
	template <typename FeatType, typename LabelType>
	FeatType  kernel_bogd_sd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
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

			SV<FeatType, LabelType>* p_alpha = gau[i].SV_begin;
			while (p_alpha != NULL)
			{
				p_alpha->SV_alpha = p_alpha->SV_alpha*(1 - this->eta*this->lambda);
				p_alpha = p_alpha->next;
			}

			if ((y * x.label <= 0) && Z_t)
			{

				gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);
			}

			if ((y*x.label < 1)&&Z_t)
			{
				SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta, x);
				gau[i].add_SV(support);
				total++;

				if (total == this->Budget + 1)
				{
					total--;
					gau[i].delete_SV(0);
				}
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

			SV<FeatType, LabelType>* p_alpha = pol[i].SV_begin;
			while (p_alpha != NULL)
			{
				p_alpha->SV_alpha = p_alpha->SV_alpha*(1 - this->eta*this->lambda);
				p_alpha = p_alpha->next;
			}


			if ((y * x.label <= 0) && Z_t)
			{
				pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);
			}

			if ((y*x.label < 1)&&Z_t)
			{
				SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta, x);
				pol[i].add_SV(support);
				total++;

				if (total == this->Budget + 1)
				{
					total--;
					pol[i].delete_SV(0);
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
