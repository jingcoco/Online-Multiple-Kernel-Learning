
#pragma once

#include "kernel_optim.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class kernel_nogd : public Kernel_optim<FeatType, LabelType>
	{
	public:
		nogd_gaussian< FeatType, LabelType > n_g[13];
		nogd_pol< FeatType, LabelType > n_p[3];
		kernel_nogd(const Params &param, DataSet<FeatType, LabelType> &dataset);
		virtual ~kernel_nogd();

	protected:
		virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	};

	template <typename FeatType, typename LabelType>
	kernel_nogd<FeatType, LabelType>::kernel_nogd(const Params &param,
		DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
	{
		cout << "eta:" << param.eta << "\t" << "B: " << param.budget_size << endl;
		for (int i = 0; i < 13; i++)
		{
			n_g[i] = nogd_gaussian< FeatType, LabelType >(param.k_nogd, param.budget_size, pow(2, i - 6), param.eta, param.alpha, param.beta, param.weight_start);
			gau[i].weight_classifier = 1.0 / 16;
		}
		for (int i = 0; i < 3; i++)
		{
			n_p[i] = nogd_pol< FeatType, LabelType >(param.k_nogd, param.budget_size, i+1, param.eta, param.alpha, param.beta, param.weight_start);
			pol[i].weight_classifier = 1.0 / 16;
		}

	}

	template <typename FeatType, typename LabelType>
	kernel_nogd<FeatType, LabelType>::~kernel_nogd()
	{
	}

	//Deterministic
	template <typename FeatType, typename LabelType>
	FeatType  kernel_nogd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		FeatType y;
		FeatType sum_weight = 0;
		FeatType sum_prediction = 0;

		//gaussian
		for (int i = 0; i < 13; i++)
		{
			y = n_g[i].UpdateWeightVec(x);
			if (y >= 0)
			{
				sum_prediction = sum_prediction + gau[i].weight_classifier;
			}
			else
			{
				sum_prediction = sum_prediction - gau[i].weight_classifier;
			}


			//record the err num of each classifier
			if (y * x.label <= 0)
			{
				gau[i].weight_classifier = gau[i].weight_classifier*this->gamma;
				gau[i].err_num++;
			}
			sum_weight = sum_weight + gau[i].weight_classifier;
		}

		//pol
		for (int i = 0; i < 3; i++)
		{
			y = n_p[i].UpdateWeightVec(x);
			if (y >= 0)
			{
				sum_prediction = sum_prediction + pol[i].weight_classifier;
			}
			else 
			{
				sum_prediction = sum_prediction - pol[i].weight_classifier;
			}

			//record the err num of each classifier
			if (y * x.label <= 0)
			{
				pol[i].weight_classifier = pol[i].weight_classifier*this->gamma;
				pol[i].err_num++;
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
