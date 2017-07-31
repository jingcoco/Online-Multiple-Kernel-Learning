
#pragma once

#include "kernel_optim.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class kernel_fogd : public Kernel_optim<FeatType, LabelType>
	{
	public:
		fourier< FeatType, LabelType > fou[13];
		kernel_fogd(const Params &param, DataSet<FeatType, LabelType> &dataset);
		virtual ~kernel_fogd();

	protected:
		virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		int Budget;
		float eta;
	};

	template <typename FeatType, typename LabelType>
	kernel_fogd<FeatType, LabelType>::kernel_fogd(const Params &param,
		DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
	{
		this->id_str="fogd";
		this->Budget = param.budget_size;
		this->eta = param.eta;

		cout << "eta:" << eta <<"\t"<<"D: "<< Budget<<endl;
		for (int i = 0; i < 13; i++)
		{
			fou[i] = fourier< FeatType, LabelType >(this->eta, pow(2, -6 + i), this->Budget);
			gau[i].weight_classifier = 1.0 / 13;
		}

	}

	template <typename FeatType, typename LabelType>
	kernel_fogd<FeatType, LabelType>::~kernel_fogd()
	{
	}

	//Deterministic
	template <typename FeatType, typename LabelType>
	FeatType  kernel_fogd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		FeatType y;
		FeatType sum_weight = 0;
		FeatType sum_prediction = 0;

		//gaussian
		for (int i = 0; i < 13; i++)
		{
			y = fou[i].UpdateWeightVec(x);
			if (y >= 0)
			{
				sum_prediction = sum_prediction + gau[i].weight_classifier;
			}
			else
			{
				sum_prediction = sum_prediction - gau[i].weight_classifier;
			}

//			cout << i << "\t" << gau[i].weight_classifier << "\t" << y << endl;
			//record the err num of each classifier
			if (y * x.label <= 0)
			{
				gau[i].weight_classifier = gau[i].weight_classifier*this->gamma;
				gau[i].err_num++;
			}
			sum_weight = sum_weight + gau[i].weight_classifier;
		}




		//scale the classifier weights to sum 1
		for (int i = 0; i < 13; i++)
		{
			gau[i].weight_classifier = gau[i].weight_classifier / sum_weight;
		}
		
//		if (sum_prediction*x.label < 0)
//			cout << sum_prediction << endl;
		return sum_prediction;
	}

}
