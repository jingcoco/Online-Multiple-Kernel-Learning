
#pragma once

#include "kernel_optim.h"
#include <random>
#include <iostream>
#include <time.h>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class kernel_bpas_sd : public Kernel_optim<FeatType, LabelType>
	{
	public:
		kernel_bpas_sd(const Params &param, DataSet<FeatType, LabelType> &dataset);
		virtual ~kernel_bpas_sd();

	protected:
		virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		FeatType max_weight;
		FeatType delta;// the smoothing parameter
		std::default_random_engine generator;
		int total;
		int Budget;
		FeatType C_bpas;
	};

	template <typename FeatType, typename LabelType>
	kernel_bpas_sd<FeatType, LabelType>::kernel_bpas_sd(const Params &param,
		DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
	{
		max_weight = param.weight_start;
		delta = param.delta_stoch;
		this->generator = default_random_engine((unsigned)time(NULL));
		this->total = 0;
		this->Budget = param.budget_size;
		this->C_bpas = param.eta;
	}

	template <typename FeatType, typename LabelType>
	kernel_bpas_sd<FeatType, LabelType>::~kernel_bpas_sd()
	{
		cout << total;
	}


	//Deterministic
	template <typename FeatType, typename LabelType>
	FeatType  kernel_bpas_sd<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		FeatType sum_weight = 0;
		FeatType sum_prediction = 0;
		FeatType p_t = 0;

		//gaussian
		for (int i = 0; i < 13; i++)
		{
			FeatType y = 0;
			FeatType *k_t = NULL;
			if (gau[i].size_SV != 0)
			{
				SV<FeatType, LabelType>* p_predict = gau[i].SV_begin;
				k_t = new FeatType[gau[i].size_SV];
				int j = 0;
				while (p_predict != NULL)
				{
					k_t[j] = gau[i].kernel(p_predict->SV_data, x);
					p_predict = p_predict->next;
					j++;
				}

				//k_t done

				//get prediction
				p_predict = gau[i].SV_begin;
				j = 0;
				while (p_predict != NULL)
				{
					y += p_predict->SV_alpha* k_t[j];
					p_predict = p_predict->next;
					j++;
				}
			}


			if (y >= 0)
				sum_prediction = sum_prediction + gau[i].weight_classifier;
			else
				sum_prediction = sum_prediction - gau[i].weight_classifier;

			p_t = (1 - delta)*gau[i].weight_classifier / max_weight + delta;
			std::bernoulli_distribution distribution(p_t);
			bool Z_t = distribution(generator);

			if ((y * x.label <= 0) && Z_t)///////////////////////////////
			{

				gau[i].weight_classifier = gau[i].weight_classifier*pow(this->gamma, 1);
			}

			FeatType l_t = 1 - x.label*y;
			if ((l_t > 0)&&Z_t)
			{

				FeatType tao = (C_bpas < l_t)*C_bpas + (C_bpas >= l_t)*l_t;
				if (total < this->Budget)
				{
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao, x);
					gau[i].add_SV(support);
					total++;
				}
				else  //full Budget
				{
					FeatType Q_star = 1000000;
					int star = 1;
					FeatType star_alpha = 1.0;

					SV<FeatType, LabelType> *p_search = gau[i].SV_begin;

					for (int j = 0; j < gau[i].size_SV; j++)
					{
						FeatType k_rt = k_t[j];
						FeatType alpha_r = p_search->SV_alpha;
						FeatType beta_t = alpha_r*k_rt + tao*x.label;
						FeatType distance = alpha_r*alpha_r + beta_t*beta_t - 2 * beta_t*alpha_r*k_rt;
						FeatType f_rt = y - alpha_r*k_rt + beta_t;
						FeatType l_rt = 1 - x.label*f_rt;
						if (l_rt < 0)
							l_rt = 0;
						FeatType Q_r = 0.5*distance + C_bpas*l_rt;
						if (Q_r < Q_star)
						{
							Q_star = Q_r;
							star = j;
							star_alpha = beta_t;
						}
						p_search = p_search->next;
					}
					gau[i].delete_SV(star);
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(FeatType(star_alpha), x);
					gau[i].add_SV(support);
				}				
			}
			delete[] k_t;
			sum_weight = sum_weight + gau[i].weight_classifier;
		}

		//pol
		for (int i = 0; i < 3; i++)
		{
			FeatType y = 0;
			FeatType *k_t = NULL;
			if (pol[i].size_SV != 0)
			{
				SV<FeatType, LabelType>* p_predict = pol[i].SV_begin;
				k_t = new FeatType[pol[i].size_SV];
				int j = 0;
				while (p_predict != NULL)
				{
					k_t[j] = pol[i].kernel(p_predict->SV_data, x);
					p_predict = p_predict->next;
					j++;
				}

				//k_t done

				//get prediction
				p_predict = pol[i].SV_begin;
				j = 0;
				while (p_predict != NULL)
				{
					y += p_predict->SV_alpha* k_t[j];
					p_predict = p_predict->next;
					j++;
				}
			}


			if (y >= 0)
				sum_prediction = sum_prediction + pol[i].weight_classifier;
			else
				sum_prediction = sum_prediction - pol[i].weight_classifier;

			p_t = (1 - delta)*pol[i].weight_classifier / max_weight + delta;
			std::bernoulli_distribution distribution(p_t);
			bool Z_t = distribution(generator);

			if ((y * x.label <= 0) && Z_t)//////////////////////////////Note the Z_t
			{

				pol[i].weight_classifier = pol[i].weight_classifier*pow(this->gamma, 1);
			}


			FeatType l_t = 1 - x.label*y;
			if ((l_t > 0)&&Z_t)
			{

				FeatType tao = (C_bpas < l_t)*C_bpas + (C_bpas >= l_t)*l_t;
				if (total < this->Budget)
				{
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*tao, x);
					pol[i].add_SV(support);
					total++;
				}
				else  //full Budget
				{
					FeatType Q_star = 1000000;
					int star = 1;
					FeatType star_alpha = 1.0;

					SV<FeatType, LabelType> *p_search = pol[i].SV_begin;

					for (int j = 0; j < pol[i].size_SV; j++)
					{
						FeatType k_rt = k_t[j];
						FeatType alpha_r = p_search->SV_alpha;
						FeatType beta_t = alpha_r*k_rt + tao*x.label;
						FeatType distance = alpha_r*alpha_r + beta_t*beta_t - 2 * beta_t*alpha_r*k_rt;
						FeatType f_rt = y - alpha_r*k_rt + beta_t;
						FeatType l_rt = 1 - x.label*f_rt;
						if (l_rt < 0)
							l_rt = 0;
						FeatType Q_r = 0.5*distance + C_bpas*l_rt;
						if (Q_r < Q_star)
						{
							Q_star = Q_r;
							star = j;
							star_alpha = beta_t;
						}
						p_search = p_search->next;
					}
					pol[i].delete_SV(star);
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(FeatType(star_alpha), x);
					pol[i].add_SV(support);
				}
				
			}
			delete[] k_t;
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
