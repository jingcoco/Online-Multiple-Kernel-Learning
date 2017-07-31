

#pragma once

#include "kernel_optim.h"
#include "math.h"
namespace SOL
{
template <typename FeatType, typename LabelType>
class kernel_projectron : public Kernel_optim<FeatType, LabelType>
{
public:
	kernel_projectron(const Params &param, DataSet<FeatType, LabelType> &dataset);
	virtual ~kernel_projectron();

protected:
    virtual FeatType UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    int Budget;
	s_array<FeatType> K_inverse;
};

template <typename FeatType, typename LabelType>
kernel_projectron<FeatType, LabelType>::kernel_projectron(const Params &param,
        DataSet<FeatType, LabelType> &dataset) : Kernel_optim<FeatType, LabelType>(param, dataset)
{
    this->Budget = param.budget_size;
	this->K_inverse.resize(Budget*Budget*16);
	this->K_inverse.zeros();
}

template <typename FeatType, typename LabelType>
kernel_projectron<FeatType, LabelType>::~kernel_projectron()
{
}


//Deterministic
template <typename FeatType, typename LabelType>
FeatType  kernel_projectron<FeatType, LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
{	
	FeatType sum_weight = 0;
	FeatType sum_prediction = 0;

	//gaussian
	for (int l = 0; l < 13; l++)
	{
		FeatType y = 0;
		FeatType *k_t = NULL;
		//calculate k_t
		if (gau[l].size_SV != 0)//SV exist
		{
			SV<FeatType, LabelType>* p_predict = gau[l].SV_begin;
			k_t = new FeatType[gau[l].size_SV];
			int i = 0;
			while (p_predict != NULL)
			{
				k_t[i] = gau[l].kernel(p_predict->SV_data, x);
				p_predict = p_predict->next;
				i++;
			}

			//k_t done

			//get prediction
			p_predict = gau[l].SV_begin;
			i = 0;
			while (p_predict != NULL)
			{
				y += p_predict->SV_alpha* k_t[i];
				p_predict = p_predict->next;
				i++;
			}
		}
		//get prediction y and k_t

		// get sum_prediction
		if (y >= 0)
			sum_prediction = sum_prediction + gau[l].weight_classifier;
		else
			sum_prediction = sum_prediction - gau[l].weight_classifier;

		if (y * x.label <= 0)
		{
			gau[l].weight_classifier = gau[l].weight_classifier*pow(this->gamma, 1);

			if (gau[l].size_SV == 0)  // no SV, add SV ini K_inverse
			{
				SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);

				gau[l].add_SV(support);
				//ini K_inverse
				K_inverse[0 + Budget*Budget*l] = 1;
			}
			else  //have SV
			{
				// calculate d_star=K_t_inver*k_t;
				FeatType * d_star = new FeatType[gau[l].size_SV];
				for (int i = 0; i < gau[l].size_SV; i++)
				{
					d_star[i] = 0;
					for (int j = 0; j < gau[l].size_SV; j++)
					{
						d_star[i] = d_star[i] + K_inverse[i*Budget + j + Budget*Budget*l] * k_t[j];
					}
				}

				//caculate delta
				FeatType k_t_d_star = 0;
				for (int i = 0; i < gau[l].size_SV; i++)
				{
					k_t_d_star = k_t_d_star + k_t[i] * d_star[i];
				}
				FeatType delta_project = 1 - k_t_d_star;


				//full budget projectron
				if (gau[l].size_SV == Budget)
				{
					SV<FeatType, LabelType> *p_predict = gau[l].SV_begin;
					for (int i = 0; i < gau[l].size_SV; i++)
					{
						p_predict->SV_alpha = p_predict->SV_alpha + x.label*d_star[i];
						p_predict = p_predict->next;
					}
				}
				else  // not full
				{
					//add SV

					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
					gau[l].add_SV(support);


					//updata K_inverse
					for (int i = 0; i < gau[l].size_SV - 1; i++)
					{
						for (int j = 0; j < gau[l].size_SV - 1; j++)
						{
							K_inverse[i*Budget + j + Budget*Budget*l] = K_inverse[i*Budget + j + Budget*Budget*l] + d_star[i] * d_star[j] / delta_project;
						}
					}
					for (int i = 0; i < gau[l].size_SV - 1; i++)
					{
						K_inverse[i*Budget + gau[l].size_SV - 1 + Budget*Budget*l] = (-1)*d_star[i] / delta_project;
						K_inverse[(gau[l].size_SV - 1)*Budget + i + Budget*Budget*l] = (-1)*d_star[i] / delta_project;
					}
					K_inverse[(gau[l].size_SV - 1)*Budget + (gau[l].size_SV - 1) + Budget*Budget*l] = 1 / delta_project;
				}
				delete[] d_star;
			}
		}
		delete[] k_t;
		sum_weight = sum_weight + gau[l].weight_classifier;
	}


	//pol
	for (int l = 0; l < 3; l++)
	{
		FeatType y = 0;
		FeatType *k_t = NULL;
		//calculate k_t
		if (pol[l].size_SV != 0)//SV exist
		{
			SV<FeatType, LabelType>* p_predict = pol[l].SV_begin;
			k_t = new FeatType[pol[l].size_SV];
			int i = 0;
			while (p_predict != NULL)
			{
				k_t[i] = pol[l].kernel(p_predict->SV_data, x);
				p_predict = p_predict->next;
				i++;
			}

			//k_t done

			//get prediction
			p_predict = pol[l].SV_begin;
			i = 0;
			while (p_predict != NULL)
			{
				y += p_predict->SV_alpha* k_t[i];
				p_predict = p_predict->next;
				i++;
			}
		}
		//get prediction y and k_t

		// get sum_prediction
		if (y >= 0)
			sum_prediction = sum_prediction + pol[l].weight_classifier;
		else
			sum_prediction = sum_prediction - pol[l].weight_classifier;

		if (y * x.label <= 0)
		{
			pol[l].weight_classifier = pol[l].weight_classifier*pow(this->gamma, 1);

			if (pol[l].size_SV == 0)  // no SV, add SV ini K_inverse
			{
				SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);

				pol[l].add_SV(support);
				//ini K_inverse
				K_inverse[0 + Budget*Budget*(l + 13)] = 1;
			}
			else  //have SV
			{
				// calculate d_star=K_t_inver*k_t;
				FeatType * d_star = new FeatType[pol[l].size_SV];
				for (int i = 0; i < pol[l].size_SV; i++)
				{
					d_star[i] = 0;
					for (int j = 0; j < pol[l].size_SV; j++)
					{
						d_star[i] = d_star[i] + K_inverse[i*Budget + j + Budget*Budget*(l + 13)] * k_t[j];
					}
				}

				//caculate delta
				FeatType k_t_d_star = 0;
				for (int i = 0; i < pol[l].size_SV; i++)
				{
					k_t_d_star = k_t_d_star + k_t[i] * d_star[i];
				}
				FeatType delta_project = 1 - k_t_d_star;


				//full budget projectron
				if (pol[l].size_SV == Budget)
				{
					SV<FeatType, LabelType> *p_predict = pol[l].SV_begin;
					for (int i = 0; i < pol[l].size_SV; i++)
					{
						p_predict->SV_alpha = p_predict->SV_alpha + x.label*d_star[i];
						p_predict = p_predict->next;
					}
				}
				else  // not full
				{
					//add SV

					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label, x);
					pol[l].add_SV(support);


					//updata K_inverse
					for (int i = 0; i < pol[l].size_SV - 1; i++)
					{
						for (int j = 0; j < pol[l].size_SV - 1; j++)
						{
							K_inverse[i*Budget + j + Budget*Budget*(l + 13)] = K_inverse[i*Budget + j + Budget*Budget*(l + 13)] + d_star[i] * d_star[j] / delta_project;
						}
					}
					for (int i = 0; i < pol[l].size_SV - 1; i++)
					{
						K_inverse[i*Budget + pol[l].size_SV - 1 + Budget*Budget*(l + 13)] = (-1)*d_star[i] / delta_project;
						K_inverse[(pol[l].size_SV - 1)*Budget + i + Budget*Budget*(l + 13)] = (-1)*d_star[i] / delta_project;
					}
					K_inverse[(pol[l].size_SV - 1)*Budget + (pol[l].size_SV - 1) + Budget*Budget*(l + 13)] = 1 / delta_project;
				}
				delete[] d_star;
			}
		}
		delete[] k_t;
		sum_weight = sum_weight + pol[l].weight_classifier;
	}



	//scale the classifier weights to sum 1
	for (int l = 0; l < 13; l++)
	{
		gau[l].weight_classifier = gau[l].weight_classifier / sum_weight;
	}

	for (int l = 0; l < 3; l++)
	{
		pol[l].weight_classifier = pol[l].weight_classifier / sum_weight;
	}

	return sum_prediction;
}

}
