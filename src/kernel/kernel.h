#pragma once
#include "../data/DataPoint.h"
#include "../data/DataSet.h"
#include "../loss/LossFunction.h"
#include "../common/init_param.h"
#include "../common/util.h"

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <math.h>
#define B_cos 1.273239
#define P_cos 0.225
#define C_cos -0.40528
#define pi_cos 3.1415926
#include <time.h>
#include <random>
#include <Eigen>
#include "cmath"
using namespace Eigen;

namespace SOL
{

	/**
	*  namespace: Kernel Online Learning
	*/
	template <typename FeatType, typename LabelType>
	struct SV
	{
	public:
		FeatType SV_alpha_sum;        // used for averaging the classifier
		FeatType SV_alpha;
		DataPoint<FeatType, LabelType> SV_data;
		SV * next;

		SV(FeatType alpha, DataPoint<FeatType, LabelType> x)
		{
			SV_alpha_sum = 0;
			SV_alpha = alpha;
			SV_data = x.clone();
			next = NULL;
		}
	};

	template <typename FeatType, typename LabelType> class fourier
	{
	public:
		FeatType weight_classifier;
		FeatType eta;
		int size_SV;
		FeatType wide;
		int D;
		int u_dimension;

		s_array<double> w_fogd;
		s_array<double> u;
		s_array<double> ux;
		s_array<double> ux_cos;
		double a;

		std::default_random_engine generator;
		std::normal_distribution<double> distribution;
		fourier()
		{
		}

		fourier(FeatType para_eta, FeatType para_wide, int para_D)
		{
			this->eta = para_eta;
			this->wide = para_wide;
			this->D = para_D;
			this->size_SV = 0;

			this->w_fogd.resize(2 * D);
			this->w_fogd.zeros();
			this->ux.resize(D);
			this->ux_cos.resize(2 * D);

			this->distribution = normal_distribution<double>(0.0, 1.0 / wide);
			this->generator = default_random_engine((unsigned)time(NULL));
			this->weight_classifier = 1.0 / 13;
			this->u_dimension = 0;
		}

		~fourier()
		{
		}

		template <typename FeatType, typename LabelType>
		float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
		{
			IndexType x_dimension = x.max_index;
			//generate u
			if (u_dimension < x.max_index)
			{
				//update dimension
				this->u.reserve(D * x_dimension);
				this->u.resize(D * x_dimension);
				for (IndexType i = (D*u_dimension); i < (D*x_dimension); i++)
					this->u[i] = distribution(generator);
				this->u_dimension = x_dimension;
			}

			this->ux.zeros();

			size_t index_begin;
			float feature;
			for (size_t j = 0; j < x.indexes.size(); j++)
			{
				index_begin = (x.indexes[j] - 1)*D;
				feature = x.features[j];
				for (int i = 0; i < D; i++)
				{
					ux[i] += u[index_begin] * feature;
					index_begin++;
				}
			}
			double *p1 = ux_cos.begin;
			double *p2 = p1 + D;

			for (int i = 0; i < D; i++)
			{
				ux_cos[i] = cos(ux[i]);
				ux_cos[i + D] = sin(ux[i]);
			}

			double y = 0;
			for (int i = 0; i < 2 * D; i++)
				y = y + w_fogd[i] * ux_cos[i];

			if (y*x.label < 1)
			{
				for (int i = 0; i < 2 * D; i++)
				{
					w_fogd[i] = w_fogd[i] + this->eta*x.label*ux_cos[i];
				}
			}
			return float(y);
		}


	};

	//a single kernel classifier
	template <typename FeatType, typename LabelType> class classifier
	{
	public:
		FeatType weight_classifier;
		SV<FeatType, LabelType> * SV_begin;
		SV<FeatType, LabelType> * SV_end;
		int size_SV;
		FeatType eta;
		FeatType alpha;
		FeatType beta;
		FeatType wide;
		FeatType gau_wide;
		int err_num;
	public:
		classifier() {};
		classifier(FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start)
		{
			this->alpha = alpha_para;
			this->beta = beta_para;
			this->eta = eta_para;
			this->wide = wide_para;
			this->gau_wide = 1 / (2 * wide_para*wide_para);

			this->SV_begin = NULL;
			this->SV_end = NULL;
			this->size_SV = 0;
			this->weight_classifier = weight_start;
			this->err_num = 0;
		}


		virtual ~classifier()
		{
			SV<FeatType, LabelType> * SV_free;
			for (int i = 0; i < size_SV; i++)
			{
				SV_free = SV_begin;
				SV_begin = SV_begin->next;
				delete SV_free;
			}
		}


		void sum_SV()// get alpha_sum
		{
			SV<FeatType, LabelType>* p_sum = SV_begin;
			while (p_sum != NULL)
			{
				p_sum->SV_alpha_sum = p_sum->SV_alpha_sum + p_sum->SV_alpha;
				p_sum = p_sum->next;
			}
		}


		// add a new support to the support vector set at the end
		void add_SV(SV<FeatType, LabelType> *p_newSV)
		{
			if (SV_end != NULL)
			{
				SV_end->next = p_newSV;
				SV_end = p_newSV;
			}
			else
			{
				SV_begin = p_newSV;
				SV_end = p_newSV;
			}
			this->size_SV++;
		}

		// delete a SV
		void delete_SV(int index_SV)
		{
			//index_SV is the index of SV to be deleted from 0 to B-1
			SV<FeatType, LabelType>* p_delete = SV_begin;
			SV<FeatType, LabelType>* q_delete = NULL;
			if ((index_SV != 0) && (index_SV != size_SV - 1))
			{
				int i = 0;
				while (i < index_SV - 1)
				{
					p_delete = p_delete->next;
					i++;
				}
				q_delete = p_delete->next;
				p_delete->next = q_delete->next;
				delete q_delete;
			}
			else if (index_SV == 0)
			{
				SV_begin = p_delete->next;
				delete p_delete;
			}
			else
			{
				int i = 0;
				while (i < index_SV - 1)
				{
					p_delete = p_delete->next;
					i++;
				}
				q_delete = p_delete->next;
				p_delete->next = NULL;
				delete q_delete;
				SV_end = p_delete;
			}
			size_SV--;
		}

		//calculate the kernel value
		virtual FeatType kernel(const DataPoint<FeatType, LabelType> &SV_data, const DataPoint<FeatType, LabelType> &x) = 0;

		//prediction f(x) of this single kernel
		FeatType Predict(const DataPoint<FeatType, LabelType> &data)
		{
			FeatType predict = 0;

			SV<FeatType, LabelType>* p_predict = this->SV_begin;
			while (p_predict != NULL)
			{
				predict += p_predict->SV_alpha* this->kernel(p_predict->SV_data, data);
				p_predict = p_predict->next;
			}
			return predict;
		}
	};



	//gaussian kernel
	template <typename FeatType, typename LabelType> class Gaussian : public classifier < FeatType, LabelType >
	{
	public:
		Gaussian() {};
		Gaussian(FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start) :classifier(wide_para, eta_para, alpha_para, beta_para, weight_start)
		{};
		virtual ~Gaussian() {};

	public:
		virtual FeatType kernel(const DataPoint<FeatType, LabelType> &SV_data, const DataPoint<FeatType, LabelType> &x)
		{
			FeatType sum = 0;
			int i = 0;
			int j = 0;
			int size_SV_dimension = SV_data.indexes.size();
			int size_data_dimension = x.indexes.size();


			while ((i != size_SV_dimension) && (j != size_data_dimension))
			{
				if ((SV_data.indexes[i]) > (x.indexes[j]))
				{
					sum = sum + x.features[j] * x.features[j];
					j++;
				}
				else if ((SV_data.indexes[i]) < (x.indexes[j]))
				{
					sum = sum + SV_data.features[i] * SV_data.features[i];
					i++;
				}
				else
				{
					sum = sum + (SV_data.features[i] - x.features[j])*(SV_data.features[i] - x.features[j]);
					i++;
					j++;
				}
			}
			if (i == size_SV_dimension)//i first reach the end
			{
				for (int a = j; a < size_data_dimension; a++)
				{
					sum = sum + x.features[a] * x.features[a];
				}
			}
			if (j == size_data_dimension)//i first reach the end
			{
				for (int a = i; a < size_SV_dimension; a++)
				{
					sum = sum + SV_data.features[a] * SV_data.features[a];
				}
			}


			sum = sum*(-1)*(this->gau_wide);
			FeatType a = exp(sum);
			return a;

		};
	};



	//polynomial kernel
	template <typename FeatType, typename LabelType> class Poly : public classifier < FeatType, LabelType >
	{
	public:
		Poly() {};
		Poly(FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start) :classifier(wide_para, eta_para, alpha_para, beta_para, weight_start)
		{};
		virtual ~Poly() {};
		virtual FeatType kernel(const DataPoint<FeatType, LabelType> &SV_data, const DataPoint<FeatType, LabelType> &x)
		{
			FeatType sum = 0;
			int i = 0;
			int j = 0;
			int size_SV_dimension = SV_data.indexes.size();
			int size_data_dimension = x.indexes.size();


			while ((i != size_SV_dimension) && (j != size_data_dimension))
			{
				if ((SV_data.indexes[i]) > (x.indexes[j]))
				{
					j++;
				}
				else if ((SV_data.indexes[i]) < (x.indexes[j]))
				{
					i++;
				}
				else
				{
					sum = sum + SV_data.features[i] * x.features[j];
					i++;
					j++;
				}
			}
			FeatType a = pow(sum, this->wide);
			return a;
		};
	};

		/*

	//Sigmoid kernel
	template <typename FeatType, typename LabelType> class Sig : public classifier < FeatType, LabelType >
			{
			public:
				Sig() {};
				Sig(FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start) :classifier(wide_para, eta_para, alpha_para, beta_para, weight_start)
				{};
				virtual ~Sig() {};
				virtual FeatType kernel(const DataPoint<FeatType, LabelType> &SV_data, const DataPoint<FeatType, LabelType> &x)
				{
					FeatType sum = 0;
					int i = 0;
					int j = 0;
					int size_SV_dimension = SV_data.indexes.size();
					int size_data_dimension = x.indexes.size();


					while ((i != size_SV_dimension) && (j != size_data_dimension))
					{
						if ((SV_data.indexes[i]) > (x.indexes[j]))
						{
							j++;
						}
						else if ((SV_data.indexes[i]) < (x.indexes[j]))
						{
							i++;
						}
						else
						{
							sum = sum + SV_data.features[i] * x.features[j];
							i++;
							j++;
						}
					}


					FeatType a = tanh(sum);
					return a;
				};
			};
				//Cauchy kernel
				template <typename FeatType, typename LabelType> class Cauchy : public classifier < FeatType, LabelType >
				{
				public:
					Cauchy() {};
					Cauchy(FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start) :classifier(wide_para, eta_para, alpha_para, beta_para, weight_start)
					{};
					virtual ~Cauchy() {};

					virtual FeatType kernel(const DataPoint<FeatType, LabelType> &SV_data, const DataPoint<FeatType, LabelType> &x)
					{
						FeatType sum = 0;
						int i = 0;
						int j = 0;
						int size_SV_dimension = SV_data.indexes.size();
						int size_data_dimension = x.indexes.size();


						while ((i != size_SV_dimension) && (j != size_data_dimension))
						{
							if ((SV_data.indexes[i]) > (x.indexes[j]))
							{
								sum = sum + x.features[j] * x.features[j];
								j++;
							}
							else if ((SV_data.indexes[i]) < (x.indexes[j]))
							{
								sum = sum + SV_data.features[i] * SV_data.features[i];
								i++;
							}
							else
							{
								sum = sum + (SV_data.features[i] - x.features[j])*(SV_data.features[i] - x.features[j]);
								i++;
								j++;
							}
						}
						if (i == size_SV_dimension)//i first reach the end
						{
							for (int a = j; a < size_data_dimension; a++)
							{
								sum = sum + x.features[a] * x.features[a];
							}
						}
						if (j == size_data_dimension)//i first reach the end
						{
							for (int a = i; a < size_SV_dimension; a++)
							{
								sum = sum + SV_data.features[a] * SV_data.features[a];
							}
						}


						sum = 1 / (1 + sum*(this->wide));
						return sum;
					};
				};
*/
	template <typename FeatType, typename LabelType> class nogd_gaussian : public Gaussian < FeatType, LabelType >
	{
	public:
		int Budget;
		int k_nogd;
		VectorXf w_nogd;
		MatrixXf M_nogd;
		MatrixXf K_budget;
		bool flag;
		nogd_gaussian() {};
		nogd_gaussian(int k_nogd, int Budget, FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start) :Gaussian(wide_para, eta_para, alpha_para,  beta_para, weight_start)
		{
			this->Budget = Budget;
			this->k_nogd = k_nogd;
			this->flag = 0;
			this->K_budget = MatrixXf(Budget, Budget);
			for (int i = 0; i < Budget; i++)
			{
				K_budget(i, i) = 1;
			}
			this->w_nogd = VectorXf(k_nogd);
			for (int i = 0; i < k_nogd; i++)
			{
				w_nogd(i) = 0;
			}
			this->M_nogd = MatrixXf(k_nogd, Budget);
		};

		virtual ~nogd_gaussian() 
		{
		};

		template <typename FeatType, typename LabelType>
		float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
		{
			float y = 0;
			VectorXf kt(this->size_SV);
			VectorXf zt(k_nogd);
			//calculate k_t
			if ((this->size_SV != 0) && (flag == 0))
			{
				SV<FeatType, LabelType>* p_predict = this->SV_begin;
				int i = 0;
				while (p_predict != NULL)
				{
					kt(i) = this->kernel(p_predict->SV_data, x);
					p_predict = p_predict->next;
					i++;
				}
				//k_t done

				//get prediction
				p_predict = this->SV_begin;
				i = 0;
				while (p_predict != NULL)
				{
					y += p_predict->SV_alpha* kt(i);
					p_predict = p_predict->next;
					i++;
				}
			}
			if (flag != 0) //linear predict
			{
				SV<FeatType, LabelType>* p_predict = this->SV_begin;
				int i = 0;
				while (p_predict != NULL)
				{
					kt[i] = this->kernel(p_predict->SV_data, x);
					p_predict = p_predict->next;
					i++;
				}
				zt = (M_nogd)*kt;
				y = (w_nogd).dot(zt);
			}
			//update
			if (y*x.label<1)
			{
				if (this->size_SV<Budget) //kernel update
				{
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta, x);
					this->add_SV(support);

					for (int i = 0; i<this->size_SV - 1; i++)
					{
						(K_budget)(i, this->size_SV - 1) = kt(i);
						(K_budget)(this->size_SV - 1, i) = kt(i);
					}

				}
				else
				{
					if (flag == 0) //SVD
					{
						flag = 1;
						EigenSolver<MatrixXf> es(K_budget);
						MatrixXcf V = es.eigenvectors();
						//cout<<es.eigenvalues()<<endl;

						for (int i = 0; i<k_nogd; i++)
						{
							float length = 0;
							for (int j = 0; j<Budget; j++)
							{
								length = length + V(j, i).real()*V(j, i).real();
							}
							for (int j = 0; j<Budget; j++)
							{
								V(j, i) = V(j, i) / length;
							}
						}

						for (int i = 0; i<k_nogd; i++)
						{
							if (es.eigenvalues()[i].real() <= 0)
							{
								for (int j = 0; j<Budget; j++)
								{
									(M_nogd)(i, j) = 0;
								}
							}
							else
							{
								for (int j = 0; j < Budget; j++)
								{
									(M_nogd)(i, j) = V(j, i).real() / sqrt(es.eigenvalues()[i].real());
								}
							}
						}
						zt = (M_nogd)*kt;
						(w_nogd) = (w_nogd) + eta*x.label*zt;
					}
					else
					{
						(w_nogd) = (w_nogd) + eta*x.label*zt;
					}
				}
			}
			if ((y>1e10) || (y < -1e10))
				cout << y<<"\t";

			return y;
		}
	};


	template <typename FeatType, typename LabelType> class nogd_pol : public  Poly< FeatType, LabelType >
	{
	public:
		int Budget;
		int k_nogd;
		VectorXf w_nogd;
		MatrixXf M_nogd;
		MatrixXf K_budget;
		bool flag;
		nogd_pol() {};
		nogd_pol(int k_nogd, int Budget, FeatType wide_para, FeatType eta_para, FeatType alpha_para, FeatType beta_para, FeatType weight_start) :Poly(wide_para, eta_para, alpha_para, beta_para, weight_start)
		{
			this->Budget = Budget;
			this->k_nogd = k_nogd;
			this->flag = 0;
			this->K_budget = MatrixXf(Budget, Budget);
			for (int i = 0; i < Budget; i++)
			{
				(K_budget)(i, i) = 1;
			}
			this->w_nogd = VectorXf(k_nogd);
			for (int i = 0; i < k_nogd; i++)
			{
				(w_nogd)(i) = 0;
			}
			this->M_nogd = MatrixXf(k_nogd, Budget);
		};

		virtual ~nogd_pol()
		{
		};

		template <typename FeatType, typename LabelType>
		float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
		{
			float y = 0;
			VectorXf kt(this->size_SV);
			VectorXf zt(k_nogd);
			//calculate k_t
			if ((this->size_SV != 0) && (flag == 0))
			{
				SV<FeatType, LabelType>* p_predict = this->SV_begin;
				int i = 0;
				while (p_predict != NULL)
				{
					kt(i) = this->kernel(p_predict->SV_data, x);
					p_predict = p_predict->next;
					i++;
				}
				//k_t done

				//get prediction
				p_predict = this->SV_begin;
				i = 0;
				while (p_predict != NULL)
				{
					y += p_predict->SV_alpha* kt(i);
					p_predict = p_predict->next;
					i++;
				}
			}
			if (flag != 0) //linear predict
			{
				SV<FeatType, LabelType>* p_predict = this->SV_begin;
				int i = 0;
				while (p_predict != NULL)
				{
					kt[i] = this->kernel(p_predict->SV_data, x);
					p_predict = p_predict->next;
					i++;
				}
				zt = (M_nogd)*kt;
				y = (w_nogd).dot(zt);
			}
			//update
			if (y*x.label<1)
			{
				if (this->size_SV<Budget) //kernel update
				{
					SV<FeatType, LabelType>* support = new SV<FeatType, LabelType>(x.label*this->eta, x);
					this->add_SV(support);

					for (int i = 0; i<this->size_SV - 1; i++)
					{
						(K_budget)(i, this->size_SV - 1) = kt(i);
						(K_budget)(this->size_SV - 1, i) = kt(i);
					}

				}
				else
				{
					if (flag == 0) //SVD
					{
						flag = 1;
						EigenSolver<MatrixXf> es(K_budget);
						MatrixXcf V = es.eigenvectors();
						//cout<<es.eigenvalues()<<endl;

						for (int i = 0; i<k_nogd; i++)
						{
							float length = 0;
							for (int j = 0; j<Budget; j++)
							{
								length = length + V(j, i).real()*V(j, i).real();
							}
							for (int j = 0; j<Budget; j++)
							{
								V(j, i) = V(j, i) / length;
							}
						}

						for (int i = 0; i<k_nogd; i++)
						{
							if (es.eigenvalues()[i].real() <= 0)
							{
								for (int j = 0; j<Budget; j++)
								{
									(M_nogd)(i, j)=0;
								}
							}
							else
							{
								for (int j = 0; j < Budget; j++)
								{
									(M_nogd)(i, j) = V(j, i).real() / sqrt(es.eigenvalues()[i].real());
								}
							}
						}
						zt = (M_nogd)*kt;
						(w_nogd) = (w_nogd) + eta*x.label*zt;
					}
					else
					{
						(w_nogd) = (w_nogd) + eta*x.label*zt;
					}
				}
			}
			return y;
		}
	};
}