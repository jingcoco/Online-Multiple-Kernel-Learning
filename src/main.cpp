/*************************************************************************
	> File Name: main.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions: 
 ************************************************************************/
#include "Params.h"
#include "common/util.h"

#include "data/DataSet.h"
#include "data/libsvmread.h"

#include "loss/LogisticLoss.h"
#include "loss/HingeLoss.h"
#include "loss/SquareLoss.h"
#include "loss/SquaredHingeLoss.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

#include "kernel/kernel_optim.h"
#include "kernel/kernel_DD.h"
#include "kernel/kernel_DDave.h"
#include "kernel/kernel_single.h"
#include "kernel/kernel_spa.h"
#include "kernel/kernel_bogd.h"
#include "kernel/kernel_sd.h"
#include "kernel/kernel_spasd.h"
#include "kernel/kernel_spasdav.h"
#include "kernel/kernel_spass.h"
#include "kernel/kernel_rbp.h"
#include "kernel/kernel_forgetron.h"
#include "kernel/kernel_projectron.h"
#include "kernel/kernel_projectronpp.h"
#include "kernel/kernel_bpas.h"
#include "kernel/kernel_sdpa.h"
#include "kernel/kernel_DDpa.h"

#include "kernel/kernel_rbp_sd.h"
#include "kernel/kernel_bogd_sd.h"
#include "kernel/kernel_forgetron_sd.h"
#include "kernel/kernel_bpas_sd.h"
#include "kernel/kernel_fogd.h"
#include "kernel/kernel_nogd.h"
using namespace std;
using namespace SOL;

#define FeatType double
#define LabelType char

///////////////////////////function declarications/////////////////////
void FakeInput(int &argc, char **args, char** &argv);

template <typename T1, typename T2>
Kernel_optim<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset);
///////////////////
int main(int argc, const char** args) {

   //check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
#endif
	Params param;
	if (param.Parse(argc, args) == false){
		return -1;
	}

	DataSet<FeatType, LabelType> dataset(param.passNum,param.buf_size);
	if (dataset.Load(param.fileName, param.cache_fileName) == false){
		cerr<<"ERROR: Load dataset "<<param.fileName<<" failed!"<<endl;
		return -1;
	}

	Kernel_optim<FeatType, LabelType> *opti = GetOptimizer(param,dataset);
	if (opti == NULL)
		return -1;

	//learning the model
    double time1 = get_current_time();

	double err_rate=opti->Train();

    double time2 = get_current_time();

	printf("\nLearn errRate: %.8f\n", err_rate);
	double time3 = 0;
    printf("Learning time: %.8f s\n", (float)(time2 - time1));

    delete opti;

    return 0;
}


template <typename T1, typename T2>
Kernel_optim<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset) {
	string method = param.str_opt;
	ToUpperCase(method);
	const char* c_str = method.c_str();
	if (strcmp(c_str, "KERNEL-DD") == 0)
		return new kernel_DD<T1, T2>(param,dataset);

	else if (strcmp(c_str, "KERNEL-DDAVE") == 0)
		return new kernel_DDave<T1, T2>(param,dataset);
	else if (strcmp(c_str, "KERNEL-SINGLE") == 0)
		return new kernel_single<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-BOGD") == 0)
		return new kernel_bogd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-PROJECTRON") == 0)
		return new kernel_projectron<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-PROJECTRONPP") == 0)
		return new kernel_projectronpp<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-RBP") == 0)
		return new kernel_rbp<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-BPAS") == 0)
		return new kernel_bpas<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-FORGETRON") == 0)
		return new kernel_forgetron<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-SD") == 0)
		return new kernel_sd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-SPASD") == 0)
		return new kernel_spasd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-SDPA") == 0)
		return new kernel_sdpa<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-DDPA") == 0)
		return new kernel_DDpa<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-RBP_SD") == 0)
		return new kernel_rbp_sd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-BOGD_SD") == 0)
		return new kernel_bogd_sd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-FORGETRON_SD") == 0)
		return new kernel_forgetron_sd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-BPAS_SD") == 0)
		return new kernel_bpas_sd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-FOGD") == 0)
		return new kernel_fogd<T1, T2>(param, dataset);
	else if (strcmp(c_str, "KERNEL-NOGD") == 0)
		return new kernel_nogd<T1, T2>(param, dataset);
	else{
		cerr<<"ERROR: unrecgonized optimization method "<<param.str_opt<<endl;
		return NULL;
	}
}
