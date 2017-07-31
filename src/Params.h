/*************************************************************************
> File Name: Params.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:51:05 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/

#ifndef HEADER_PARSER_PARAM
#define HEADER_PARSER_PARAM

#include "common/ezOptionParser.hpp"

#include "data/parser.h"

#include <string>
#include <map>


using std::string;
using std::map;

//using namespace ez;

namespace SOL
{
	class Params
	{
	private:
		ez::ezOptionParser opt;
		ez::ezOptionValidator* vfloat; 
		ez::ezOptionValidator* vint; 
		ez::ezOptionValidator* vbool; 

		map<std::string, float*> flag2storage_float;
		map<std::string, int*> flag2storage_int;
		map<std::string, bool*> flag2storage_bool;
		map<std::string, std::string*> flag2storage_str;

		typedef map<std::string, float*>::iterator map_float_iter;
		typedef map<std::string, int*>::iterator map_int_iter;
		typedef map<std::string, bool*>::iterator map_bool_iter;
		typedef map<std::string, std::string*>::iterator map_str_iter;

	public:
		//input data
		string fileName; //source file name
		string cache_fileName; //cached file name
		string test_fileName; //test file name
		string test_cache_fileName; //cached test file name

		//dataset type
		string str_data_type;
		//loss function type
		string str_loss;
		//optimization method
		string str_opt;

		//optimzation parameters
		float eta; //learning rate
		float gamma;
		float lambda;
		float alpha;
		float beta;
		float weight_start;
		int single_classifier;
		int budget_size;
		float delta_stoch;
		int k_nogd;

		int buf_size; //number of chunks in dataset 
		int passNum;
		bool is_learn_best_param; //whether learn best parameter

		bool is_normalize;

	public:
		Params();
		~Params();

		bool Parse(int argc, const char** args);
		void Help();

	private:
		void Init();

		void add_option(float default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, float *storage);
		void add_option(int default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, int *storage);
		void add_option(bool default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, bool *storage);
		void add_option(const char* default_val, bool is_required, int expectArgs, 
			const char* descr, const char* flag, string *storage);
	};
}
#endif
