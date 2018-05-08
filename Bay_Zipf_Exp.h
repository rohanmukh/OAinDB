#ifndef BAY_ZIPF_EXP_H_   /* Include guard */
#define BAY_ZIPF_EXP_H_

#include <iostream>
#include <vector>		
#include <cmath>
#include <cassert>
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand

#include "utils.h"

class Bay_Zipf_Exp{
	public:
		int N_MAX;
		int _N;
		

	public:
	
		Bay_Zipf_Exp(int N_MAX, int _N){
			this->N_MAX = N_MAX;
			this->_N = _N;
		}
		double sample_s(vector<int> Sample, vector<int> H_in, gsl_rng* _rng);
		double hill_climb(double currentPoint, vector<int> Sample, vector<int> H_in);
		double lnEVAL(vector<int> Sample, double s_in, vector<int> H_in);
		double EVAL(vector<int> Sample, double s_in, vector<int> H_in);
};

#endif