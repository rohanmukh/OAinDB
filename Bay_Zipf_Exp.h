#ifndef BAY_ZIPF_EXP_H_   /* Include guard */
#define BAY_ZIPF_EXP_H_

#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include <iostream>
#include <vector>		

#include "utils.h"
#include "config.h"

class Bay_Zipf_Exp{
	public:
	

	public:
	
		Bay_Zipf_Exp(){
			
		}
		double sample_s(vector<int> Sample, vector<int> H_in, int _N, gsl_rng* _rng);
		double hill_climb(double currentPoint, vector<int> Sample, vector<int> H_in, int _N);
		double lnEVAL(vector<int> Sample, double s_in, vector<int> H_in, int _N);
		double EVAL(vector<int> Sample, double s_in, vector<int> H_in, int _N);
};

#endif