#ifndef Bay_RJMC_N_H_   /* Include guard */
#define Bay_RJMC_N_H_

#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "utils.h"
#include "config.h" 

class Bay_RJMC_N{
	public:
	
	public:
		Bay_RJMC_N(){
		};
		
		int sample_N(vector<int> Sample1, int N_in, double s_in, vector<int> H_in, gsl_rng* _rng);
		double get_Jacobian_value(int curr_N, int new_N, double s_in);
		double get_ln_prob_sample_N_variable_N(vector<int> Sample1, int _N_in, double s_in, vector<int> H_in);
};

#endif