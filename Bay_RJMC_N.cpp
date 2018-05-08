#include <iostream>
#include <vector>		
#include <cmath>
#include <cassert>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_sf_gamma.h" //lnchoose
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand

#include "Bay_RJMC_N.h"


int Bay_RJMC_N::sample_N(vector<int> Sample1, int _N_in, double s_in, vector<int> H_in, gsl_rng* _rng){

	const int curr_N = _N_in;
	int val = gsl_ran_binomial(_rng,0.5,1);
	int new_N;
	
	if(curr_N == N_MAX){
		new_N = curr_N - 1;
	}else if(curr_N == N_MIN){
		new_N = curr_N + 1;
	}else if(val == 0){
		new_N = curr_N - 1;
	}else{
		new_N = curr_N + 1;
	}
	
	// put your assertions here
	double likelihood_ratio = exp(get_ln_prob_sample_N_variable_N(Sample1, new_N, s_in, H_in) - get_ln_prob_sample_N_variable_N(Sample1, curr_N, s_in, H_in));
	double prior_ratio = gsl_ran_poisson_pdf(new_N, lambda)/gsl_ran_poisson_pdf(curr_N, lambda);
	double transition_ratio  = 0.0005;
	double Jacobian = get_Jacobian_value(curr_N,new_N,s_in);
	
	double ratio = likelihood_ratio * prior_ratio * transition_ratio * Jacobian ;

	if(gsl_rng_uniform(_rng) < std::min(ratio,1.0))
		return new_N;
	else
		return curr_N;
}


double Bay_RJMC_N::get_Jacobian_value(int curr_N, int new_N, double s_in){
	assert(abs(curr_N - new_N) == 1);
	double S_sum,p,temp;
	if(new_N == (curr_N+1)){
		S_sum = get_zipf_sum(s_in, curr_N);
		p = 1/pow(curr_N+1,s_in);
		temp = (pow(S_sum/(S_sum+p),curr_N+2))/S_sum;
	}
	else{
		S_sum = get_zipf_sum(s_in, curr_N);
		p = 1/pow(curr_N,s_in);
		temp = pow(S_sum/(S_sum-p),curr_N-1)*S_sum;
	}
	
	return temp;
}


double Bay_RJMC_N::get_ln_prob_sample_N_variable_N(vector<int> Sample1, int _N_in, double s_in, vector<int> H_in){
	//vector<double> data_temp = gen_sorted_data(s_in);
	
	int flag = 0;
	
	double* prob_multinomial = (double*)malloc(N_MAX*sizeof(double));
	double gp_sum1 = get_zipf_sum(s_in, _N_in);

	unsigned int* sample_count = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	for(int i=0;i<N_MAX;i++){
		sample_count[i] = (unsigned int) Sample1.at(i);
		if(H_in.at(i) < _N_in)
			prob_multinomial[i] = get_zipf(H_in.at(i)+1,s_in) / gp_sum1;
		else
			prob_multinomial[i] = 0.0;
		
		if(sample_count[i] > 0 && prob_multinomial[i] == 0.0 )
			flag = 1;
	}
	
	double ans;
	if(flag == 1)
		ans = -9999999.99999;
	else
		ans = gsl_ran_multinomial_lnpdf (N_MAX, prob_multinomial, sample_count);
		
	free(sample_count);
	free(prob_multinomial);
	return ans;
}

