#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_sf_gamma.h" //lnchoose
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand

#include "Random_Table.h"



void Random_Table::sample_s(vector<int> Sample){
	this->s = this->S_sampler->sample_s(Sample, this->H, this->_N, _rng);
	return;
}

void Random_Table::sample_N(vector<int> Sample){
	this->_N = this->N_sampler->sample_N(Sample, this->_N, this->s, this->H, _rng);
	return;
}

void Random_Table::sample_H(vector<int> Sample1, vector<int> Pred_Table_2_H, vector<int> Sample2){
	this->H = this->H_sampler->sample_H(Sample1, Pred_Table_2_H, Sample2, this->_N, this->s, this->H, _rng);
	return;
}

long double Random_Table::Bay_Join(Random_Table othr_tbl, vector<int> Sample1, vector<int> Sample2){

	assert(Sample1.size() == N_MAX);
	assert(Sample2.size() == N_MAX);



	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double));
	double* prob_multinomial_2 = (double*)malloc(N_MAX*sizeof(double));



	//double gp_sum1 = 0,gp_sum2 = 0;

	double ln_gp_sum1 = std::numeric_limits<double>::lowest();
	double ln_gp_sum2 = std::numeric_limits<double>::lowest();

	for(int i=0;i<N_MAX;i++){
		if(i==0){
			ln_gp_sum1 = 0;
			ln_gp_sum2 = 0;
			continue;
		}

		if(i < this->_N)
			ln_gp_sum1 = virtual_log_sum(ln_gp_sum1, get_zipf_log(i+1,this->s));
		if(i < othr_tbl._N)
			ln_gp_sum2 = virtual_log_sum(ln_gp_sum2, get_zipf_log(i+1,othr_tbl.s));
	}


	int Sample1_count = 0;
	int Sample2_count = 0;
	for(int i=0;i<N_MAX;i++){
		if(this->H.at(i)<this->_N){
			prob_multinomial_1[i] =  exp(get_zipf_log(this->H.at(i)+1,this->s) - ln_gp_sum1);
			assert(prob_multinomial_1[i]>0.00);
		}
		else
			prob_multinomial_1[i] = 0.0;

		if(othr_tbl.H.at(i) < othr_tbl._N)
			prob_multinomial_2[i] =  exp(get_zipf_log(othr_tbl.H.at(i)+1,othr_tbl.s)  - ln_gp_sum2);
		else
			prob_multinomial_2[i] = 0.0;

		Sample1_count += Sample1.at(i);
		Sample2_count += Sample2.at(i);

	}
	int rem_table_1 = this->Total_Size - Sample1_count;
	int rem_table_2 = othr_tbl.Total_Size - Sample2_count;
	vector<int> Table1_data = gsl_ran_categorical_batch(prob_multinomial_1,N_MAX,rem_table_1, Sample1);
	vector<int> Table2_data = gsl_ran_categorical_batch(prob_multinomial_2,N_MAX,rem_table_2, Sample2);

	int D_sum_1 = 0;
	int D_sum_2 = 0;
	long double sum = 0;
	for(int i=0;i<N_MAX;i++){
		D_sum_1 += Table1_data.at(i);
		D_sum_2 += Table2_data.at(i);

		long double temp1 = (long double) Table1_data.at(i);
		long double temp2 = (long double) Table2_data.at(i);

		sum += (long double) temp1*temp2;
	}
	assert(D_sum_1 == this->Total_Size);
	assert(D_sum_2 == othr_tbl.Total_Size);

	free(prob_multinomial_1);
	free(prob_multinomial_2);
	return (sum);
}
