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
#include "Correlated_Table.h"
	
	
void Correlated_Table::sample_corr(vector<int> Sample2, vector<int>  Pred_Table_1_H, vector<int> Sample1){
	
	sample_H(Sample2,Pred_Table_1_H, Sample1);
	compute_rank2(Pred_Table_1_H);
	return;
}
void Correlated_Table::sample_corr_new(vector<int> Sample2, vector<int>  Pred_Table_1_H, vector<int> Sample1){
	
	sample_H(Sample2,Pred_Table_1_H, Sample1);
	compute_rank2(Pred_Table_1_H);
	return;
}

void Correlated_Table::compute_rank2(vector<int> ref_H){
	for(int i=0;i<N_MAX;i++){
		this->rank2.at(ref_H.at(i)) = this->H.at(i);
	}
}