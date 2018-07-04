#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand

#include "Generat_Table.h"


double Generat_Table::gen_s(){
	double temp ;
	while(temp<=s_min || temp >s_max)
		temp = s_mu + gsl_ran_gaussian (_rng, s_theta);
	return temp;
}

void Generat_Table::gen_H(){
	for (int i=0; i<N_MAX; ++i)
		H.at(i) = i;
	random_shuffle ( H.begin(), H.end());
	return;
}

void CorrelatedGenTableFixedSN::gen_H(vector<int> ref_H, int ref_N, vector<int> corr_vec_ranks){ // this will return an H on length N_MAX but it will be ineff for ranks >= _N
	assert(corr_vec_ranks.size() == ref_N);
	assert(ref_H.size() == N_MAX);
	int assert_sum = 0;
	vector<int> rank_used(N_MAX);
	for(int i=0;i<N_MAX;i++){
		rank_used.at(i) = 0;
		this->H.at(i) = -1;
	}
	for(int i=0; i < N_MAX;i++){
		if(ref_H.at(i) >= ref_N){;}
		else{
			this->H.at(i) = corr_vec_ranks.at(ref_H.at(i));
			assert_sum += this->H.at(i);
			rank_used.at(this->H.at(i)) = 1;
		}
	}

	vector<int> rank_unused;
	for(int i=0;i<N_MAX;i++){
		if(rank_used.at(i) == 0){
			rank_unused.push_back(i);
		}
	}
	//random_shuffle ( rank_unused.begin(), rank_unused.end());
	int id = 0;
	for(int i=0;i<N_MAX;i++){
		if(this->H.at(i) == -1){
			this->H.at(i) = rank_unused.at(id++);
			assert_sum += this->H.at(i);
		}
	}

	assert(assert_sum=((N_MAX)*(N_MAX-1)/2)); // since i starts from 0
	return;
}

vector<int> CorrelatedGenTableFixedSN::gen_corr_vector_Kendall_Tau(int N_in){
	int tot_bs = round((1.00 - KenT) * N_in * (N_in-1)/(double)2);
	cout << tot_bs << endl;
	vector<int> rank2(N_in);
	for(int i=0;i<N_in;i++){
		rank2.at(i) = i;
	}
	for(int i=tot_bs;i>0;i){
		int id = rand()%(N_in-1);
		int swap_id = id+1;
		if(rank2.at(swap_id) > rank2.at(id)){
			int temp = rank2.at(id);
			rank2.at(id)=rank2.at(swap_id);
			rank2.at(swap_id) = temp;
			i--;
		}
	}
	assert(rank2.size()==N_in);
	return rank2;
}


int Generat_Table::gen_N_eff( int N_MIN, int N_MAX ){

	int N_eff = (int) gsl_ran_poisson (_rng, lambda);
	N_eff = min(N_eff,N_MAX);
	N_eff = max(N_eff,N_MIN);
	return N_eff;
}

void Generat_Table::gen_data(){
	double sum = 0;
	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double)); // we will have a prob dist over all possible ids but will selectively zero out the ones with H.at(id) >= _N
	int Sample1_count = 0;

	double gp_sum1 = 0;
	for(int i=0;i< _N; i++){ // for gp_sum it is _N
		gp_sum1 += get_zipf(i+1,this->s) ;  // + Noise
	}

	for(int i=0;i<N_MAX;i++){
		if(H.at(i) >= _N){
			prob_multinomial_1[i] = 0;
		}
		else{
			prob_multinomial_1[i] = ceil(get_zipf(H.at(i)+1,this->s)/(double)(gp_sum1) * this->Total_Size);
		}

	}
	//vector<int> init(N_MAX,0);
	//vector<int> Table1_data = prob_multinomial_1[i] //gsl_ran_categorical_batch(prob_multinomial_1,N_MAX,this->Total_Size, init);

	int D_sum = 0;
	for(int i=0;i<N_MAX;i++){
		this->data.at(i) = prob_multinomial_1[i]; //(double) Table1_data.at(i);
		if(this->H.at(i) > this->_N){
			assert(data.at(i)==0);
		}
		D_sum += this->data.at(i);
	}
	//assert(D_sum==this->Total_Size);
	free(prob_multinomial_1);
	return;
}

vector<int> Generat_Table::gsl_ran_categorical_batch(double* prob_vec, int len, int batch_size, vector<int> Sample){ // independent function (except _rng which is OK)
		//assert(prob_vec.size()==len); // Wont be able to check this unless you use a vector
		vector<int> output(len,0);
		unsigned int* catch_multinomial = (unsigned int*)malloc(len*sizeof(unsigned int));
		memset(catch_multinomial,0,len*sizeof(unsigned int));
		gsl_ran_multinomial(_rng,len,batch_size,prob_vec,catch_multinomial);

		for(int i=0;i<len;i++){
			output.at(i) = Sample.at(i) + (int) catch_multinomial[i];
		}

		free(catch_multinomial);
		return output;
	}
