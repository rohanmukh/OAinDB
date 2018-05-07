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


		
void Random_Table::sample_N(vector<int> Sample1, int n_min, int n_max){

	int curr_N = this->_N;
	int val = gsl_ran_binomial(_rng,0.5,1);
	int new_N;
	assert(n_max <= NUM_ID);
	
	if(curr_N == n_max){
		new_N = curr_N - 1;
	}else if(curr_N == n_min){
		new_N = curr_N + 1;
	}else if(val == 0){
		new_N = curr_N - 1;
	}else{
		new_N = curr_N + 1;
	}
	
	// put your assertions here
	double likelihood_ratio = exp(get_ln_prob_sample_N_variable_N(Sample1, this->H, new_N) - get_ln_prob_sample_N_variable_N(Sample1, this->H, curr_N));
	
	double prior_ratio = gsl_ran_poisson_pdf(new_N,lambda)/gsl_ran_poisson_pdf(curr_N,lambda);
	double transition_ratio  = 0.0005;
	
	
	double Jacobian = get_Jacobian_value(curr_N,new_N);
	
	double ratio = likelihood_ratio * prior_ratio * transition_ratio * Jacobian ;

	if(gsl_rng_uniform(_rng) < std::min(ratio,1.0)){
		this->_N = new_N;
	}
		
	
	return;
	
}

double Random_Table::get_Jacobian_value(int curr_N, int new_N){
	assert(abs(curr_N - new_N) == 1);
	double S_sum,p,temp;
	if(new_N == (curr_N+1)){
		S_sum = get_zipf_sum(this->s, curr_N);
		p = 1/pow(curr_N+1,s);
		temp = (pow(S_sum/(S_sum+p),curr_N+2))/S_sum;
	}
	else{
		S_sum = get_zipf_sum(this->s, curr_N);
		p = 1/pow(curr_N,s);
		temp = pow(S_sum/(S_sum-p),curr_N-1)*S_sum;
	}
	
	return temp;
}

double Random_Table::get_zipf_sum(double s_in, double N_in){ // independent
	double gp_sum = 0;
	for(int i = 0; i<N_in;i++){ 
		gp_sum += get_zipf(i+1,s_in) ; //1/(double)(pow(i+1,s_in)); // + Noise
	}
	return gp_sum;
}


void Random_Table::sample_H(vector<int> Sample1, Random_Table Pred_Table_2, vector<int> Sample2){
	
	int _N_xcs_flag = 0;
	int count = 0;
	for(int i=0;i<N_MAX;i++){
		if(Sample1.at(i)>0){
			//assert(this->H.at(i)<this->_N); this will be N_stateful now
			count++;
		}
	}
	
	assert(this->_N >= count); //sample_N should precede sample_H
	
	/*if(this->_N < count){ // Just dont point out error, you can correct them as well
		this->_N = count;
	}*/
		
	int MAX_APP = MAX_APP_NUM; // Max approximation. A value of N means no approximation
	
	vector<int> rank_map1(N_MAX,0); // this is the inverse mapper of H vector
	vector<double> log_prob_temp1 = gen_sorted_prob_ln(this->s);
	
	vector<int> Rank_wise_Sample1(N_MAX,0);
	vector<int> corr_rank_mapper1(N_MAX);
	vector<int> corr_rank_mapper2(N_MAX);
	vector<int> corr_rank_mapper3(N_MAX);
	


		
	int N_stateful = N_MAX;
	//this->_N = N_stateful; // A huge change this is
	//Pred_Table_2._N = N_stateful;
	
	// MATLAB Code ends
	
	for(int i=0;i<N_MAX;i++){
		// Rank wise Sample runs till N_MAX but I think we dont need to access ones over _N, lets see
		// Same goes for rank_map1
		Rank_wise_Sample1.at(this->H.at(i)) = Sample1.at(i);
		rank_map1.at(this->H.at(i)) = i;
		corr_rank_mapper1.at(this->H.at(i)) = Pred_Table_2.H.at(i); 
	}
	
	
	for(int i=0;i<N_MAX;i++){
		corr_rank_mapper2.at(i) = corr_rank_mapper1.at(i);
		corr_rank_mapper3.at(i) = corr_rank_mapper1.at(i);
	}
	
	for(int i = N_stateful;i<N_MAX;i++){
		assert(corr_rank_mapper1.at(i) == i);
		assert(corr_rank_mapper2.at(i) == i);
	}
	
	int old_inv_count = get_inv_count(corr_rank_mapper1, N_stateful ); // Nlog(N)
	
	double* p = (double*)malloc(N_stateful*sizeof(double));
	double* q = (double*)malloc(N_stateful*sizeof(double));
	double* pq = (double*)malloc(N_stateful*sizeof(double));
	
	double old_best1 = get_ln_prob_sample_new(Sample1, this->s, this->H);
	vector<int> inv_count_store(N_stateful);
	for(int sid=0;sid<N_MAX;sid++){ // how many keys ;; the first big O(N) loop
		
		/*memset(p,0,N*sizeof(double));
		memset(q,0,N*sizeof(double));
		memset(pq,0,N*sizeof(double));*/
		
		int rank1 = this->H.at(sid); 

		int inv_count = old_inv_count;
		inv_count_store.at(rank1) = old_inv_count;
		
		
		p[rank1] = old_best1;
	
		
		assert(inv_count>=0);

		q[rank1] = get_prior_prob_inv_count(N_stateful,inv_count); // O(1) 	// shud be N_stateful
		
		pq[rank1] = p[rank1] + q[rank1];
		
		//assert(p[rank1] <= 1000);
		assert(std::isinf(p[rank1])==false);
		assert(std::isnan(p[rank1])==false);
		
		//assert(q[rank1] <= 1000);
		
		assert(std::isinf(q[rank1])==false);
		assert(std::isnan(q[rank1])==false);
		//assert(pq[rank1] <= 1000);
		assert(std::isinf(pq[rank1])==false);
		assert(std::isnan(pq[rank1])==false);
		
		//printf("real inv_count %d ", inv_count);.
		int range_min;
		range_min = max(0,rank1 - MAX_APP);
	
			
		for(int j=rank1-1; j>=range_min;j--){ //place to go
			// j is the rank that table 1's rank1 is trying to swap to
			//cout << "Type 1" << rank1 << " " << j << endl;
			
			p[j] = p[j+1] - Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j+1) - Rank_wise_Sample1.at(j)*log_prob_temp1.at(j) 
			+ Rank_wise_Sample1.at(j)*log_prob_temp1.at(j+1) + Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j);
		
			
			int srid = j+1;
			int drid = j; 
			
			assert(srid < N_stateful);
			assert(drid < N_stateful);
			
			if(  corr_rank_mapper1.at(srid) > corr_rank_mapper1.at(drid)){
				inv_count++;
			}
			else if (corr_rank_mapper1.at(srid) < corr_rank_mapper1.at(drid))
				inv_count--;
			
			inv_count_store.at(j) = inv_count;
			
			int temp = corr_rank_mapper1.at(srid);
			corr_rank_mapper1.at(srid) = corr_rank_mapper1.at(drid);
			corr_rank_mapper1.at(drid) = temp;

			/*int tmp_inv_count = get_inv_count(corr_rank_mapper1, N_stateful); // NlogN
			inv_count_store.at(j) = tmp_inv_count;
			assert(tmp_inv_count == inv_count );
			*/
	
			
			assert(inv_count>=0);
			
			
			q[j] = get_prior_prob_inv_count(N_stateful,inv_count);
			
	
			pq[j] = p[j] + q[j];
			
			//assert(p[j] <= 1000);
			assert(std::isinf(p[j])==false);
			assert(std::isnan(p[j])==false);
			//assert(q[j] <= 1000);
			assert(std::isinf(q[j])==false);
			assert(std::isnan(q[j])==false);
			//assert(pq[j] <= 1000);
			assert(std::isinf(pq[j])==false);
			assert(std::isnan(pq[j])==false);
		}
		
		inv_count = old_inv_count;

		int range_max;
		range_max = min(this->_N-1, rank1 + MAX_APP );
		
		
		
		for(int j=rank1+1; j<=range_max; j++){

			if(_N_xcs_flag == 0)
				p[j] = p[j-1] - Rank_wise_Sample1.at(rank1)* log_prob_temp1.at(j-1) - Rank_wise_Sample1.at(j)*log_prob_temp1.at(j)
			+ Rank_wise_Sample1.at(j)*log_prob_temp1.at(j-1) + Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j);
			else
				p[j] = old_best1;

			
			int srid = j-1; 
			int drid = j; 
			
			assert(srid < N_stateful);
			assert(drid < N_stateful);
			
		
			if ( (corr_rank_mapper2.at(srid) > corr_rank_mapper2.at(drid)))
				inv_count--;
			else 
				inv_count++;
			
			inv_count_store.at(j) = inv_count;
			
			int temp = corr_rank_mapper2.at(srid);
			corr_rank_mapper2.at(srid) = corr_rank_mapper2.at(drid);
			corr_rank_mapper2.at(drid) = temp;
								
			/*int tmp_inv_count = get_inv_count(corr_rank_mapper2, N_stateful);
			inv_count_store.at(j) = tmp_inv_count;
			assert(tmp_inv_count == inv_count);*/
		
			assert(inv_count>=0);
			q[j] = get_prior_prob_inv_count(N_stateful,inv_count); // O(1)
			
		
			
			pq[j] = p[j] + q[j];
			
			
		//	assert(p[j] <= 1000);
			assert(std::isinf(p[j])==false);
			assert(std::isnan(p[j])==false);
			//assert(q[j] <= 1000);
			assert(std::isinf(q[j])==false);
			assert(std::isnan(q[j])==false);
			//assert(pq[j] <= 1000);
			assert(std::isinf(pq[j])==false);
			assert(std::isnan(pq[j])==false);
		}
		
		int rank_sid1 = rank1;
		int rank_did1 = gsl_ran_categorical_smart(pq,range_min,range_max);
		assert(rank_did1 >= 0);
		//int rank_did1 = gsl_ran_categorical_smart(pq,0,N-1);
		
		//assert(rank_sid1 < this->_N);
		//assert(rank_did1 < this->_N);
	
		
		old_best1 = p[rank_did1];
		old_inv_count = inv_count_store.at(rank_did1);
		
		Modify_H_smart(&this->H,&rank_map1,rank_sid1,rank_did1);
		Rank_Wise_Sample_Smart_Update(&Rank_wise_Sample1, &Sample1, &rank_map1, rank_sid1, rank_did1);


		// let us copy the ones that got changed
		
		corr_rank_mapper1.at(rank1) = corr_rank_mapper3.at(rank1);
		corr_rank_mapper2.at(rank1) = corr_rank_mapper3.at(rank1);
		for(int j=rank1-1; j>=range_min;j--){
			corr_rank_mapper1.at(j) = corr_rank_mapper3.at(j);
		}	
		for(int j=rank1+1; j<=range_max; j++){
			corr_rank_mapper2.at(j) = corr_rank_mapper3.at(j);
		}
		
		
		
		if(rank_did1 > rank_sid1){
			
			for(int i = rank_sid1;i<=(rank_did1 - 1);i++){
				int j = i + 1;
				int temp = corr_rank_mapper1.at(j);
				corr_rank_mapper1.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper2.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper3.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper1.at(i) = temp;
				corr_rank_mapper2.at(i) = temp;
				corr_rank_mapper3.at(i) = temp;
			}
			//assert(get_inv_count(corr_rank_mapper1, N_stateful ) == old_inv_count);

			
		}else if(rank_did1 < rank_sid1){
			for(int i = rank_sid1; i>=(rank_did1 + 1);i-- ){
				int j = i - 1;
				int temp = corr_rank_mapper1.at(j);
				corr_rank_mapper1.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper2.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper3.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper1.at(i) = temp;
				corr_rank_mapper2.at(i) = temp;
				corr_rank_mapper3.at(i) = temp;
			}
			//assert(get_inv_count(corr_rank_mapper1, N_stateful ) == old_inv_count);
		}	
	}
	
	free(p);
	free(q);
	free(pq);
	return;
}


			
void Random_Table::sample_H_new(vector<int> Sample1, Random_Table Pred_Table_2, vector<int> Sample2){
	
	int _N_xcs_flag = 0;
	int count = 0;
	for(int i=0;i<N_MAX;i++){
		if(Sample1.at(i)>0){
			//assert(this->H.at(i)<this->_N); this will be N_stateful now
			count++;
		}
	}
	
	assert(this->_N >= count); //sample_N should precede sample_H
	
	/*if(this->_N < count){ // Just dont point out error, you can correct them as well
		this->_N = count;
	}*/
		
	int MAX_APP = MAX_APP_NUM; // Max approximation. A value of N means no approximation
	
	vector<int> rank_map1(N_MAX,0); // this is the inverse mapper of H vector
	vector<double> log_prob_temp1 = gen_sorted_prob_ln(this->s);
	
	vector<int> Rank_wise_Sample1(N_MAX,0);
	vector<int> corr_rank_mapper1(N_MAX);
	vector<int> corr_rank_mapper2(N_MAX);
	vector<int> corr_rank_mapper3(N_MAX);
	


		
	int N_stateful = N_MAX;
	//this->_N = N_stateful; // A huge change this is
	//Pred_Table_2._N = N_stateful;
	
	// MATLAB Code ends
	
	for(int i=0;i<N_MAX;i++){
		// Rank wise Sample runs till N_MAX but I think we dont need to access ones over _N, lets see
		// Same goes for rank_map1
		Rank_wise_Sample1.at(this->H.at(i)) = Sample1.at(i);
		rank_map1.at(this->H.at(i)) = i;
		corr_rank_mapper1.at(this->H.at(i)) = Pred_Table_2.H.at(i); 
	}
	
	
	for(int i=0;i<N_MAX;i++){
		corr_rank_mapper2.at(i) = corr_rank_mapper1.at(i);
		corr_rank_mapper3.at(i) = corr_rank_mapper1.at(i);
	}
	
	for(int i = N_stateful;i<N_MAX;i++){
		assert(corr_rank_mapper1.at(i) == i);
		assert(corr_rank_mapper2.at(i) == i);
	}
	
	int old_inv_count = get_inv_count(corr_rank_mapper1, N_stateful ); // Nlog(N)
	
	double* p = (double*)malloc(N_stateful*sizeof(double));
	double* q = (double*)malloc(N_stateful*sizeof(double));
	double* pq = (double*)malloc(N_stateful*sizeof(double));
	
	double old_best1 = get_ln_prob_sample_new(Sample1, this->s, this->H);
	vector<int> inv_count_store(N_stateful);
	for(int sid=0;sid<N_MAX;sid++){ // how many keys ;; the first big O(N) loop
		
		/*memset(p,0,N*sizeof(double));
		memset(q,0,N*sizeof(double));
		memset(pq,0,N*sizeof(double));*/
		
		int rank1 = this->H.at(sid); 

		int inv_count = old_inv_count;
		inv_count_store.at(rank1) = old_inv_count;
		
		
		p[rank1] = old_best1;
	
		
		assert(inv_count>=0);

		q[rank1] = get_prior_prob_inv_count(N_stateful,inv_count); // O(1) 	// shud be N_stateful
		
		pq[rank1] = p[rank1] + q[rank1];
		
		//assert(p[rank1] <= 1000);
		assert(std::isinf(p[rank1])==false);
		assert(std::isnan(p[rank1])==false);
		
		//assert(q[rank1] <= 1000);
		
		assert(std::isinf(q[rank1])==false);
		assert(std::isnan(q[rank1])==false);
		//assert(pq[rank1] <= 1000);
		assert(std::isinf(pq[rank1])==false);
		assert(std::isnan(pq[rank1])==false);
		
		//printf("real inv_count %d ", inv_count);.
		int range_min;
		range_min = max(0,rank1 - MAX_APP);
	
			
		for(int j=rank1-1; j>=range_min;j--){ //place to go
			// j is the rank that table 1's rank1 is trying to swap to
			//cout << "Type 1" << rank1 << " " << j << endl;
			
			p[j] = p[j+1] - Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j+1) - Rank_wise_Sample1.at(j)*log_prob_temp1.at(j) 
			+ Rank_wise_Sample1.at(j)*log_prob_temp1.at(j+1) + Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j);
		
			
			int srid = j+1;
			int drid = j; 
			
			assert(srid < N_stateful);
			assert(drid < N_stateful);
			
			if(  corr_rank_mapper1.at(srid) > corr_rank_mapper1.at(drid)){
				inv_count++;
			}
			else if (corr_rank_mapper1.at(srid) < corr_rank_mapper1.at(drid))
				inv_count--;
			
			inv_count_store.at(j) = inv_count;
			
			int temp = corr_rank_mapper1.at(srid);
			corr_rank_mapper1.at(srid) = corr_rank_mapper1.at(drid);
			corr_rank_mapper1.at(drid) = temp;

			/*int tmp_inv_count = get_inv_count(corr_rank_mapper1, N_stateful); // NlogN
			inv_count_store.at(j) = tmp_inv_count;
			assert(tmp_inv_count == inv_count );
			*/
	
			
			assert(inv_count>=0);
			
			
			q[j] = get_prior_prob_inv_count(N_stateful,inv_count);
			
	
			pq[j] = p[j] + q[j];
			
			//assert(p[j] <= 1000);
			assert(std::isinf(p[j])==false);
			assert(std::isnan(p[j])==false);
			//assert(q[j] <= 1000);
			assert(std::isinf(q[j])==false);
			assert(std::isnan(q[j])==false);
			//assert(pq[j] <= 1000);
			assert(std::isinf(pq[j])==false);
			assert(std::isnan(pq[j])==false);
		}
		
		inv_count = old_inv_count;

		int range_max;
		range_max = min(this->_N-1, rank1 + MAX_APP );
		
		
		
		for(int j=rank1+1; j<=range_max; j++){

			if(_N_xcs_flag == 0)
				p[j] = p[j-1] - Rank_wise_Sample1.at(rank1)* log_prob_temp1.at(j-1) - Rank_wise_Sample1.at(j)*log_prob_temp1.at(j)
			+ Rank_wise_Sample1.at(j)*log_prob_temp1.at(j-1) + Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j);
			else
				p[j] = old_best1;

			
			int srid = j-1; 
			int drid = j; 
			
			assert(srid < N_stateful);
			assert(drid < N_stateful);
			
		
			if ( (corr_rank_mapper2.at(srid) > corr_rank_mapper2.at(drid)))
				inv_count--;
			else 
				inv_count++;
			
			inv_count_store.at(j) = inv_count;
			
			int temp = corr_rank_mapper2.at(srid);
			corr_rank_mapper2.at(srid) = corr_rank_mapper2.at(drid);
			corr_rank_mapper2.at(drid) = temp;
								
			/*int tmp_inv_count = get_inv_count(corr_rank_mapper2, N_stateful);
			inv_count_store.at(j) = tmp_inv_count;
			assert(tmp_inv_count == inv_count);*/
		
			assert(inv_count>=0);
			q[j] = get_prior_prob_inv_count(N_stateful,inv_count); // O(1)
			
		
			
			pq[j] = p[j] + q[j];
			
			
		//	assert(p[j] <= 1000);
			assert(std::isinf(p[j])==false);
			assert(std::isnan(p[j])==false);
			//assert(q[j] <= 1000);
			assert(std::isinf(q[j])==false);
			assert(std::isnan(q[j])==false);
			//assert(pq[j] <= 1000);
			assert(std::isinf(pq[j])==false);
			assert(std::isnan(pq[j])==false);
		}
		
		int rank_sid1 = rank1;
		int rank_did1 = gsl_ran_categorical_smart(p,range_min,range_max);
		assert(rank_did1 >= 0);
		//int rank_did1 = gsl_ran_categorical_smart(pq,0,N-1);
		
		//assert(rank_sid1 < this->_N);
		//assert(rank_did1 < this->_N);
	
		
		old_best1 = p[rank_did1];
		old_inv_count = inv_count_store.at(rank_did1);
		
		Modify_H_smart(&this->H,&rank_map1,rank_sid1,rank_did1);
		Rank_Wise_Sample_Smart_Update(&Rank_wise_Sample1, &Sample1, &rank_map1, rank_sid1, rank_did1);


		// let us copy the ones that got changed
		
		corr_rank_mapper1.at(rank1) = corr_rank_mapper3.at(rank1);
		corr_rank_mapper2.at(rank1) = corr_rank_mapper3.at(rank1);
		for(int j=rank1-1; j>=range_min;j--){
			corr_rank_mapper1.at(j) = corr_rank_mapper3.at(j);
		}	
		for(int j=rank1+1; j<=range_max; j++){
			corr_rank_mapper2.at(j) = corr_rank_mapper3.at(j);
		}
		
		
		
		if(rank_did1 > rank_sid1){
			
			for(int i = rank_sid1;i<=(rank_did1 - 1);i++){
				int j = i + 1;
				int temp = corr_rank_mapper1.at(j);
				corr_rank_mapper1.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper2.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper3.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper1.at(i) = temp;
				corr_rank_mapper2.at(i) = temp;
				corr_rank_mapper3.at(i) = temp;
			}
			//assert(get_inv_count(corr_rank_mapper1, N_stateful ) == old_inv_count);

			
		}else if(rank_did1 < rank_sid1){
			for(int i = rank_sid1; i>=(rank_did1 + 1);i-- ){
				int j = i - 1;
				int temp = corr_rank_mapper1.at(j);
				corr_rank_mapper1.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper2.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper3.at(j) = corr_rank_mapper1.at(i);
				corr_rank_mapper1.at(i) = temp;
				corr_rank_mapper2.at(i) = temp;
				corr_rank_mapper3.at(i) = temp;
			}
			//assert(get_inv_count(corr_rank_mapper1, N_stateful ) == old_inv_count);
		}	
	}
	
	free(p);
	free(q);
	free(pq);
	return;
}

vector<int> Random_Table::Modify_H(vector<int> H_in,int sid, int did){
	vector<int> Mod_H = H_in;
	
	for(int i=0;i<Mod_H.size();i++){
		if(Mod_H.at(i) == sid)
			Mod_H.at(i) = did;
		else if((did>sid)&&(Mod_H.at(i)>sid)&&(Mod_H.at(i)<=did)){
			Mod_H.at(i)--;
		}
		else if((did<sid)&&(Mod_H.at(i)<sid)&&(Mod_H.at(i)>=did)){
			Mod_H.at(i)++;
		}
	
	}
	return Mod_H;
	
}
	

		
vector<double> Random_Table::gen_sorted_prob_ln(double s){ //  returns a vector of size N_MAX but takes care of _N, _N to N_MAX - 1 is zeroed out
	double sum = -999999999999999.999999999999;

	vector<double> prob_temp(N_MAX,0);
	for(int i=0;i<N_MAX;i++){
		if(i>=this->_N){
			prob_temp.at(i) = -999999999999999.999999999999;
			sum = virtual_log_sum(sum,prob_temp.at(i));
		}
		else{
			double temp = get_zipf_log(i+1,s);//-1* s*log(i+1) ;
			prob_temp.at(i) = temp;
			sum = virtual_log_sum(sum,prob_temp.at(i));
		}
	}
	assert(std::isnan(sum) == 0);
	return prob_temp;
}
		
void Random_Table::Rank_Wise_Sample_Smart_Update(vector<int>* Rank_wise_Sample, vector<int>* Sample, vector<int>* rank_map, int sid, int did){ // independent
//	assert(sid <= _N);
//	assert(did <= _N);
	
	if(did>sid){
		int len = did - sid + 1;int temp;
		for(int i=0;i<len;i++){
			temp = rank_map->at(sid+i);
			Rank_wise_Sample->at(sid+i) = Sample->at(temp);
		}
	}
	else if(did<sid){
		int len = sid - did + 1;int temp;
		for(int i=0;i<len;i++){
			temp = rank_map->at(sid-i);
			Rank_wise_Sample->at(sid-i) = Sample->at(temp);
		}
	}
	return;
}
		
int Random_Table::gsl_ran_categorical_smart(double* prob_vec,int sid, int did){ // independent
	int len = did - sid + 1;
	double* prob_vec_mod = (double*)malloc(len*sizeof(double));
	for(int i=sid;i<(sid+len);i++){
		prob_vec_mod[i-sid] = prob_vec[i];
	}
	int id = gsl_ran_categorical_(prob_vec_mod,len);
	assert(id>=0);
	free(prob_vec_mod);
	int temp = sid + id;
	return temp;	
}
		
int Random_Table::gsl_ran_categorical_(double* prob_vec, int len){ // independent
	
	double sumln = std::numeric_limits<double>::lowest();
	for(int i=0;i<len;i++){
		if(i==0)
			sumln = prob_vec[i];
		else
			sumln = virtual_log_sum(sumln,prob_vec[i]);
	}

	
	for(int i=0;i<len;i++){
		prob_vec[i] = exp(prob_vec[i] - sumln);
		assert(std::isinf(prob_vec[i])==false);
	}
	
	unsigned int* catch_multinomial = (unsigned int*)malloc(len*sizeof(unsigned int));
	memset(catch_multinomial,0,len*sizeof(unsigned int));
	gsl_ran_multinomial(_rng,len,1,prob_vec,catch_multinomial);
	int id = -1;int flag = 0;
	for (int k = 0; k < len; k++) {
		if (catch_multinomial[k]>0){
			id = k;flag = 1;
			break;
		}
	}
	assert(flag==1);
	if(flag==0)
		printf("I am gone here %lf\n",  sumln);
	
	free(catch_multinomial);
	return id;	
}


void Random_Table::Modify_H_smart(vector<int>* H_in, vector<int>* rank_map1, int sid, int did){ // independent
	//assert(sid <= _N);
	//assert(did <= _N);
	
	int rank_map_sid1 = rank_map1->at(sid);
		
	if(did>sid){
		for(int i=sid+1;i<=did;i++){
			rank_map1->at(i-1) = rank_map1->at(i);
			H_in->at(rank_map1->at(i))= i - 1;
		}
	}
	else if(did<sid){
		for(int i=sid-1;i>=did;i--){ // here i is the rank
			rank_map1->at(i+1) = rank_map1->at(i);
			H_in->at(rank_map1->at(i))= i + 1;
		}
	}
	rank_map1->at(did) = rank_map_sid1;
	H_in->at(rank_map_sid1)= did;
	return;
}


void Random_Table::sample_s(vector<int> Sample){
	
	double currentPoint=-1,cpoint=-1;
	double glob_best = -99999999999999.999999999;
	
	assert(lnEVAL(Sample,s_mu)>glob_best);
	for(int nos = 0; nos<50;nos++){
		double temp = s_mu + gsl_ran_gaussian (_rng,theta);  // the zero-magnitude vector is common		
		temp = std::max(temp,S_MIN);
		temp = std::min(temp,S_MAX);
		assert(temp>=S_MIN);
		assert(temp<=S_MAX);
		assert(std::isnan(temp)==false);
		
		if(lnEVAL(Sample,temp)>glob_best){
			currentPoint = temp;
			assert(temp>=S_MIN);
			assert(temp<=S_MAX);
			assert(std::isnan(temp)==false);

			glob_best = lnEVAL(Sample,temp);
		}
	}
	assert(currentPoint>=S_MIN);
	assert(currentPoint<=S_MAX);
	assert(std::isnan(currentPoint)==false);

	cpoint = hill_climb(currentPoint, Sample);
	//assert(glob_best > 0);

	
	
	assert(cpoint>=S_MIN);
	assert(cpoint<=S_MAX);
	assert(std::isnan(cpoint)==false);

	double rect_bound = lnEVAL(Sample,cpoint);
	
	int iter = 0;
	double left_cutoff = cpoint - 0.01;
	
	assert(left_cutoff>=S_MIN);
	assert(left_cutoff<=S_MAX);
	
	assert(std::isnan(left_cutoff)==false);
	
	while(lnEVAL(Sample,left_cutoff) >= (rect_bound - log(1000))){
		left_cutoff -= 0.01;
		if(left_cutoff<=S_MIN){
			left_cutoff = S_MIN - 0.01;
			break;
		}
		iter++;
		if(iter>300)
			printf("Danger1!\n");fflush(stdout);
	}
	left_cutoff += 0.01;
	
	iter = 0;
	double right_cutoff = cpoint + 0.01;
	assert(std::isnan(right_cutoff)==false);
	
	assert(right_cutoff>=S_MIN);
	assert(right_cutoff<=S_MAX);
	
	while(lnEVAL(Sample,right_cutoff) >= (rect_bound - log(1000))){
		right_cutoff += 0.01;
		if(right_cutoff>=S_MAX){
			right_cutoff = S_MAX + 0.01;
			break;
		}
		iter++;
		if(iter>300)
			printf("Danger2!\n");fflush(stdout);
	}
	right_cutoff -= 0.01;

	if(left_cutoff >= right_cutoff){
		this->s = cpoint;
		assert(cpoint == left_cutoff);
		assert(cpoint == right_cutoff);
	}
	else{
		double rand_x;
		double rand_y;
		iter = 0;
		while(1){
			rand_x = left_cutoff + gsl_rng_uniform (_rng) * (right_cutoff - left_cutoff);
			rand_y = log(gsl_rng_uniform(_rng)) + rect_bound;
			
			assert(std::isnan(rand_x)==false);
			assert(rand_x>=S_MIN);
			assert(rand_x<=S_MAX);
	
			if(rand_y < (lnEVAL(Sample, rand_x))){
				this->s = rand_x;
				break;
			}
			iter++;
			if(iter>500){
				this->s = cpoint;
				printf("Danger3!\n");
				cout << " left cutoff :: " << left_cutoff << " right cutoff :: " << right_cutoff << " cpoint :: " << cpoint << "currentPoint :: " << currentPoint << "glob_best " << glob_best << " " << endl;
				fflush(stdout);
				break;
			}
		}
	}
	//printf(" sample is :: %lf\n",rand_x);;
	return;
}


double Random_Table::hill_climb(double currentPoint, vector<int> Sample){
	double bestScore;
	double stepSize = 0.01;
	double candidate[5];
	double epsilon = 0.0001;
	double acceleration = 1.2; // a value such as 1.2 is common
	candidate[0] = -1 * acceleration;
	candidate[1] = -1 / acceleration;
	candidate[2] = 0.00;
	candidate[3] = 1 / acceleration;
	candidate[4] = acceleration;
	int count = 0;
	int iter = 0;
	while(1){
		//cout << " " << candidate[0]<< " " << candidate[1]<< " " << candidate[2]<< " " << candidate[3]<< " " << candidate[4] << endl;
		assert(currentPoint<=S_MAX);
		assert(currentPoint>=S_MIN);
		assert(std::isnan(currentPoint)==false);

		double before = lnEVAL(Sample, currentPoint);
		int best = -1;
		bestScore = -99999999999999999.9999999;
		for(int j=0;j<5;j++){         // try each of 5 candidate locations
			currentPoint = currentPoint + stepSize * candidate[j];
			double temp;
			if((currentPoint<=S_MIN)||(currentPoint>=S_MAX)){
				currentPoint = std::max(currentPoint,S_MIN);
				currentPoint = std::min(currentPoint,S_MAX);
				
				temp = 0;
			}else{
				assert(std::isnan(currentPoint)==false);

				temp =  lnEVAL(Sample, currentPoint);
	
				/*if(std::isnan(temp) || temp < 0)
					temp = 0;
				*/
				
				//assert(temp >= 0);
			}
			currentPoint = currentPoint - stepSize * candidate[j];
			if(temp >= bestScore){
				 bestScore = temp;
				 best = j;
				// cout << temp << endl;
					 
				 assert(best>=0);
			}
		}
		if (candidate[best] == 0.00){
			stepSize = stepSize / acceleration;
		}
		else{
			assert(best>=0);
			currentPoint = currentPoint + stepSize * candidate[best];
			stepSize = stepSize * candidate[best]; // accelerate
			
		}
		stepSize = std::max(stepSize,0.5);
		stepSize = std::min(stepSize,0.001);
		
		assert(std::isinf(stepSize)==false);
		assert(std::isnan(stepSize)==false);
		
		currentPoint = std::max(currentPoint,S_MIN);
		currentPoint = std::min(currentPoint,S_MAX);
		assert(best>=0);
		
		
		assert(std::isnan(currentPoint)==false);
		assert(currentPoint<=S_MAX);
		assert(currentPoint>=S_MIN);
		if (abs(lnEVAL(Sample,currentPoint) - before)/before <= epsilon){
			count++;
		}else{
			count = 0;
		}
		if(iter>500){
			cout << bestScore << endl;fflush(stdout);
			break;
		}
		iter++;
		if(count == 10)
			break;
			
	}
	
	assert(currentPoint>=S_MIN);
	assert(currentPoint<=S_MAX);
	
	return currentPoint;
}


double Random_Table::lnEVAL(vector<int> Sample, double s_in){
	vector<int> temp_H(N_MAX);
	for(int i=0;i>N_MAX;i++){
		temp_H.at(i) = this->H.at(i);
	}
	assert(std::isnan(s_in)==false);

	assert(s_in<=S_MAX);
	assert(s_in>=S_MIN);
	double temp = get_ln_prob_sample_new(Sample, s_in, temp_H)  + log(gsl_ran_gaussian_pdf (s_in - s_mu, theta));
	/*if(std::isnan(temp)){
		temp = std::numeric_limits<double>::lowest();
	}*/
	//assert(temp >= 0);
	return temp;
}
	
double Random_Table::EVAL(vector<int> Sample, double s_in){
	vector<int> temp_H(N_MAX);
	for(int i=0;i>N_MAX;i++){
		temp_H.at(i) = this->H.at(i);
	}
	assert(std::isnan(s_in)==false);

	assert(s_in<=S_MAX);
	assert(s_in>=S_MIN);
	double temp = get_prob_sample_new(Sample, s_in, temp_H)  * gsl_ran_gaussian_pdf (s_in - s_mu, theta);
	/*if(std::isnan(temp)){
		temp = std::numeric_limits<double>::lowest();
	}*/
	assert(temp >= 0);
	return temp;
}
		
double Random_Table::get_prob_sample_new(vector<int> Sample1, double s_in, vector<int> H){
	//vector<double> data_temp = gen_sorted_data(s_in);
	
	int flag = 0;
	double sum = 0;
	
	assert(s_in<=S_MAX);
	assert(s_in>=S_MIN);
	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double));
	
	double gp_sum1 = 0;
	for(int i=0;i< this->_N;i++){
		gp_sum1 += get_zipf(i+1,s_in) ; //1/(double)(pow(i+1,s_in)); // + Noise
	}
	assert(std::isinf(gp_sum1) == false);
	
	unsigned int* sample_count = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	for(int i=0;i<N_MAX;i++){
		if(H.at(i) < this->_N){
			//prob_multinomial_1[i] = 1/(double)(pow(H.at(i)+1,s_in)*gp_sum1);
			prob_multinomial_1[i] = get_zipf(H.at(i),s_in);
			prob_multinomial_1[i] /= gp_sum1;
			assert(prob_multinomial_1[i] > 0.00);
			sample_count[i] = (unsigned int) Sample1.at(i);

		}
		else{
			prob_multinomial_1[i] = 0.0;
			sample_count[i] = 0; // lets force this

		}
		
		/*if(H.at(i) >= this->_N && sample_count[i] >= 1)
			flag = 1;*/
	}
	
	//double result = gsl_ran_multinomial_pdf (N_MAX, prob_multinomial_1, sample_count);
	vector<double> prob_new;
	vector<unsigned int> sample_new;	
	if(flag==0){
		for(int i=0;i<N_MAX;i++){
			if(H.at(i) < this->_N){
				assert(prob_multinomial_1[i] > 0.0);
				prob_new.push_back(prob_multinomial_1[i]);
				sample_new.push_back(sample_count[i]);
			}
		}
		assert(sample_new.size() == prob_new.size());
	}
	double result;
	if(flag == 0)
		result = gsl_ran_multinomial_pdf (sample_new.size(), prob_new.data(), sample_new.data());
	else 
		result = 0;
	
	/*if(std::isnan(result)){
		result = std::numeric_limits<double>::lowest();
	}*/
	
	assert(std::isnan(result)==0);
	assert(result>=0);
	free(sample_count);
	free(prob_multinomial_1);
	return result;
}

double Random_Table::get_ln_prob_sample_N_variable_N(vector<int> Sample1, vector<int> H, int _N_in){
	//vector<double> data_temp = gen_sorted_data(s);
	
	int flag = 0;
	double sum = 0;
	
	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double));
	
	double gp_sum1 = 0;
	for(int i=0;i<_N_in;i++){
		gp_sum1 += get_zipf(i+1,this->s) ; //1/(double)(pow(i+1,this->s)); // + Noise
	}
	//std::numeric_limits<double>::lowest();
	unsigned int* sample_count = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	for(int i=0;i<N_MAX;i++){
		if(H.at(i) < _N_in){
			prob_multinomial_1[i] = get_zipf(H.at(i)+1,this->s);
			assert(prob_multinomial_1[i] > 0.0);
			sample_count[i] = (unsigned int) Sample1.at(i);
		}else{
			prob_multinomial_1[i] = 0.0;
			sample_count[i] = (unsigned int) Sample1.at(i);
		}
		
		
		if(sample_count[i] > 0 && prob_multinomial_1[i] == 0.0 )
			flag = 1;
	}
	
	double ans;
	if(flag == 1)
		ans = -9999999.99999;
	else
		ans = gsl_ran_multinomial_lnpdf (N_MAX, prob_multinomial_1, sample_count);
		
	free(sample_count);
	free(prob_multinomial_1);
	return ans;
}


double Random_Table::get_ln_prob_sample_new(vector<int> Sample1, double s, vector<int> H){
	//vector<double> data_temp = gen_sorted_data(s);
	
	int flag = 0;
	double sum = 0;
	
	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double));
	
	double gp_sum1 = 0;
	for(int i=0;i<this->_N;i++){
		gp_sum1 += get_zipf(i+1,s);//1/(double)(pow(i+1,s)); // + Noise
	}
	double ln_fact_sum=0;
	unsigned int* sample_count = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	for(int i=0;i<N_MAX;i++){
		if(H.at(i) < _N){
			prob_multinomial_1[i] = get_zipf(H.at(i)+1,s)/gp_sum1;//1/(double)(pow(H.at(i)+1,s)*gp_sum1);
			sample_count[i] = (unsigned int) Sample1.at(i);
			ln_fact_sum += sample_count[i] * get_zipf_log(H.at(i)+1,s);//(-1)*s*log(H.at(i)+1);
		}
		else{
			prob_multinomial_1[i] = 0;
			sample_count[i] = 0;
		}
	}

	assert(std::isnan(ln_fact_sum)==false);
	free(sample_count);
	free(prob_multinomial_1);
	return ln_fact_sum;
}


int Random_Table::get_inv_count(vector<int> rank2, int N_stateful){ // it only takes into account _N elements
	
	assert(rank2.size()==N_MAX);
	
	vector<int> arr_temp;
	for(int i=0;i<N_stateful;i++){
		arr_temp.push_back(rank2.at(i));
	}

	int array_size = arr_temp.size();
	int arr[array_size];
	for(int i=0;i<array_size;i++){
		arr[i] = arr_temp.at(i);
	}
	int inv_count = mergeSort(arr,array_size);
	return inv_count;
}

double Random_Table::get_prior_prob_inv_count(int n, int inv_count){ // n = N , independent
	

	double mu = (n)*(n-1)/(double)4;
	double sigma = sqrt((n)*(n-1)*(2*n+5)/(double)72);
	
	double prob_val_deno_part1 = lognormpdf(inv_count,mu,sigma);
	double prob_val_deno_part2 = 0;//logfactorial(n);
	double prob_val_deno = prob_val_deno_part1 + prob_val_deno_part2;
	assert(std::isnan(prob_val_deno)==false);
	//prob_val_deno = std::max(prob_val_deno , 0.0);
	
	assert(inv_count>=0);
	//cout << prob_val_deno << endl;
	double prob_val_nume =  find_beta_count_prior(inv_count, n); // O(1) , check second element if it is n
	assert(std::isnan(prob_val_nume)==false);
	return  (prob_val_nume - prob_val_deno);
}

double Random_Table::get_prior_prob_deno(int n, int inv_count){ 
	double mu = (n)*(n-1)/(double)4;
	double sigma = sqrt((n)*(n-1)*(2*n+5)/(double)72);
	
	double prob_val_deno_part1 = lognormpdf(inv_count,mu,sigma);
	double prob_val_deno_part2 = 0;//logfactorial(n);
	double prob_val_deno = prob_val_deno_part1 + prob_val_deno_part2;
	return prob_val_deno;
}

double Random_Table::lognormpdf( double x, double mu, double sigma ){
	double pi = 22/7;
	double ratio = (x-mu)/sigma;
	double op = -1 * log(2*pi)/2 - log(sigma) - pow(ratio,2)/2;
	return op;
}


double Random_Table::find_beta_count_prior(int inv_count, int n_max){ // independent
	
	double MAX = (n_max*(n_max-1))/(double)2;
	double map_to_beta = 1-(inv_count/(double) MAX);
	
	if(inv_count > MAX){
		//cout << inv_count << " " << MAX << endl;
		printf("Oh my god1!! \n");
	}
	assert(inv_count >= 0);
	double log_prob_val = gsl_ran_beta_log_pdf (map_to_beta, corr_alpha, corr_beta);
	return log_prob_val;
}

double Random_Table::gsl_ran_beta_log_pdf (double inp, double  alpha, double beta){
	double term1,term2;
	if(beta == 1.0){
		term2 = 0;
	}else{
		term2 = (beta - 1) * log(1 - inp);
	}
	
	if((alpha == 1.0)){
		term1 = 0;
	}else{
		term1 = (alpha - 1)*log(inp);
	}
	//assert(inp>0);
	//assert(inp<1);
	double term = term1 + term2;
	return term;
}

double Random_Table::logfactorial(int n) {
	assert(n>0);
	double prod = log_factorial_vals.at(n-1);
	return prod;
}

int Random_Table::mergeSort(int arr[], int array_size){
	int *temp = (int *)malloc(sizeof(int)*array_size);
	int ans = _mergeSort(arr, temp, 0, array_size - 1);
	free(temp);
	return ans;
}


int Random_Table::_mergeSort(int arr[], int temp[], int left, int right){		/* An auxiliary recursive function that sorts the input array and returns the number of inversion in the array. */
  int mid, inv_count = 0;
  if (right > left){
	/* Divide the array into two parts and call _mergeSortAndCountInv()
	   for each of the parts */
	mid = (right + left)/2;
 
	/* Inversion count will be sum of inversions in left-part, right-part
	  and number of inversions in merging */
	inv_count  = _mergeSort(arr, temp, left, mid);
	inv_count += _mergeSort(arr, temp, mid+1, right);
 
	/*Merge the two parts*/
	inv_count += merge(arr, temp, left, mid+1, right);
  }
  return inv_count;
}
 

int Random_Table::merge(int arr[], int temp[], int left, int mid, int right){    /* This funt merges two sorted arrays and returns inversion count in
   the arrays.*/
  int i, j, k;
  int inv_count = 0;
 
  i = left; /* i is index for left subarray*/
  j = mid;  /* j is index for right subarray*/
  k = left; /* k is index for resultant merged subarray*/
  while ((i <= mid - 1) && (j <= right))
  {
	if (arr[i] <= arr[j])
	{
	  temp[k++] = arr[i++];
	}
	else
	{
	  temp[k++] = arr[j++];
 
	 /*this is tricky -- see above explanation/diagram for merge()*/
	  inv_count = inv_count + (mid - i);
	}
  }
 
  /* Copy the remaining elements of left subarray
   (if there are any) to temp*/
  while (i <= mid - 1)
	temp[k++] = arr[i++];
 
  /* Copy the remaining elements of right subarray
   (if there are any) to temp*/
  while (j <= right)
	temp[k++] = arr[j++];
 
  /*Copy back the merged elements to original array*/
  for (i=left; i <= right; i++)
	arr[i] = temp[i];
 
  return inv_count;
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


