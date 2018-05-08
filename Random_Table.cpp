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
	
	double old_best1 = get_ln_prob_sample_new(Sample1, this->s, this->H, this->_N);
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
	
	double old_best1 = get_ln_prob_sample_new(Sample1, this->s, this->H, this->_N);
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


