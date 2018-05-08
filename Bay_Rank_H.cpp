#include <iostream>
#include <vector>		
#include <cmath>
#include <cassert>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_sf_gamma.h" //lnchoose
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand
#include <string.h>      // std::rand, std::srand

#include "Bay_Rank_H.h"


void Bay_Rank_H::sample_H(vector<int> Sample1, vector<int> Pred_Table_2_H, vector<int> Sample2, int _N, double s, vector<int> H, gsl_rng* _rng){
	
	int _N_xcs_flag = 0;
	int count = 0;
	for(int i=0;i<N_MAX;i++){
		if(Sample1.at(i)>0){
			//assert(H.at(i)<_N); it will be N_stateful now
			count++;
		}
	}
	
	assert(_N >= count); //sample_N should precede sample_H
	
	/*if(_N < count){ // Just dont point out error, you can correct them as well
		_N = count;
	}*/
		
	int MAX_APP = MAX_APP_NUM; // Max approximation. A value of N means no approximation
	
	vector<int> rank_map1(N_MAX,0); // it is the inverse mapper of H vector
	vector<double> log_prob_temp1 = gen_sorted_prob_ln(s, _N);
	
	vector<int> Rank_wise_Sample1(N_MAX,0);
	vector<int> corr_rank_mapper1(N_MAX);
	vector<int> corr_rank_mapper2(N_MAX);
	vector<int> corr_rank_mapper3(N_MAX);
	


		
	int N_stateful = N_MAX;
	//_N = N_stateful; // A huge change it is
	//Pred_Table_2._N = N_stateful;
	
	for(int i=0;i<N_MAX;i++){
		// Rank wise Sample runs till N_MAX but I think we dont need to access ones over _N, lets see
		// Same goes for rank_map1
		Rank_wise_Sample1.at(H.at(i)) = Sample1.at(i);
		rank_map1.at(H.at(i)) = i;
		corr_rank_mapper1.at(H.at(i)) = Pred_Table_2_H.at(i); 
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
	
	double old_best1 = get_ln_prob_sample_new(Sample1, s, H, _N);
	vector<int> inv_count_store(N_stateful);
	for(int sid=0;sid<N_MAX;sid++){ // how many keys ;; the first big O(N) loop
		
		/*memset(p,0,N*sizeof(double));
		memset(q,0,N*sizeof(double));
		memset(pq,0,N*sizeof(double));*/
		
		int rank1 = H.at(sid); 

		int inv_count = old_inv_count;
		inv_count_store.at(rank1) = old_inv_count;
		
		
		p[rank1] = old_best1;
	
		
		assert(inv_count>=0);

		q[rank1] = get_prior_prob_inv_count(N_stateful,inv_count); // O(1) 	// shud be N_stateful
		pq[rank1] = p[rank1] + q[rank1];
		
		//assert(p[rank1] <= 1000);//assert(q[rank1] <= 1000);//assert(pq[rank1] <= 1000);
		assert(std::isinf(p[rank1])==false);assert(std::isnan(p[rank1])==false);
		assert(std::isinf(q[rank1])==false);assert(std::isnan(q[rank1])==false);
		assert(std::isinf(pq[rank1])==false);assert(std::isnan(pq[rank1])==false);
		
		int range_min = max(0,rank1 - MAX_APP);
	
			
		for(int j=rank1-1; j>=range_min;j--){ //place to go
			// j is the rank that table 1's rank1 is trying to swap to
			//cout << "Type 1" << rank1 << " " << j << endl;
			
			p[j] = p[j+1] - Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j+1) - Rank_wise_Sample1.at(j)*log_prob_temp1.at(j) 
			+ Rank_wise_Sample1.at(j)*log_prob_temp1.at(j+1) + Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j);
		
			
			int srid = j+1;
			int drid = j; 
			
			assert(srid < N_stateful);assert(drid < N_stateful);
			
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
			
			//assert(p[j] <= 1000);//assert(q[j] <= 1000);//assert(pq[j] <= 1000);
			assert(std::isinf(p[j])==false);assert(std::isnan(p[j])==false);
			assert(std::isinf(q[j])==false);assert(std::isnan(q[j])==false);
			assert(std::isinf(pq[j])==false);assert(std::isnan(pq[j])==false);
		}
		
		inv_count = old_inv_count;

		int range_max;
		range_max = min(_N-1, rank1 + MAX_APP );
		
		
		
		for(int j=rank1+1; j<=range_max; j++){

			if(_N_xcs_flag == 0)
				p[j] = p[j-1] - Rank_wise_Sample1.at(rank1)* log_prob_temp1.at(j-1) - Rank_wise_Sample1.at(j)*log_prob_temp1.at(j)
			+ Rank_wise_Sample1.at(j)*log_prob_temp1.at(j-1) + Rank_wise_Sample1.at(rank1)*log_prob_temp1.at(j);
			else
				p[j] = old_best1;

			
			int srid = j-1; 
			int drid = j; 
			
			assert(srid < N_stateful);	assert(drid < N_stateful);
			
		
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
			
			
			//assert(p[j] <= 1000);//assert(q[j] <= 1000);
			assert(std::isinf(p[j])==false);assert(std::isnan(p[j])==false);
			assert(std::isinf(q[j])==false);assert(std::isnan(q[j])==false);
			//assert(pq[j] <= 1000);assert(std::isinf(pq[j])==false);
			assert(std::isnan(pq[j])==false);
		}
		
		int rank_sid1 = rank1;
		int rank_did1 = gsl_ran_categorical_smart(pq,range_min,range_max,_rng); // can try only p if using an uniform prior
		assert(rank_did1 >= 0);
		//int rank_did1 = gsl_ran_categorical_smart(pq,0,N-1);//assert(rank_sid1 < _N);//assert(rank_did1 < _N);

		old_best1 = p[rank_did1];
		old_inv_count = inv_count_store.at(rank_did1);
		
		Modify_H_smart(&H,&rank_map1,rank_sid1,rank_did1);
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


vector<int> Bay_Rank_H::Modify_H(vector<int> H_in,int sid, int did){
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
		
vector<double> Bay_Rank_H::gen_sorted_prob_ln(double s, int _N){ //  returns a vector of size N_MAX but takes care of _N, _N to N_MAX - 1 is zeroed out
	double sum;
	vector<double> prob_temp(N_MAX,0);
	for(int i=0;i<N_MAX;i++){
		if(i>=_N){
			prob_temp.at(i) = std::numeric_limits<double>::lowest();
		}
		else if (i==0){
			prob_temp.at(i) = get_zipf_log(i+1,s);//-1* s*log(i+1) ;
			sum = prob_temp.at(i);
		}
		else {
			prob_temp.at(i) = get_zipf_log(i+1,s);//-1* s*log(i+1) ;
			sum = virtual_log_sum(sum,prob_temp.at(i));
		}
	}
	assert(std::isnan(sum) == 0);
	return prob_temp;
}

void Bay_Rank_H::Rank_Wise_Sample_Smart_Update(vector<int>* Rank_wise_Sample, vector<int>* Sample, vector<int>* rank_map, int sid, int did){ // independent
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

		
int Bay_Rank_H::gsl_ran_categorical_smart(double* prob_vec,int sid, int did, gsl_rng* _rng){ // independent
	int len = did - sid + 1;
	double* prob_vec_mod = (double*)malloc(len*sizeof(double));
	for(int i=sid;i<(sid+len);i++){
		prob_vec_mod[i-sid] = prob_vec[i];
	}
	int id = gsl_ran_categorical_(prob_vec_mod,len, _rng);
	assert(id>=0);
	free(prob_vec_mod);
	int temp = sid + id;
	return temp;	
}
		
int Bay_Rank_H::gsl_ran_categorical_(double* prob_vec, int len, gsl_rng* _rng){ // independent
	
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


void Bay_Rank_H::Modify_H_smart(vector<int>* H_in, vector<int>* rank_map1, int sid, int did){
	// independent
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


int Bay_Rank_H::get_inv_count(vector<int> rank2, int N_stateful){ // it only takes into account _N elements
	
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

double Bay_Rank_H::get_prior_prob_inv_count(int n, int inv_count){ // n = N , independent
	

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

double Bay_Rank_H::get_prior_prob_deno(int n, int inv_count){ 
	double mu = (n)*(n-1)/(double)4;
	double sigma = sqrt((n)*(n-1)*(2*n+5)/(double)72);
	
	double prob_val_deno_part1 = lognormpdf(inv_count,mu,sigma);
	double prob_val_deno_part2 = 0;//logfactorial(n);
	double prob_val_deno = prob_val_deno_part1 + prob_val_deno_part2;
	return prob_val_deno;
}

double Bay_Rank_H::lognormpdf( double x, double mu, double sigma ){
	double pi = 22/7;
	double ratio = (x-mu)/sigma;
	double op = -1 * log(2*pi)/2 - log(sigma) - pow(ratio,2)/2;
	return op;
}


double Bay_Rank_H::find_beta_count_prior(int inv_count, int n_max){ // independent
	
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

double Bay_Rank_H::gsl_ran_beta_log_pdf (double inp, double  alpha, double beta){
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

double Bay_Rank_H::logfactorial(int n) {
	assert(n>0);
	double prod = this->lfv_ptr->at(n-1);
	return prod;
}
