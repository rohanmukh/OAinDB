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
