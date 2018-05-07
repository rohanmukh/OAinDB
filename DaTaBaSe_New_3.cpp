/*
	Bayesian Approximate Query Processing
	@author: Rohan Mukherjee, rm38@rice.edu, Rice University
	@advisor: Dr. Christopher M. Jermaine
*/

#include <iostream>
#include <string>
#include <vector>		
#include <cmath>
#include <cassert>
#include <cstdlib>      // std::rand, std::srand
#include <cstring>
#include <omp.h>
#include <chrono>

using namespace std;
#include "config.h"
#include "utils.h"
#include "table.h"
#include "Generat_Table.h"
#include "Random_Table.h"
#include "Correlated_Table.h"

// 10,000 /1 when corr = 1
// 1/10,000 when corr = 0
// 100 samples test

vector<double> log_factorial_vals;

int main(){
	
	log_factorial_vals = eval_logfactorial(NUM_ID);
	
	std::srand ( unsigned ( std::time(0) ) );
	
	int No_of_sample_sizes = 7;
	int sample_size_arr[7] = {500000,5000000,10000000,20000000,50000000,100000000,500000000};	
	
	clock_t begin = clock();
	std::chrono::time_point<std::chrono::system_clock> start,end;
	start=std::chrono::system_clock::now();
	
	
	
	int n_max = NUM_ID, n_min = 60 * n_max/100;
	
	int INP_DIM = 100*NUM_ID/100;
	Generat_Table _Table1( TOTAL_SIZE_1, n_max, n_min, INP_DIM , S_MU );
	Generat_Table _Table2( TOTAL_SIZE_2, n_max, n_min, INP_DIM, S_MU, &_Table1 );
	vector<long double> Orig_Join(NUM_RUNS);
	for(int iter=0;iter<NUM_RUNS;iter++){
		Orig_Join.at(iter) = _Table1.Join(_Table2);
	}
	
	printf("No of keys : %d,  Correlation :: %lf, S_MU :: %lf , Prior for correlation %lf / %lf ", INP_DIM, KenT, S_MU, corr_alpha,corr_beta );	
	printf("Original Join :: %Le \n\n\n",  Orig_Join.at(0));
			
	for(int ij=0;ij<No_of_sample_sizes;ij++){
			
		int sample_size = sample_size_arr[ij];
	
		vector<long double> Freq_val(NUM_RUNS);
		vector<long double> Bay_val_cmp(NUM_RUNS);
		
	
		
		omp_set_num_threads(NUM_THREADS);
		#pragma omp parallel for
		for(int iter =0;iter<NUM_RUNS;iter++){	
			vector<int> Sample_1 = _Table1.get_samples(sample_size,ij);
		
			for(int i=0;i<NUM_ID;i++){
				if(Sample_1.at(i) > 0)
					assert(_Table1.H.at(i) < _Table1._N);
			}
			
			
			vector<int> Sample_2 = _Table2.get_samples(sample_size,ij);
			for(int i=0;i<NUM_ID;i++){
				if(Sample_2.at(i) > 0)
					assert(_Table2.H.at(i) < _Table2._N);
			}
			long double Freq_Join = get_Freq_Join(Sample_1,Sample_2, sample_size);
			//************************//
			int NUM_MAX = NUM_ID;
			int N_MIN=0; 
			for(int i=0;i<Sample_1.size();i++){
				if(Sample_1.at(i)>0 || Sample_2.at(i) > 0)
					N_MIN++;
				//cout << Sample_1.at(i) << " " << Sample_2.at(i) << endl ;
			}
			N_MIN = std::max(N_MIN , n_min);


			long double Bay_Join_runng_avg_cmp = 0;
			
			Random_Table Pred_Table_1_cmp(TOTAL_SIZE_1,NUM_MAX, N_MIN, log_factorial_vals);
			Pred_Table_1_cmp.H = get_ranks(Sample_1,Sample_2);
			Correlated_Table Pred_Table_2_cmp(TOTAL_SIZE_2,NUM_MAX, N_MIN, log_factorial_vals) ;//, &Pred_Table_1_cmp);
			Pred_Table_2_cmp.H = get_ranks(Sample_2,Sample_1);
					
			int Gibbs_iter = 0;
			
			double avg_temp_rank=0;
			int sampling_iter_num = 0;
			while(Gibbs_iter<NUM_GIBBS_ITER){
				
				Pred_Table_1_cmp.sample_s(Sample_1);
				Pred_Table_2_cmp.sample_s(Sample_2);
							
				Pred_Table_1_cmp._N = std::max(_Table1._N , N_MIN);
				Pred_Table_2_cmp._N = std::max(_Table2._N , N_MIN);
								
				Pred_Table_1_cmp.sample_H(Sample_1,Pred_Table_2_cmp, Sample_2);
				Pred_Table_2_cmp.sample_corr(Sample_2, Pred_Table_1_cmp, Sample_1);
				
				int curr_N = Pred_Table_2_cmp._N;
				double corr = Pred_Table_2_cmp.get_inv_count(Pred_Table_2_cmp.rank2, curr_N )/(double)((curr_N)*(curr_N-1)/2);
				//cout << 1-corr << endl;
			
				if(Gibbs_iter > BURN_IN && Gibbs_iter %10 >= 0){
					sampling_iter_num++;
					long double frac1 = (sampling_iter_num - 1) / (long double)(sampling_iter_num);
					long double temp2 = Pred_Table_1_cmp.Bay_Join(Pred_Table_2_cmp ,Sample_1, Sample_2)/(long double)(sampling_iter_num);
					Bay_Join_runng_avg_cmp = Bay_Join_runng_avg_cmp * frac1  + temp2;
					//cout << Gibbs_iter << " " << Orig_Join.at(0) << " " << Pred_Table_1_cmp.Bay_Join(Pred_Table_2_cmp ,Sample_1, Sample_2) << endl;
				}
				
				Gibbs_iter++;
			}
					
			Bay_val_cmp.at(iter) = Bay_Join_runng_avg_cmp;	
			//cout << Bay_Join_runng_avg_cmp << " " << Freq_val.at(iter) << endl;
		}
		
		cout << " Sample Size:: " << sample_size << " ";

		int num_runs = NUM_RUNS;long double sigma;
		sigma = get_SE(Bay_val_cmp,Orig_Join,num_runs);
		printf("Bayesian Comp SE :: %Le ", sigma);
		
		sigma = get_SE(Freq_val, Orig_Join, num_runs);
		printf("Frequentist SE :: %Le \n", sigma);

	}
	clock_t endp = clock();
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;	

	double elapsed_secs = double(endp - begin) / CLOCKS_PER_SEC;
	double elapsed_mins = elapsed_secs/60;
	//printf("The elapsed time is :: %lf mins \n", elapsed_mins);
	cout << "elapsed time: " << elapsed_seconds.count()/(double) 60 << "m\n";
	return 0;	
};

