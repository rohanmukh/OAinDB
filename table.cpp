#include <iostream>
#include <vector>		
#include <cmath>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <cstdlib>      // std::rand, std::srand
#include <cstring>

#include "table.h"

table::table(int Total_Size, int N_MAX){
		this->Total_Size = Total_Size;
		this->N_MAX = N_MAX;
		//******************************//	
		gsl_env_setup();
		data.resize(N_MAX);
		H.resize(N_MAX);
	}

void table::gsl_env_setup(){
		const gsl_rng_type * T_stat;
		unsigned long int seed = rand() % 42922295;
		//cout << seed << endl; fflush(stdout);
		//fflush(stdout);
		gsl_rng_env_setup();
		T_stat = gsl_rng_default;
		_rng = gsl_rng_alloc(T_stat);
		_rng_get_samp = gsl_rng_alloc(T_stat);
		gsl_rng_set(_rng,seed);
	}
	
	
long double table::Join(table othr_tbl){
	long double sum = 0;
	for(int i=0;i<N_MAX;i++){
		long double temp1 = (long double) data.at(i);
		long double temp2 = (long double) othr_tbl.data.at(i);
		
		sum += (long double) temp1*temp2;
	}
	
	return sum;
}

vector<int> table::get_samples(int Sample_Size,int seeder){
			
	seeder = rand()%1133333;
	gsl_rng_set(_rng_get_samp,seeder);
	
	vector<int> samples(N_MAX,0);
	double* data_prob = (double*)malloc(N_MAX*sizeof(double));
	double sum = 0;
	for(int i=0;i<N_MAX;i++) {
		
		data_prob[i] = (double) (this->data.at(i));
		sum += data_prob[i];
	}
	
	for(int i=0;i<N_MAX;i++) {
		data_prob[i] /= sum;
	}
	unsigned int* catch_multinomial = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	memset(catch_multinomial,0,N_MAX*sizeof(unsigned int));
	gsl_ran_multinomial(_rng_get_samp,N_MAX,Sample_Size,data_prob,catch_multinomial);
	
	for(int i=0;i<N_MAX;i++) {
		samples.at(i) = catch_multinomial[i];
	}
	free(catch_multinomial);
	
	

	/*int iter = 0;
	while(iter++<Sample_Size){
		unsigned int* catch_multinomial = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
		memset(catch_multinomial,0,N_MAX*sizeof(unsigned int));
		gsl_ran_multinomial(_rng_get_samp,N_MAX,1,data_prob,catch_multinomial);
		for (int k = 0; k < N_MAX; k++) {
			if (catch_multinomial[k]>0){
				
				samples.at(k) = samples.at(k) + 1;
				if((data_prob[k] < 1.00)){
					printf("Error Spotted in here\n");
				}
				data_prob[k] -= 1.0;
				sum -= 1;
				break;
			}
		}
		free(catch_multinomial);
	}*/
	
	free(data_prob);
	
	
	return samples;
}