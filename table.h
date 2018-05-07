#ifndef TABLE_H_   /* Include guard */
#define TABLE_H_

#include <iostream>
#include <vector>		
#include <cmath>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <cstdlib>      // std::rand, std::srand
#include <cstring>

#include "config.h"

using namespace std;

class table
{
	public:
		int N_MAX; // max number of data buckets or keys
		int Total_Size; // number of traces in the table
			
		vector<double> data; // All the data
		vector<int> H; // All the hidden rank variables
		
		gsl_rng* _rng;
		gsl_rng* _rng_get_samp;
		
	public:	
		table(int Total_Size, int N_MAX);
		void gsl_env_setup();
		long double Join(table othr_tbl);
		vector<int> get_samples(int Sample_Size,int seeder);
};


#endif // TABLE_H_
