#ifndef GENERAT_TABLE_H_   /* Include guard */
#define GENERAT_TABLE_H_

#include "table.h"
#include "config.h"
#include "utils.h" //get_zipf

class Generat_Table : public table{

	public:
	
		double s; // The Zipfian Parameter
		int _N;   // _N is also called N_eff
		
		// Hyperparameters
		// For s
		double theta = S_THETA;
		double s_mu = S_MU;
		
		// For N_eff/_N
		double lambda = LAMBDA;
		

	public:
		Generat_Table(int Total_Size, int N_MAX, int N_MIN, int _N = -1, double s=-1, Generat_Table* ref_table = NULL );
		double gen_s();
		void gen_H();
		void gen_H(vector<int> ref_H, int ref_N, vector<int> corr_vec_ranks);
		vector<int> gen_corr_vector_Kendall_Tau(int N_in);
		int gen_N_eff( int N_MIN, int N_MAX );
		vector<int> gsl_ran_categorical_batch(double* prob_vec, int len, int batch_size, vector<int> Sample);
		void gen_data();
};

#endif
