#ifndef GENERAT_TABLE_H_   /* Include guard */
#define GENERAT_TABLE_H_

#include "table.h"
#include "config.h"
#include "utils.h" //get_zipf

class Generat_Table : public table{
	public:
		double s; // The Zipfian Parameter
		int _N;   // _N is also called N_eff
		// Hyperparameters for s
		double theta = S_THETA;double s_mu = S_MU;
		// For N_eff/_N
		double lambda = LAMBDA;
		
	public:
		Generat_Table(int Total_Size, int N_MAX, int N_MIN) : table(Total_Size,N_MAX){
			this->s = gen_s();
			this->_N = gen_N_eff(N_MIN, N_MAX); // returns a value of _N
			gen_H(); // Remember, gen_H() and gen_H(.. , ..) will generate a H which is N_MAX long but effectively those id's with H.at(id)>_N wont be counted towards data 
			gen_data();	
		}

		double gen_s();
		void gen_H();
		int gen_N_eff( int N_MIN, int N_MAX );
		vector<int> gsl_ran_categorical_batch(double* prob_vec, int len, int batch_size, vector<int> Sample);
		void gen_data();
};

	
class Generat_TableFixedSN : public Generat_Table{
	public:
		Generat_TableFixedSN(int Total_Size, int N_MAX, int N_MIN, int _N, double s) : Generat_Table(Total_Size, N_MAX, N_MIN){
			this->s = s;
			this->_N = _N;
			gen_data();
		}
};

class CorrelatedGenTableFixedSN : public Generat_Table{
	public:
		CorrelatedGenTableFixedSN(int Total_Size, int N_MAX, int N_MIN, int _N, double s, Generat_Table* ref_table):Generat_Table(Total_Size, N_MAX, N_MIN) {
			this->s = s;
			this->_N = _N;
			vector<int> corr_vec_ranks = gen_corr_vector_Kendall_Tau(ref_table->_N);
			gen_H(ref_table->H, ref_table->_N , corr_vec_ranks);
			gen_data();
		}
		public:
			void gen_H(vector<int> ref_H, int ref_N, vector<int> corr_vec_ranks);
			vector<int> gen_corr_vector_Kendall_Tau(int N_in);
};

#endif
