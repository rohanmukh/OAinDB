#ifndef RANDOM_TABLE_H_   /* Include guard */
#define RANDOM_TABLE_H_

#include "Generat_Table.h"
#include "utils.h" 
#include "Bay_Zipf_Exp.h"
#include "Bay_RJMC_N.h"
#include "Bay_Rank_H.h"


class Random_Table : public Generat_Table{
	public:
		vector<double> log_factorial_vals;
		Bay_Zipf_Exp* S_sampler;
		Bay_RJMC_N* N_sampler;
		Bay_Rank_H* H_sampler;

	public:	
		Random_Table(int Total_Size, int N_MAX, int N_MIN, vector<double> log_factorial_vals) :  Generat_Table(Total_Size, N_MAX, N_MIN){
			this->log_factorial_vals = log_factorial_vals;
			this->S_sampler = new Bay_Zipf_Exp();
			this->N_sampler = new Bay_RJMC_N();
			this->H_sampler = new Bay_Rank_H(&log_factorial_vals);
		};
		
		void sample_N(vector<int> Sample);
		void sample_s(vector<int> Sample);
		void sample_H(vector<int> Sample1, vector<int> Pred_Table_2_H, vector<int> Sample2);
				
		long double Bay_Join(Random_Table othr_tbl, vector<int> Sample1, vector<int> Sample2);
};

#endif
