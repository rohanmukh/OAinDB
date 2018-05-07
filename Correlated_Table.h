#ifndef CORRELATED_TABLE_H_   /* Include guard */
#define CORRELATED_TABLE_H_

#include "Random_Table.h"
#include "config.h"
#include "utils.h" 


class Correlated_Table : public Random_Table{
	public:
	vector<int> rank2;
	public:	
	
	/*Correlated_Table(int Total_Size, int N_MAX, int N_MIN, Random_Table* ref_table) : Random_Table(Total_Size, N_MAX, N_MIN, ref_table) {
		rank2.resize(N_MAX);
		compute_rank2(ref_table->H);
	};*/
		
	Correlated_Table(int Total_Size, int N_MAX, int N_MIN, vector<double> log_factorial_vals) : Random_Table(Total_Size, N_MAX, N_MIN, log_factorial_vals) {
		rank2.resize(N_MAX);
		for(int i=0;i<N_MAX;i++){
			this->rank2.at(i) = i;
		}
	};
	
	void sample_corr(vector<int> Sample2, Random_Table Pred_Table_1, vector<int> Sample1);
	void sample_corr_new(vector<int> Sample2, Random_Table Pred_Table_1, vector<int> Sample1);
	void compute_rank2(vector<int> ref_H);
};

#endif