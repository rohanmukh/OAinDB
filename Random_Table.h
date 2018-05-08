#ifndef RANDOM_TABLE_H_   /* Include guard */
#define RANDOM_TABLE_H_

#include "Generat_Table.h"
#include "utils.h" 
#include "Bay_Zipf_Exp.h"


class Random_Table : public Generat_Table{
	public:
		vector<double> log_factorial_vals;
		Bay_Zipf_Exp* S_sampler;

	public:	
		Random_Table(int Total_Size, int N_MAX, int N_MIN, vector<double> log_factorial_vals) :  Generat_Table(Total_Size, N_MAX, N_MIN){
			this->log_factorial_vals = log_factorial_vals;
			this->S_sampler = new Bay_Zipf_Exp(N_MAX, this->_N);
		};
		
		void sample_N(vector<int> Sample1, int n_min, int n_max);
		double get_Jacobian_value(int curr_N, int new_N);	
		double get_zipf_sum(double s_in, double N_in);	
		void sample_H(vector<int> Sample1, Random_Table Pred_Table_2, vector<int> Sample2);
		void sample_H_new(vector<int> Sample1, Random_Table Pred_Table_2, vector<int> Sample2);
		vector<int> Modify_H(vector<int> H_in,int sid, int did);
		vector<double> gen_sorted_prob_ln(double s);
		void Rank_Wise_Sample_Smart_Update(vector<int>* Rank_wise_Sample, vector<int>* Sample, vector<int>* rank_map, int sid, int did);
		int gsl_ran_categorical_smart(double* prob_vec,int sid, int did);
		int gsl_ran_categorical_(double* prob_vec, int len);
		void Modify_H_smart(vector<int>* H_in, vector<int>* rank_map1, int sid, int did);
		void sample_s(vector<int> Sample);
		
		double get_ln_prob_sample_N_variable_N(vector<int> Sample1, vector<int> H, int _N_in);

		
		int get_inv_count(vector<int> rank2, int N_stateful);
		double get_prior_prob_inv_count(int n, int inv_count);
		double get_prior_prob_deno(int n, int inv_count);
		double lognormpdf( double x, double mu, double sigma );
		double find_beta_count_prior(int inv_count, int n_max);
		double gsl_ran_beta_log_pdf(double inp, double  alpha, double beta) ;
		double logfactorial(int n) ;
		
		
		int mergeSort(int arr[], int array_size);
		int _mergeSort(int arr[], int temp[], int left, int right);
		int merge(int arr[], int temp[], int left, int mid, int right);
		long double Bay_Join(Random_Table othr_tbl, vector<int> Sample1, vector<int> Sample2);
};

#endif
