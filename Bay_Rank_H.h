#ifndef Bay_Rank_H_H_   /* Include guard */
#define Bay_Rank_H_H_

#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "utils.h"
#include "config.h"
#include "MergeSort.h"

class Bay_Rank_H{
	public:
		vector<double>* lfv_ptr;
	public:
		Bay_Rank_H(vector<double>* log_factorial_vals){
			this->lfv_ptr = log_factorial_vals;
		};
		std::vector<int> sample_H(vector<int> Sample1, vector<int> Pred_Table_2_H, vector<int> Sample2, int _N, double s, vector<int> H, gsl_rng* _rng);
		vector<int> Modify_H(vector<int> H_in,int sid, int did);
		vector<double> gen_sorted_prob_ln(double s, int _N);
		void Rank_Wise_Sample_Smart_Update(vector<int>* Rank_wise_Sample, vector<int>* Sample, vector<int>* rank_map, int sid, int did);
		int gsl_ran_categorical_smart(double* prob_vec,int sid, int did, gsl_rng* _rng);
		int gsl_ran_categorical_(double* prob_vec, int len, gsl_rng* _rng);
		void Modify_H_smart(vector<int>* H_in, vector<int>* rank_map1, int sid, int did);

		int get_inv_count(vector<int> rank2, int N_stateful);
		double get_prior_prob_inv_count(int n, int inv_count);
		double get_prior_prob_deno(int n, int inv_count);
		double lognormpdf( double x, double mu, double sigma );
		double find_beta_count_prior(int inv_count, int n_max);
		double gsl_ran_beta_log_pdf (double inp, double  alpha, double beta);
		double logfactorial(int n);

};

#endif
