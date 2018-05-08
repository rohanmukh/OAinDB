#include <iostream>
#include <vector>		
#include <cmath>
#include <cassert>
#include <climits>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand
#include <map>
#include <cstring>
#include "utils.h"
#include "config.h"

using namespace std;



int cmp(int a, int b) {
	return a>b;
}


long double get_SE(vector<long double> Freq_val, vector<long double> Orig_Join, int NUM_RUNS){
	long double sq_sum2 = 0;
	for(int n = 0; n<NUM_RUNS; n++){
			sq_sum2 += pow(Freq_val.at(n) - Orig_Join.at(n),2);
	}
	long double sigma2 = sqrt(sq_sum2/NUM_RUNS);
	return sigma2;
}
		
vector<int> get_ranks(vector<int> input1, vector<int> input2){
	
	vector<int> input(input1.size());
	for(int i=0;i<input.size();i++){
		input.at(i) = input1.at(i) ;
	}
	map<int, vector<int> > pos;
	for(int i=0; i<input.size(); i++) {
		if(pos.find(input[i]) == pos.end()) { // if first occurence of the element
			vector<int> temp;
			temp.push_back(i);
			pos[input[i]] = temp;
		}
		else {
			vector<int> temp = pos[input[i]];
			temp.push_back(i);
			pos[input[i]] = temp;
		}
	}
	
	for (std::map<int, vector<int>>::iterator it=pos.begin(); it!=pos.end(); ++it)
		random_shuffle ( pos[it->first].begin(), pos[it->first].end());
	
	sort(input.begin(), input.end(),cmp);
 
	vector<int> rank_vec(input.size());
	for(int i=0; i<input.size(); i++) {
		vector<int> temp = pos[input[i]];
		for(int j=0; j<temp.size(); j++) {
			rank_vec[temp[j]] = i;
			i++;
		}
		i--;
	}
	return rank_vec;
}



vector<double> eval_logfactorial(int N_maximum) {
			
	vector<double> temp(N_maximum); // remember i-th vlue is stored in i-1 th index
	double prod = 0;
	for (int i=0;i<N_maximum;i++){
		prod += log(i+1);
		temp.at(i) = prod;
	}
	return temp;
}

long double get_Freq_Join(vector<int> Sample_1  , vector<int> Sample_2, int sample_size){
	assert(Sample_1.size() == NUM_ID); // NUM_ID is N_MAX
	assert(Sample_2.size() == NUM_ID); // NUM_ID is N_MAX
	
	long double scale_up = TOTAL_SIZE_1/(long double)sample_size;//exp(log(TOTAL_SIZE_1) + log(TOTAL_SIZE_2) - log(sample_size) - log(sample_size));
	scale_up *= TOTAL_SIZE_2/(long double)sample_size;
	long double Freq_Join = 0;//((double) inner_product(Sample_1.begin(),Sample_1.end(), Sample_2.begin(), 0)) * scale_up ;
	for(int i=0;i<NUM_ID;i++){
		assert(Sample_1.at(i)>0);
		assert(Sample_2.at(i)>0);
		long double temp1 = (long double) Sample_1.at(i);
		long double temp2 = (long double) Sample_2.at(i);
		long double Freq_Join_old = Freq_Join;
		Freq_Join += temp1*temp2;
		assert(Freq_Join >= Freq_Join_old);
	}
	//cout << Freq_Join << " " << scale_up << endl;
	Freq_Join *= scale_up;
	return Freq_Join;
}

double virtual_log_sum(double x, double y){
	/*we have x = log(a) and y = log(b) and we want to find z = log(a+b)
	%% a = exp(x) and b = exp(y) 	%% z = log(exp(x) + exp(y)) %% z = log(exp(x) * (1+exp(y-x)))   (x>y)  	%% z = x + log(1+exp(y-x))*/
	if(std::isinf(x) && x<0)
		return y;
	if(x==std::numeric_limits<double>::lowest())
		return y;
	if(std::isinf(y) && y<0)
		return x;
	if(y==std::numeric_limits<double>::lowest())
		return x;
	if(y>x){
	   double temp = x;
	   x = y;
	   y = temp;
	}
	return x + log(1+exp(y-x));
}

double get_zipf(int in,double s){
	assert(in>=0);
	double out = 1/(double)(pow(in,s));
	return out;
}

double get_zipf_log(int in,double s){
	//double out = -1*s*log(in);
	double out = log(get_zipf(in,s));
	return out;
}

double get_zipf_sum(double s_in, double N_in){ // independent
	double gp_sum = 0;
	for(int i = 0; i<N_in;i++){ 
		gp_sum += get_zipf(i+1,s_in) ; //1/(double)(pow(i+1,s_in)); // + Noise
	}
	return gp_sum;
}

double get_prob_sample_new(vector<int> Sample1, double s_in, vector<int> H, int _N){
	//vector<double> data_temp = gen_sorted_data(s_in);
	
	int flag = 0;
	double sum = 0;
	
	assert(s_in<=s_max);
	assert(s_in>=s_min);
	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double));
	
	double gp_sum1 = 0;
	for(int i=0;i< _N;i++){
		gp_sum1 += get_zipf(i+1,s_in) ; //1/(double)(pow(i+1,s_in)); // + Noise
	}
	assert(std::isinf(gp_sum1) == false);
	
	unsigned int* sample_count = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	for(int i=0;i<N_MAX;i++){
		if(H.at(i) < _N){
			//prob_multinomial_1[i] = 1/(double)(pow(H.at(i)+1,s_in)*gp_sum1);
			prob_multinomial_1[i] = get_zipf(H.at(i),s_in);
			prob_multinomial_1[i] /= gp_sum1;
			assert(prob_multinomial_1[i] > 0.00);
			sample_count[i] = (unsigned int) Sample1.at(i);

		}
		else{
			prob_multinomial_1[i] = 0.0;
			sample_count[i] = 0; // lets force this

		}
		
		/*if(H.at(i) >= this->_N && sample_count[i] >= 1)
			flag = 1;*/
	}
	
	//double result = gsl_ran_multinomial_pdf (N_MAX, prob_multinomial_1, sample_count);
	vector<double> prob_new;
	vector<unsigned int> sample_new;	
	if(flag==0){
		for(int i=0;i<N_MAX;i++){
			if(H.at(i) < _N){
				assert(prob_multinomial_1[i] > 0.0);
				prob_new.push_back(prob_multinomial_1[i]);
				sample_new.push_back(sample_count[i]);
			}
		}
		assert(sample_new.size() == prob_new.size());
	}
	double result;
	if(flag == 0)
		result = gsl_ran_multinomial_pdf (sample_new.size(), prob_new.data(), sample_new.data());
	else 
		result = 0;
	
	/*if(std::isnan(result)){
		result = std::numeric_limits<double>::lowest();
	}*/
	
	assert(std::isnan(result)==0);
	assert(result>=0);
	free(sample_count);
	free(prob_multinomial_1);
	return result;
}



double get_ln_prob_sample_new(vector<int> Sample1, double s, vector<int> H, int _N){
	//vector<double> data_temp = gen_sorted_data(s);
	
	int flag = 0;
	double sum = 0;
	
	double* prob_multinomial_1 = (double*)malloc(N_MAX*sizeof(double));
	
	double gp_sum1 = 0;
	for(int i=0;i< _N;i++){
		gp_sum1 += get_zipf(i+1,s);//1/(double)(pow(i+1,s)); // + Noise
	}
	double ln_fact_sum=0;
	unsigned int* sample_count = (unsigned int*)malloc(N_MAX*sizeof(unsigned int));
	for(int i=0;i<N_MAX;i++){
		if(H.at(i) < _N){
			prob_multinomial_1[i] = get_zipf(H.at(i)+1,s)/gp_sum1;//1/(double)(pow(H.at(i)+1,s)*gp_sum1);
			sample_count[i] = (unsigned int) Sample1.at(i);
			ln_fact_sum += sample_count[i] * get_zipf_log(H.at(i)+1,s);//(-1)*s*log(H.at(i)+1);
		}
		else{
			prob_multinomial_1[i] = 0;
			sample_count[i] = 0;
		}
	}

	assert(std::isnan(ln_fact_sum)==false);
	free(sample_count);
	free(prob_multinomial_1);
	return ln_fact_sum;
}
