#include <iostream>
#include <vector>		
#include <cmath>
#include <cassert>
#include <climits>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand
#include <map>
#include <cstring>

using namespace std;

#define TOTAL_SIZE_1 5000000000
#define TOTAL_SIZE_2 5000000000
#define NUM_ID 10

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
	int _N_MAX = NUM_ID/5;
	double out = 1/(double)(pow(std::min(in,in),s));
	return out;
}

double get_zipf_log(int in,double s){
	//double out = -1*s*log(in);
	double out = log(get_zipf(in,s));
	return out;
}