#include <iostream>
#include <vector>		
#include <cmath>
#include <cassert>
#include "gsl/gsl_rng.h"	  // rng , rng_uniform
#include "gsl/gsl_sf_gamma.h" //lnchoose
#include "gsl/gsl_randist.h"  // ran_multinomial
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand, std::srand

#include "Bay_Zipf_Exp.h"
#include "utils.h"
#include "config.h"

double Bay_Zipf_Exp::sample_s(vector<int> Sample, vector<int> H_in, int _N, gsl_rng* _rng){
	double s_out;
	double currentPoint=-1,cpoint=-1;
	double glob_best = -99999999999999.999999999;
	
	assert(lnEVAL(Sample,s_mu,H_in,_N)>glob_best);
	for(int nos = 0; nos<50;nos++){
		double temp = s_mu + gsl_ran_gaussian (_rng,s_theta);  // the zero-magnitude vector is common		
		temp = std::max(temp,s_min);
		temp = std::min(temp,s_max);
		assert(temp>=s_min);
		assert(temp<=s_max);
		assert(std::isnan(temp)==false);
		
		if(lnEVAL(Sample,temp,H_in, _N)>glob_best){
			currentPoint = temp;
			assert(temp>=s_min);
			assert(temp<=s_max);
			assert(std::isnan(temp)==false);

			glob_best = lnEVAL(Sample,temp,H_in, _N);
		}
	}
	assert(currentPoint>=s_min);
	assert(currentPoint<=s_max);
	assert(std::isnan(currentPoint)==false);

	cpoint = hill_climb(currentPoint, Sample, H_in, _N);
	//assert(glob_best > 0);

	
	assert(cpoint>=s_min);
	assert(cpoint<=s_max);
	assert(std::isnan(cpoint)==false);

	double rect_bound = lnEVAL(Sample,cpoint,H_in,_N);
	
	int iter = 0;
	double left_cutoff = cpoint - 0.01;
	
	assert(left_cutoff>=s_min);
	assert(left_cutoff<=s_max);
	
	assert(std::isnan(left_cutoff) == false);
	
	while(lnEVAL(Sample,left_cutoff,H_in,_N) >= (rect_bound - log(1000))){
		left_cutoff -= 0.01;
		if(left_cutoff<=s_min){
			left_cutoff = s_min - 0.01;
			break;
		}
		iter++;
		if(iter>300)
			printf("Danger1!\n");fflush(stdout);
	}
	left_cutoff += 0.01;
	
	iter = 0;
	double right_cutoff = cpoint + 0.01;
	assert(std::isnan(right_cutoff)==false);
	
	assert(right_cutoff>=s_min);
	assert(right_cutoff<=s_max);
	
	while(lnEVAL(Sample,right_cutoff,H_in,_N) >= (rect_bound - log(1000))){
		right_cutoff += 0.01;
		if(right_cutoff>=s_max){
			right_cutoff = s_max + 0.01;
			break;
		}
		iter++;
		if(iter>300)
			printf("Danger2!\n");fflush(stdout);
	}
	right_cutoff -= 0.01;

	if(left_cutoff >= right_cutoff){
		s_out = cpoint;
		assert(cpoint == left_cutoff);
		assert(cpoint == right_cutoff);
	}
	else{
		double rand_x;
		double rand_y;
		iter = 0;
		while(1){
			rand_x = left_cutoff + gsl_rng_uniform (_rng) * (right_cutoff - left_cutoff);
			rand_y = log(gsl_rng_uniform(_rng)) + rect_bound;
			
			assert(std::isnan(rand_x)==false);
			assert(rand_x>=s_min);
			assert(rand_x<=s_max);
	
			if(rand_y < (lnEVAL(Sample, rand_x, H_in,_N))){
				s_out = rand_x;
				break;
			}
			iter++;
			if(iter>500){
				s_out = cpoint;
				printf("Danger3!\n");
				cout << " left cutoff :: " << left_cutoff << " right cutoff :: " << right_cutoff << " cpoint :: " << cpoint << "currentPoint :: " << currentPoint << "glob_best " << glob_best << " " << endl;
				fflush(stdout);
				break;
			}
		}
	}
	//printf(" sample is :: %lf\n",rand_x);;
	return s_out;
}

double Bay_Zipf_Exp::hill_climb(double currentPoint, vector<int> Sample, vector<int> H_in, int _N){
	double bestScore;
	double stepSize = 0.01;
	double candidate[5];
	double epsilon = 0.0001;
	double acceleration = 1.2; // a value such as 1.2 is common
	candidate[0] = -1 * acceleration;
	candidate[1] = -1 / acceleration;
	candidate[2] = 0.00;
	candidate[3] = 1 / acceleration;
	candidate[4] = acceleration;
	int count = 0;
	int iter = 0;
	while(1){
		//cout << " " << candidate[0]<< " " << candidate[1]<< " " << candidate[2]<< " " << candidate[3]<< " " << candidate[4] << endl;
		assert(currentPoint<=s_max);
		assert(currentPoint>=s_min);
		assert(std::isnan(currentPoint)==false);

		double before = lnEVAL(Sample, currentPoint, H_in, _N);
		int best = -1;
		bestScore = -99999999999999999.9999999;
		for(int j=0;j<5;j++){         // try each of 5 candidate locations
			currentPoint = currentPoint + stepSize * candidate[j];
			double temp;
			if((currentPoint<=s_min)||(currentPoint>=s_max)){
				currentPoint = std::max(currentPoint,s_min);
				currentPoint = std::min(currentPoint,s_max);
				
				temp = 0;
			}else{
				assert(std::isnan(currentPoint)==false);

				temp =  lnEVAL(Sample, currentPoint, H_in, _N);
	
				/*if(std::isnan(temp) || temp < 0)
					temp = 0;
				 assert(temp >= 0);
				*/

			}
			currentPoint = currentPoint - stepSize * candidate[j];
			if(temp >= bestScore){
				 bestScore = temp;
				 best = j;
				// cout << temp << endl;
				 assert(best>=0);
			}
		}
		if (candidate[best] == 0.00){
			stepSize = stepSize / acceleration;
		}
		else{
			assert(best>=0);
			currentPoint = currentPoint + stepSize * candidate[best];
			stepSize = stepSize * candidate[best]; // accelerate
			
		}
		stepSize = std::max(stepSize,0.5);
		stepSize = std::min(stepSize,0.001);
		
		assert(std::isinf(stepSize)==false);
		assert(std::isnan(stepSize)==false);
		
		currentPoint = std::max(currentPoint,s_min);
		currentPoint = std::min(currentPoint,s_max);
		assert(best>=0);
		
		
		assert(std::isnan(currentPoint)==false);
		assert(currentPoint<=s_max);
		assert(currentPoint>=s_min);
		if (abs(lnEVAL(Sample,currentPoint,H_in,_N) - before)/before <= epsilon){
			count++;
		}else{
			count = 0;
		}
		if(iter>500){
			cout << bestScore << endl;fflush(stdout);
			break;
		}
		iter++;
		if(count == 10)
			break;
			
	}
	
	assert(currentPoint>=s_min);
	assert(currentPoint<=s_max);
	
	return currentPoint;
}


double Bay_Zipf_Exp::lnEVAL(vector<int> Sample, double s_in, vector<int> H, int _N){
	vector<int> temp_H(N_MAX);
	for(int i=0;i>N_MAX;i++){
		temp_H.at(i) = H.at(i);
	}
	assert(std::isnan(s_in)==false);
	assert(s_in<=s_max);
	assert(s_in>=s_min);
	double temp = get_ln_prob_sample_new(Sample, s_in, temp_H, _N)  + log(gsl_ran_gaussian_pdf (s_in - s_mu, s_theta));

	return temp;
}
	
double Bay_Zipf_Exp::EVAL(vector<int> Sample, double s_in, vector<int> H, int _N){
	vector<int> temp_H(N_MAX);
	for(int i=0;i>N_MAX;i++){
		temp_H.at(i) = H.at(i);
	}
	assert(std::isnan(s_in)==false);
	assert(s_in<=s_max);
	assert(s_in>=s_min);
	
	double temp = get_prob_sample_new(Sample, s_in, temp_H, _N)  * gsl_ran_gaussian_pdf (s_in - s_mu, s_theta);

	return temp;
}
		


