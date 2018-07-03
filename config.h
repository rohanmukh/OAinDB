#ifndef CONFIG_H_   /* Include guard */
#define CONFIG_H_

const int TOTAL_SIZE_1 = 500000000;
const int TOTAL_SIZE_2 = 500000000;
const int NUM_ID = 100;
const int N_MAX = NUM_ID;
const int N_MIN = 600;
const int NUM_GIBBS_ITER = 200;
const int BURN_IN = 100;
const int NUM_RUNS = 8;
const int NUM_THREADS = 8;
const int MAX_APP_NUM = 10000;
const double KenT = 0.0;
const double corr_alpha =  1.0;
const double corr_beta = 10000.0;
const int lambda = NUM_ID;

// 10,000 /1 when corr = 1
// 1/10,000 when corr = 0
// 100 samples test


const double s_mu = 1.1;
const double s_theta = 0.0001;
const double s_min = 0.1;
const double s_max = 3.0;

#endif // CONFIG_H_
