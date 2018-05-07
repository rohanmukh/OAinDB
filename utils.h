#ifndef UTILS_H_   /* Include guard */
#define UTILS_H_


vector<int> get_ranks(vector<int> input1, vector<int> input2);
int cmp(int a, int b);
vector<double> eval_logfactorial(int N_maximum);
long double get_Freq_Join(vector<int> Sample_1  , vector<int> Sample_2, int sample_size);
double virtual_log_sum(double x, double y);
double get_zipf(int in,double s);
double get_zipf_log(int in,double s);
long double get_SE(vector<long double> Freq_val, vector<long double> Orig_Join, int num_runs);

#endif // UTILS_H_