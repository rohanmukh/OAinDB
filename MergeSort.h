#ifndef MERGE_SORT_H_   /* Include guard */
#define MERGE_SORT_H_


int mergeSort(int arr[], int array_size);
int _mergeSort(int arr[], int temp[], int left, int right);
int merge(int arr[], int temp[], int left, int mid, int right);
	
#endif