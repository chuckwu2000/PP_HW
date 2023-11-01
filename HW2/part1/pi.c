#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<time.h>
#include<pthread.h>

double pi;
int thread_total;
long long number_of_toss;
long long number_in_circle;

pthread_mutex_t mutex;

void* toss_func(void* rank)
{
	unsigned int seed = (unsigned)time(NULL) + (unsigned)rank;

	int tmp_in_circle = 0;
	long long size = number_of_toss/thread_total;
	for(long long i = 0;i < size;i++)
	{
		double x = ((double)rand_r(&seed) / RAND_MAX);	//use rand() will loss performance since it is thread safety
		double y = ((double)rand_r(&seed) / RAND_MAX);
		double distance_squared = x * x + y * y;
		if(distance_squared <= 1)
		{
			tmp_in_circle++;
		}
	}
	
	pthread_mutex_lock(&mutex);
	number_in_circle += tmp_in_circle;
	pthread_mutex_unlock(&mutex);
	return NULL;
}

int main(int argc, char* argv[])
{
	if(argc < 3)
	{
		printf("command-line argument not enough !!!\n");
		return 0;
	}

	pi = 0.f;
	thread_total = atoi(argv[1]);
	number_of_toss = atoll(argv[2]);
	number_in_circle = 0;

	pthread_mutex_init(&mutex, NULL);

	pthread_t* thread_arr;
	thread_arr = (pthread_t*)malloc(thread_total * sizeof(pthread_t));

	for(int i = 0;i < thread_total;i++)
	{
		pthread_create(&thread_arr[i], NULL, toss_func, (void*)i);
	}

	for(int i = 0;i < thread_total;i++)
	{
		pthread_join(thread_arr[i], NULL);
	}

	pi = 4 * number_in_circle / ((double) number_of_toss);
	printf("%lf\n",pi);

	pthread_mutex_destroy(&mutex);
	free(thread_arr);
	return 0;
}
