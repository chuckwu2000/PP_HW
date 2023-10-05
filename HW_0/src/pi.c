#include<stdlib.h>
#include<stdio.h>
#include<time.h>

int main()
{
	int number_in_circle = 0;
	long long number_of_toss = 100000000;
	double pi;
	srand((unsigned) time(NULL));
	for(long long toss = 0;toss < number_of_toss;toss++)
	{
		double x = ((double)rand() / RAND_MAX);
		double y = ((double)rand() / RAND_MAX);
		double distance_squared = x * x + y * y;
		if(distance_squared <= 1)
		{
			number_in_circle++;
		}
	}
	pi = 4 * number_in_circle / ((double) number_of_toss);
	printf("pi : %lf\n",pi);
	return 0;
}
