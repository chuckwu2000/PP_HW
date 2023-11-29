#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	long long number_of_toss = tosses;
	long long number_in_circle = 0;
	long long size = number_of_toss / world_size;
	long long number_in_circle_per_process = 0;

	unsigned int seed = (unsigned)time(NULL) * (unsigned)world_rank;
	for(int i = 0; i < size; i++)
	{
		double x = ((double)rand_r(&seed) / RAND_MAX);
		double y = ((double)rand_r(&seed) / RAND_MAX);
		double distance_squared = x * x + y * y;
		if(distance_squared <= 1)
		{
			number_in_circle_per_process++;
		}
	}

    if (world_rank == 0)
    {
        // Master
		MPI_Win_create(&number_in_circle, sizeof(long long), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
		/*  
			MPI_LOCK_EXCLUSIVE -> only one process can access win
			0 -> rank of locked window
			0 -> default optimize
		 */
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		/*  
			0 -> target_rank
			0 -> target_disp
			1 -> target_count
		 */
		MPI_Accumulate(&number_in_circle_per_process, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);
		MPI_Win_unlock(0, win);
    }

	MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
		number_in_circle += number_in_circle_per_process;
		pi_result = 4 * number_in_circle / ((double) number_of_toss);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
