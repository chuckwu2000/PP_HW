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

    // TODO: binary tree redunction
	int reduction_time = 0;
    int tmp = world_size;
    while(tmp > 0)
    {
        reduction_time++;
        tmp /= 2;
    }

    for(int i = 0; i < reduction_time; i++)
    {
        int offset = pow(2, i);
        if(world_rank & offset)
        {
            MPI_Send(&number_in_circle_per_process, 1, MPI_LONG_LONG, world_rank - offset, 0, MPI_COMM_WORLD);
            break;  //no need to send data again
        }
        else if(world_rank + offset < world_size) //world_size may smaller than host_total
        {
            int recv_data;
            MPI_Recv(&recv_data, 1, MPI_LONG_LONG, world_rank + offset, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            number_in_circle_per_process += recv_data;
        }
    }

    if (world_rank == 0)
    {
		number_in_circle = number_in_circle_per_process;
        // TODO: PI result
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
