#include <mpi.h>
#include <cstdio>
#include <iostream>

using namespace std;

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
// build_array & distribute to worker
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int n,m,l;
	int* a_mat = NULL;
	int* b_mat = NULL;

	if(world_rank == 0)	//master handle input
	{
		cin >> n >> m >> l;

		a_mat = (int*)malloc(sizeof(int) * n * m);
		b_mat = (int*)malloc(sizeof(int) * m * l);
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m; j++)
			{
				cin >> a_mat[i * m + j];
			}
		}
		for(int i = 0; i < m ; i++)	//traverse matrix to get space locality
		{
			for(int j = 0; j < l; j++)
			{
				cin >> b_mat[j * m + i];
			}
		}

		for(int i = 1; i < world_size; i++)	//sync info with worker
		{
			MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&m, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&l, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(b_mat, m * l, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&l, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		b_mat = (int*)malloc(sizeof(int) * m * l);
		MPI_Recv(b_mat, m * l, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	*n_ptr = n;
	*m_ptr = m;
	*l_ptr = l;
	*a_mat_ptr = a_mat;
	*b_mat_ptr = b_mat;
	return;
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int* result = (int*)malloc(sizeof(int) * n * l);

	int total_row = n;
	int size = total_row / world_size;
	int offset = size + (total_row % world_size);	//master handle tail(do a little more work)

	if(world_rank == 0)
	{
		MPI_Request *requests = new MPI_Request[world_size - 1];

		for(int i = 0; i < offset; i++)	//master's work
		{
			for(int j = 0; j < l; j++)
			{
				int sum = 0;
				const int* a_mat_row = &(a_mat[i * m]);
				const int* b_mat_row = &(b_mat[j * m]);
				for(int k = 0; k < m; k++)
				{
					sum += (a_mat_row[k] * b_mat_row[k]);
				}
				result[i * l + j] = sum;
			}
		}
		
		for(int i = 1; i < world_size; i++)	//distribute work
		{
			MPI_Send(&a_mat[(offset + (i - 1) * size) * m], size * m, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Irecv(&result[(offset + (i - 1) * size) * l], size * l, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
		}
		MPI_Waitall(world_size - 1, requests, MPI_STATUS_IGNORE);
        delete[] requests;
	}
	else
	{
		int* a_mat_row = (int*)malloc(sizeof(int) * size * m);
		int* part_result = (int*)malloc(sizeof(int) * size * l);

		MPI_Recv(a_mat_row, size * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for(int i = 0; i < size; i++)
		{
			for(int j = 0; j < l; j++)
			{
				int sum = 0;
				const int* b_mat_row = &(b_mat[j * m]);
				for(int k = 0; k < m; k++)
				{
					sum += (a_mat_row[i * m + k] * b_mat_row[k]);
				}
				part_result[i * l + j] = sum;
			}
		}
		MPI_Send(part_result, size * l, MPI_INT, 0, 0, MPI_COMM_WORLD);
		delete[] a_mat_row;
		delete[] part_result;
	}
	
	if(world_rank == 0)
	{
		for(int i = 0; i < n ; i++)	//traverse matrix to get space locality
		{
			for(int j = 0; j < l; j++)
			{
				cout << result[i * l + j] << " ";
			}
			cout << endl;
		}
	}
	return;
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
	delete[] a_mat;
	delete[] b_mat;
	return;
}

