#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:		   graph to process (see common/graph.h)
// solution:	array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:	 page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  	int numNodes = num_nodes(g);
  	double equal_prob = 1.0 / numNodes;
  	for (int i = 0; i < numNodes; ++i)
  	{
		solution[i] = equal_prob;
  	}

  	double* node_out_edge_give = (double*)malloc(sizeof(double) * numNodes);
	int converged = 0;

	while(!converged)
	{
		int node_no_out_edge_sum = 0;
		for (int i = 0; i < numNodes; ++i)
		{
			int size = outgoing_size(g, i);
			if(size == 0)
		  	{
				node_no_out_edge_sum += solution[i];
			}
			else
			{
				node_out_edge_give[i] = solution[i] / size;
			}
		}
		node_no_out_edge_sum = node_no_out_edge_sum * (damping / numNodes);

		printf("node_no_out_edge_sum : %d\n",node_no_out_edge_sum);

		double global_diff = 0;
		for(int i = 0;i < numNodes;i++)
		{
			double score_new = 0.0;
			const Vertex* in_edge_begin = incoming_begin(g, i);
			const Vertex* in_edge_end = incoming_end(g, i);

			for(const Vertex *iter = in_edge_begin;iter != in_edge_end;iter++)
			{
				score_new = score_new + node_out_edge_give[*iter];
			}

			score_new = (damping * score_new) + (1.0-damping) / numNodes + node_no_out_edge_sum;

			global_diff  = global_diff + abs(solution[i] - score_new);
			solution[i] = abs(solution[i] - score_new);
		}

		printf("global_diff : %lf , convergence : %lf\n",global_diff, convergence);

		converged = (global_diff < convergence) ?1 :0;
	}

	free(node_out_edge_give);
	return;
  /*
	 For PP students: Implement the page rank algorithm here.  You
	 are expected to parallelize the algorithm using openMP.  Your
	 solution may need to allocate (and free) temporary arrays.

	 Basic page rank pseudocode is provided below to get you started:

	 // initialization: see example code above
	 score_old[vi] = 1/numNodes;

	 while (!converged) {

	   // compute score_new[vi] for all nodes vi:
	   score_new[vi] = sum over all nodes vj reachable from incoming edges
						  { score_old[vj] / number of edges leaving vj  }
	   score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

	   score_new[vi] += sum over all nodes v in graph with no outgoing edges
						  { damping * score_old[v] / numNodes }

	   // compute how much per-node scores have changed
	   // quit once algorithm has converged

	   global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
	   converged = (global_diff < convergence)
	 }

   */
}
