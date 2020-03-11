#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<mpi.h>
int main(int argc, char *argv[])
{
	int rank, size, elements_per_process, n_elements_recieved, i, j, k;
	int n = atoi(argv[1]), A[n][n],B[n][n],C[n][n];
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank==0) {
		int r0,r1,a,m;
		a = 25;
		m = 997;
		r0 = rand()%m;
		r1 = rand()%m;
		for(i=0;i<n;i++) {
			for(j=0;j<n;j++) {
				r0 = (a*r0)%m;
				r1 = (a*r1)%m;
				A[i][j] = r0;
				B[i][j] = r1;
			}
		}
		printf("A :\n");
		for(i=0;i<n;i++) {
			for(j=0;j<n;j++)
				printf("%d ", A[i][j]);
			printf("\n");
		}printf("B :\n");
		for(i=0;i<n;i++) {
			for(j=0;j<n;j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}
	}
	
	// if(n<size){ size = n;}
	if(rank<size){
		// int from = rank*(n/size), to = (rank+1)*(n/size);
		MPI_Bcast (B, n*n, MPI_INT, 0, MPI_COMM_WORLD);

		// compute the work distribution
	    int remainder = (n % size)*n;
	    int local_counts[size], offsets[size];
	    int sum = 0;
	    for (i = 0; i < size; i++) {
	        local_counts[i] = (n / size)*n;
	        if (remainder > 0) {
	            local_counts[i] += n;
	            remainder-=n;
	        }
	        offsets[i] = sum;
	        sum += local_counts[i];
	    }

	    // int localArray[local_counts[rank]/n][n];
	    int from = offsets[rank]/n, to = offsets[rank]/n+local_counts[rank]/n;
	    
		MPI_Scatterv(A, local_counts, offsets, MPI_INT, A[from], local_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
		for (i = from; i < to; ++i)
		{
			for(j=0;j<n;j++){
				C[i][j] = 0;
				for (k = 0; k < n; ++k)
				{
					C[i][j] += A[i][k]*B[k][j];
					// printf("%d ", A[i][k]);
				}
				// printf("from:%d to:%d i:%d j:%d Cij = %d\n", from,to,i,j,C[i][j]);
			}

		}
		MPI_Gatherv(C[from], local_counts[rank], MPI_INT, C, local_counts, offsets, MPI_INT, 0, MPI_COMM_WORLD);
	}
	if(rank==0){
		int D[n][n];
		for (int i = 0; i < n; ++i) {
	        for (int j = 0; j < n; ++j) {
	            D[i][j] = 0;
	        }
	    }

	    for (int i = 0; i < n; ++i) {
	        for (int j = 0; j < n; ++j) {
	            for (int k = 0; k < n; ++k) {
	                D[i][j] += A[i][k] * B[k][j];
	            }
	        }
	    }
	
		int err = 0;
		for(int i=0;i<n;i++) {
			for(int j=0;j<n;j++) {
				if(D[i][j]!=C[i][j])
					{err = 1;break;}
			}
		}
		
		printf("C :\n");
		for(i=0;i<n;i++) {
			for(j=0;j<n;j++)
				printf("%d ", C[i][j]);
			printf("\n");
		}printf("\n");
		if(err==0)
			printf("Calculated correctly!!\n");
		else
			printf("!!!!ERROR!!!!\n");

	}
	MPI_Finalize();
	return 0;
}