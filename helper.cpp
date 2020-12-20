/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include <mpi.h>
#include "cblock.h"

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);
double *alloc1D(int m, int n);

extern control_block cb;
//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init(double *E, double *E_prev, double *R, int m, int n)
{
    int i, rank, num_process, x_rank, y_rank, m_rank, n_rank, extra_m, extra_n;
    int E_cur_tag =0, E_prev_tag = 1, R_tag = 2;
#ifdef _MPI_    
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

    x_rank = rank /cb.py;
    y_rank = rank %cb.py;

    m_rank = m / cb.px;
    n_rank = n / cb.py;

    //when px pr py don't evenly divide m or n
    extra_m = m % cb.px;                                                                                                                                                             
    extra_n = n % cb.py;
    if (x_rank < extra_m)
        m_rank++;
    if (y_rank < extra_n)
        n_rank++;
    // m_rank += (x_rank < m - m_rank * cb.px);
    // n_rank += (y_rank < n - n_rank * cb.py);
    //init matrix and send to other process
    if (rank == 0)
    {

        for (i = 0; i < (m + 2) * (n + 2); i++)
            E_prev[i] = R[i] = 0;

        for (i = (n + 2); i < (m + 1) * (n + 2); i++)
        {
            int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

            // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
                continue;

            E_prev[i] = 1.0;
        }

        for (i = 0; i < (m + 2) * (n + 2); i++)
        {
            int rowIndex = i / (n + 2); // gives the current row number in 2D array representation
            int colIndex = i % (n + 2); // gives the base index (first row's) of the current index

            // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
                continue;

            R[i] = 1.0;
        }
        
        //for each process
        for (int p = num_process-1; p >=0 ; p--)
        {
            int sub_x_rank = p / cb.py;
            int sub_y_rank= p % cb.py;

            int sub_m_rank = m / cb.px;
            sub_m_rank += (sub_x_rank < m - sub_m_rank * cb.px);
            int sub_n_rank = n / cb.py;
            int sub_extra_m = m % cb.px;                                                                                                                                                             
            int sub_extra_n = n % cb.py;
            if (sub_x_rank < sub_extra_m)
                sub_m_rank++;
            if (sub_y_rank < sub_extra_n)
                sub_n_rank++;


            //create size+2 pad arrays
            // double *subarray_E = alloc1D(sub_m_rank + 2, sub_n_rank + 2);
            double *subarray_E_prev = alloc1D(sub_m_rank + 2, sub_n_rank + 2); 
            double *subarray_R = alloc1D(sub_m_rank + 2, sub_n_rank + 2);

            //fill subarrays
            for (int x = 1; x < sub_m_rank+1; x++)
            {
                for (int y = 1; y < sub_n_rank+1; y++)
                {
                    //ignore for process 0; already has all matrices
                    // subarray_E[(sub_n_rank + 2) * x + y] = E[(sub_x_rank * sub_m_rank + x) * (n + 2) + sub_y_rank* sub_n_rank + y];
                    subarray_E_prev[(sub_n_rank + 2) * x + y] = E_prev[(sub_x_rank * sub_m_rank + x) * (n + 2) + sub_y_rank* sub_n_rank + y];
                    subarray_R[(sub_n_rank + 2) * x + y] = R[(sub_x_rank * sub_m_rank + x) * (n + 2) + sub_y_rank* sub_n_rank + y];
                }
            }

            //send to others
            if(p!=0){
                MPI_Request send_req[2];
                MPI_Status send_stat[2];
                // MPI_Isend(subarray_E, (sub_m_rank + 2) * (sub_n_rank + 2), MPI_DOUBLE, p, E_cur_tag, MPI_COMM_WORLD, &send_req[0]);
                MPI_Isend(subarray_E_prev, (sub_m_rank + 2) * (sub_n_rank + 2), MPI_DOUBLE, p, E_prev_tag, MPI_COMM_WORLD, &send_req[0]);
                MPI_Isend(subarray_R, (sub_m_rank + 2) * (sub_n_rank + 2), MPI_DOUBLE, p, R_tag, MPI_COMM_WORLD, &send_req[1]);
                // MPI_Waitall(2,send_req, send_stat);
                MPI_Wait(&send_req[0],&send_stat[0]);
                MPI_Wait(&send_req[1],&send_stat[1]);


                // printf("===subhelper %d %d %d %d\n", p, sub_x_rank, sub_y_rank* sub_n_rank, sub_m_rank);
            }
            else{
                // E = subarray_E;  
                // E_prev = subarray_E_prev;
                // R = subarray_R;
                for (int x = 1; x < sub_m_rank+1; x++)
                {
                    for (int y = 1; y < sub_n_rank+1; y++)
                    {
                        //ignore for process 0; already has all matrices
                        // subarray_E[(sub_n_rank + 2) * x + y] = E[(sub_x_rank * sub_m_rank + x) * (n + 2) + sub_y_rank* sub_n_rank + y];
                        E_prev[(sub_n_rank + 2) * x + y] = subarray_E_prev[(sub_n_rank + 2) * x + y];
                        R[(sub_n_rank + 2) * x + y] = subarray_R[(sub_n_rank + 2) * x + y];
                    }
                }
            }
        }


    }
    //receive submatrix from process 0
    else
    {
        //receive from proc 0
        MPI_Request recv_req[2];
        MPI_Status recv_stat[2];

        // MPI_Irecv(E, (m_rank + 2) * (n_rank + 2), MPI_DOUBLE, 0, E_cur_tag, MPI_COMM_WORLD, &recv_req[0]);
        MPI_Irecv(E_prev, (m_rank + 2) * (n_rank + 2), MPI_DOUBLE, 0, E_prev_tag, MPI_COMM_WORLD, &recv_req[0]);
        MPI_Irecv(R, (m_rank + 2) * (n_rank + 2), MPI_DOUBLE, 0, R_tag, MPI_COMM_WORLD, &recv_req[1]);

        // MPI_Waitall(2, recv_req, recv_stat);
        MPI_Wait(&recv_req[0],&recv_stat[0]);
        MPI_Wait(&recv_req[1],&recv_stat[1]);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // We only print the meshes if they are small enough
#if 1
    // if(rank==0){
    //     printf("rank: %d\n",rank);
    //     printMat("E", E, m_rank, n_rank);
    //     printMat("E_prev", E_prev, m_rank, n_rank);
    //     printMat("R", R, m_rank, n_rank);
    // // }
    // MPI_Barrier(MPI_COMM_WORLD);
#endif
}

double *alloc1D(int m, int n)
{
    int nx = n, ny = m;
    double *temp;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(temp = (double *)memalign(16, sizeof(double) * nx * ny));
    return (temp);
}

void printMat(const char mesg[], double *E, int m, int n)
{
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m > 34)
        return;
#endif
    printf("%s\n", mesg);
    for (i = 0; i < (m + 2) * (n + 2); i++)
    {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        // if ((colIndex > 0) && (colIndex < n + 1))
        //     if ((rowIndex > 0) && (rowIndex < m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}
