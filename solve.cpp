/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <mpi.h>
#include <malloc.h>

using namespace std;

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
double *alloc1D(int m, int n);

extern control_block cb;

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//

double
L2Norm(double sumSq)
{
    double l2norm = sumSq / (double)((cb.m) * (cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{

    // Simulated time is different from the integer timestep number
    double t = 0.0;
    double mx, sumSq;
    int niter;
    int m = cb.m, n = cb.n;
    int num_process, extra_m, extra_n;
    int top_to_bottom = 0, bottom_to_top = 1, left_to_right = 2, right_to_left = 3;
    register int rank, x_rank, y_rank, m_rank, n_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;

    x_rank = rank / cb.py;
    y_rank = rank % cb.py;

    m_rank = m / cb.px;
    n_rank = n / cb.py;

    //when px pr py don't evenly divide m or n
    extra_m = m % cb.px;
    extra_n = n % cb.py;
    if (x_rank < extra_m)
        m_rank++;
    if (y_rank < extra_n)
        n_rank++;

    register int innerBlockRowStartIndex = (n_rank + 2) + 1;
    // int innerBlockRowEndIndex = (n_rank + 2) * (m_rank) + n_rank;
    register int innerBlockRowEndIndex = (((m_rank + 2) * (n_rank + 2) - 1) - (n_rank)) - (n_rank + 2);

    double *send_left_ghost_cells = alloc1D(m_rank, 1);
    double *send_right_ghost_cells = alloc1D(m_rank, 1);
    double *recv_left_ghost_cells = alloc1D(m_rank, 1);
    double *recv_right_ghost_cells = alloc1D(m_rank, 1);

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    // MPI_Barrier(MPI_COMM_WORLD);
    // if(rank==0){
    // printMat2("Rank 0 Matrix E_prev", E_prev, m_rank,n_rank);
    // printf("=== %d %d %d %d %d %d %d\n", num_process, x_rank, y_rank, n_rank, m_rank, innerBlockRowStartIndex, innerBlockRowEndIndex);
    // }
    // if(rank==1){
    // printMat2("Rank 1 Matrix E_prev", E_prev, m_rank,n_rank);
    // printf("=== %d %d %d %d %d %d %d\n", num_process, x_rank, y_rank, n_rank, m_rank, innerBlockRowStartIndex, innerBlockRowEndIndex);
    // }
    for (niter = 0; niter < cb.niters; niter++)
    {

        if (cb.debug && (niter == 0))
        {
            stats(E_prev, m_rank, n_rank, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m_rank, n_rank, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, m_rank, n_rank);
        }

        // 4 FOR LOOPS set up the padding needed for the boundary and ghost conditions
        int i, j;
        if (!cb.noComm)
        {
            MPI_Request recv_req[4], send_req[4];
            MPI_Status recv_stat[4];

            if (x_rank == 0)
            {

                for (i = 1; i < n_rank + 1; i++)
                {
                    E_prev[i] = E_prev[i + (n_rank + 2) * 2];
                }
            }
            else
            { // Fills in the TOP Ghost Cells

                //send to bottom
                MPI_Isend(E_prev + innerBlockRowStartIndex, n_rank, MPI_DOUBLE, rank - cb.py, top_to_bottom, MPI_COMM_WORLD, &send_req[0]);
                //receive from bottom
                MPI_Irecv(E_prev + 1, n_rank, MPI_DOUBLE, rank - cb.py, bottom_to_top, MPI_COMM_WORLD, &recv_req[0]);
            }
            if (x_rank == cb.px - 1)
            {
                for (i = innerBlockRowEndIndex + (n_rank + 2); i < innerBlockRowEndIndex + (n_rank + 2) + n_rank; i++)
                {
                    E_prev[i] = E_prev[i - (n_rank + 2) * 2];
                }
            }
            else
            { // Fills in the BOTTOM Ghost Cells

                //send to top
                MPI_Isend(E_prev + innerBlockRowEndIndex, n_rank, MPI_DOUBLE, rank + cb.py, bottom_to_top, MPI_COMM_WORLD, &send_req[1]);
                //receive from top
                MPI_Irecv(E_prev + innerBlockRowEndIndex + (n_rank + 2), n_rank, MPI_DOUBLE, rank + cb.py, top_to_bottom, MPI_COMM_WORLD, &recv_req[1]);
            }

            if (y_rank == 0)
            {
                for (i = innerBlockRowStartIndex - 1; i <= innerBlockRowEndIndex - 1; i += (n_rank + 2))
                {
                    E_prev[i] = E_prev[i + 2];
                }
            }
            else
            { // Fills in the LEFT Ghost Cells
                for (i = 0; i < m_rank; i++)
                {
                    send_left_ghost_cells[i] = E_prev[i * (n_rank + 2) + innerBlockRowStartIndex];
                }

                //send to right
                MPI_Isend(send_left_ghost_cells, m_rank, MPI_DOUBLE, rank - 1, left_to_right, MPI_COMM_WORLD, &send_req[2]);
                //receive from right
                MPI_Irecv(recv_left_ghost_cells, m_rank, MPI_DOUBLE, rank - 1, right_to_left, MPI_COMM_WORLD, &recv_req[2]);
            }
            if (y_rank == cb.py - 1)
            {

                for (i = n_rank + innerBlockRowStartIndex; i <= n_rank + innerBlockRowEndIndex; i += (n_rank + 2))
                {
                    E_prev[i] = E_prev[i - 2];
                }
            }
            else
            { // Fills in the RIGHT Ghost Cells
                for (i = 0; i < m_rank; i++)
                {
                    send_right_ghost_cells[i] = E_prev[i * (n_rank + 2) + innerBlockRowStartIndex + n_rank - 1];
                }

                //send to left
                MPI_Isend(send_right_ghost_cells, m_rank, MPI_DOUBLE, rank + 1, right_to_left, MPI_COMM_WORLD, &send_req[3]);
                //receive from right
                MPI_Irecv(recv_right_ghost_cells, m_rank, MPI_DOUBLE, rank + 1, left_to_right, MPI_COMM_WORLD, &recv_req[3]);
            }
            // if(cb.px!=1 || cb.py!=1){
            // MPI_Waitall(4, recv_req, recv_stat);
            if (x_rank != 0)
            {
                MPI_Wait(&recv_req[0], &recv_stat[0]);
            }
            if (x_rank != cb.px - 1)
            {
                MPI_Wait(&recv_req[1], &recv_stat[1]);
            }
            if (y_rank != 0)
            {
                MPI_Wait(&recv_req[2], &recv_stat[2]);
                for (i = 0; i < m_rank; i++)
                {
                    E_prev[(i + 1) * (n_rank + 2)] = recv_left_ghost_cells[i];
                }
            }
            if (y_rank != cb.py - 1)
            {
                MPI_Wait(&recv_req[3], &recv_stat[3]);
                for (i = 0; i < m_rank; i++)
                {
                    E_prev[(i + 1) * (n_rank + 2) + n_rank + 1] = recv_right_ghost_cells[i];
                }
            }
        }
        //////////////////////////////////////////////////////////////////////////////

#define FUSED 1
#define SSE_VEC 1

#ifdef SSE_VEC
        __m128d alph = _mm_set1_pd(alpha);
        __m128d four = _mm_set1_pd(4);
        __m128d neg1 = _mm_set1_pd(-1);
        __m128d val_a = _mm_set1_pd(a);
        __m128d val_kk = _mm_set1_pd(kk);
        __m128d val_dt = _mm_set1_pd(dt);
        __m128d val_b = _mm_set1_pd(b);
        __m128d val_M2 = _mm_set1_pd(M2);
        __m128d val_M1 = _mm_set1_pd(M1);
        __m128d epsil = _mm_set1_pd(epsilon);
#endif

        //#define FUSED 1

#ifdef FUSED
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n_rank + 2))
        {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            R_tmp = R + j;
            int step;
#ifdef SSE_VEC
            step = 2;
#else
            step = 1;
#endif
            for (i = 0; i < n_rank; i += step)
            {
#ifdef SSE_VEC
                __m128d ECen, ETop, EBot, ELef, ERig, ETmp, TmpETmp, RTmp, ECenMinA, ECenMin1, ECenR;
                __m128d Temp1, Temp2, Temp3, Temp4, kkECen, M1R, ECenPluM2, ECenMinB1;
                ECen = _mm_loadu_pd(&E_prev_tmp[i]);
                ETop = _mm_loadu_pd(&E_prev_tmp[i - (n_rank + 2)]);
                EBot = _mm_loadu_pd(&E_prev_tmp[i + (n_rank + 2)]);
                ERig = _mm_loadu_pd(&E_prev_tmp[i + 1]);
                ELef = _mm_loadu_pd(&E_prev_tmp[i - 1]);
                RTmp = _mm_loadu_pd(&R_tmp[i]);
                Temp1 = _mm_mul_pd(ECen, four);
                Temp2 = _mm_add_pd(ETop, EBot);
                Temp3 = _mm_add_pd(ELef, ERig);
                Temp4 = _mm_sub_pd(_mm_add_pd(Temp2, Temp3), Temp1);
                TmpETmp = _mm_add_pd(ECen, _mm_mul_pd(alph, Temp4));
                //First Equation Done
                kkECen = _mm_mul_pd(val_kk, ECen);
                ECenMinA = _mm_sub_pd(ECen, val_a);
                ECenMin1 = _mm_add_pd(ECen, neg1);
                ECenR = _mm_mul_pd(ECen, RTmp);
                M1R = _mm_mul_pd(val_M1, RTmp);
                ECenPluM2 = _mm_add_pd(ECen, val_M2);
                ECenMinB1 = _mm_add_pd(_mm_sub_pd(ECen, val_b), neg1);
                Temp4 = _mm_mul_pd(ECenMinA, ECenMin1);
                Temp1 = _mm_mul_pd(kkECen, ECenMinB1);
                Temp2 = _mm_mul_pd(RTmp, neg1);
                Temp3 = _mm_div_pd(M1R, ECenPluM2);
                ETmp = _mm_sub_pd(TmpETmp, _mm_mul_pd(val_dt, _mm_add_pd(_mm_mul_pd(kkECen, Temp4), ECenR)));
                RTmp = _mm_add_pd(RTmp, _mm_mul_pd(val_dt, _mm_mul_pd(_mm_add_pd(epsil, Temp3), _mm_sub_pd(Temp2, Temp1))));
                _mm_storeu_pd(&E_tmp[i], ETmp);
                _mm_storeu_pd(&R_tmp[i], RTmp);
#else
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n_rank + 2)] + E_prev_tmp[i - (n_rank + 2)]);
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
#endif
            }
        }
#else
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n_rank + 2))
        {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            int step;
#ifdef SSE_VEC
            step = 2;
#else
            step = 1;
#endif
            for (i = 0; i < n_rank; i += step)
            {
#ifdef SSE_VEC
                __m128d ECen, ETop, EBot, ELef, ERig, ETmp, TmpETmp;
                __m128d Temp1, Temp2, Temp3, Temp4;
                ECen = _mm_loadu_pd(&E_prev_tmp[i]);
                ETop = _mm_loadu_pd(&E_prev_tmp[i - (n_rank + 2)]);
                EBot = _mm_loadu_pd(&E_prev_tmp[i + (n_rank + 2)]);
                ERig = _mm_loadu_pd(&E_prev_tmp[i + 1]);
                ELef = _mm_loadu_pd(&E_prev_tmp[i - 1]);
                Temp1 = _mm_mul_pd(ECen, four);
                Temp2 = _mm_add_pd(ETop, EBot);
                Temp3 = _mm_add_pd(ELef, ERig);
                Temp4 = _mm_sub_pd(_mm_add_pd(Temp2, Temp3), Temp1);
                TmpETmp = _mm_add_pd(ECen, _mm_mul_pd(alph, Temp4));
                _mm_storeu_pd(&E_tmp[i], TmpETmp);
#else
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n_rank + 2)] + E_prev_tmp[i - (n_rank + 2)]);
#endif
            }
        }

        /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n_rank + 2))
        {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            int step;
#ifdef SSE_VEC
            step = 2;
#else
            step = 1;
#endif
            for (i = 0; i < n_rank; i += step)
            {
#ifdef SSE_VEC
                __m128d ECen, ETmp, TmpETmp, RTmp, ECenMinA, ECenMin1, ECenR;
                __m128d Temp1, Temp2, Temp3, Temp4, kkECen, M1R, ECenPluM2, ECenMinB1;
                ECen = _mm_loadu_pd(&E_prev_tmp[i]);
                RTmp = _mm_loadu_pd(&R_tmp[i]);
                TmpETmp = _mm_loadu_pd(&E_tmp[i]);
                kkECen = _mm_mul_pd(val_kk, ECen);
                ECenMinA = _mm_sub_pd(ECen, val_a);
                ECenMin1 = _mm_add_pd(ECen, neg1);
                ECenR = _mm_mul_pd(ECen, RTmp);
                M1R = _mm_mul_pd(val_M1, RTmp);
                ECenPluM2 = _mm_add_pd(ECen, val_M2);
                ECenMinB1 = _mm_add_pd(_mm_sub_pd(ECen, val_b), neg1);
                Temp4 = _mm_mul_pd(ECenMinA, ECenMin1);
                Temp1 = _mm_mul_pd(kkECen, ECenMinB1);
                Temp2 = _mm_mul_pd(RTmp, neg1);
                Temp3 = _mm_div_pd(M1R, ECenPluM2);
                ETmp = _mm_sub_pd(TmpETmp, _mm_mul_pd(val_dt, _mm_add_pd(_mm_mul_pd(kkECen, Temp4), ECenR)));
                RTmp = _mm_add_pd(RTmp, _mm_mul_pd(val_dt, _mm_mul_pd(_mm_add_pd(epsil, Temp3), _mm_sub_pd(Temp2, Temp1))));
                _mm_storeu_pd(&E_tmp[i], ETmp);
                _mm_storeu_pd(&R_tmp[i], RTmp);
#else
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
#endif
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq)
        {
            if (!(niter % cb.stats_freq))
            {
                stats(E, m_rank, n_rank, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m_rank, n_rank, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq)
        {
            if (!(niter % cb.plot_freq))
            {
                plotter->updatePlot(E, niter, m_rank, n_rank);
            }
        }

        // Swap current and previous meshes
        double *tmp = E;
        E = E_prev;
        E_prev = tmp;

    } //end of 'niter' loop at the beginning

    // printMat2("Rank 0 Matrix E_prev", E_prev, m_rank,n_rank);  // return the L2 and infinity norms via in-out parameters

    stats(E_prev, m_rank, n_rank, &Linf, &sumSq);
    MPI_Barrier(MPI_COMM_WORLD);
    double total_sumSq, max_Linf;
    MPI_Reduce(&sumSq, &total_sumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Linf, &max_Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    sumSq = total_sumSq;
    Linf = max_Linf;
    L2 = L2Norm(sumSq);
    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}

void printMat2(const char mesg[], double *E, int m, int n)
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
        if ((colIndex > 0) && (colIndex < n + 1))
            if ((rowIndex > 0) && (rowIndex < m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}
