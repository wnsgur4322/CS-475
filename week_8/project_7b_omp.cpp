/*
CS 475 - Project #7
Autocorrelation using CPU OpenMP, CPU SIMD, and GPU
author: Junhyeok Jeong
email: jeongju@oregonstate.edu
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// setting the number of threads:
#ifndef NUMT
#define NUMT		1
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

int main( int argc, char *argv[ ] ){

        #ifndef _OPENMP
                fprintf( stderr, "No OpenMP support!\n" );
                return 1;
        #endif

        omp_set_num_threads( NUMT );	// set the number of threads to use in the for-loop:`

        // file read
        FILE *fp = fopen( "signal.txt", "r" );
        if( fp == NULL )
        {
                fprintf( stderr, "Cannot open file 'signal.txt'\n" );
                exit( 1 );
        }
        int Size;
        fscanf( fp, "%d", &Size );
        printf("Size: %d\n", Size);

        float *A =     new float[ 2*Size ];
        float *Sums  = new float[ 1*Size ];
        for( int i = 0; i < Size; i++ )
        {
                fscanf( fp, "%f", &A[i] );
                A[i+Size] = A[i];		// duplicate the array
        }
        fclose( fp );

        // get ready to record the maximum performance
        double maxPerformance = 0.;      // must be declared outside the NUMTRIES loop

        // looking for the maximum performance:
	for (int t = 0; t < NUMTRIES; t++){
		double time0 = omp_get_wtime(); // start time

		#pragma omp parallel for default(none) shared(Size, A, Sums)
		for (int shift = 0; shift < Size; shift++){
			float sum = 0.;
			for (int i = 0; i < Size; i++)
			{
				sum += A[i] * A[i + shift];
			}

			Sums[shift] = sum;
		}

		double time1 = omp_get_wtime(); // end time

		double megaAutocorPerSecond = (double)(Size * Size) / (time1 - time0) / 1000000.;
		if (megaAutocorPerSecond > maxPerformance){
			maxPerformance = megaAutocorPerSecond;
		}
	}

	// write Sums[*] vs the shift result file for the hidden sine-wave in the signal
	FILE* shift_res = fopen("omp_sums_result.txt", "a");
	for (int i = 1; i < 513; i++){
		fprintf (shift_res, "%d\t%f\n", i, Sums[i]);
	}

	fclose(shift_res);

        // print result and write result.txt file of openMP
        printf("(1) the number of threads: %d \n(2) the maxPerformance : %8.2lf MegaMults/Sec\n", NUMT, maxPerformance);

        FILE *res = fopen("omp_result.txt", "a");
        fprintf(res, "%d\t%f\n", NUMT, maxPerformance);
        fclose(res);

        delete [] A;
        delete [] Sums;

}
