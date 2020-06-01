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
#include <omp.h>        // for timing
#include <xmmintrin.h>

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	50
#endif

#ifndef SSE_WIDTH
#define SSE_WIDTH		4
#endif


//prototypes from project 4
void SimdMul(float*, float*, float*, int);
float SimdMulSum(float*, float*, int);

int main( int argc, char *argv[ ] ) {

    #ifndef _OPENMP
	    fprintf( stderr, "No OpenMP support!\n" );
	    return 1;
    #endif

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
        
        for( int shift = 0; shift < Size; shift++ ){

	        Sums[shift] = SimdMulSum(&A[0], &A[0+shift], Size);	// note the "fix #2" from false sharing if you are using OpenMP
        }

        double time1 = omp_get_wtime(); // end time

		double megaAutocorPerSecond = (double)(Size * Size) / (time1 - time0) / 1000000.;
		if (megaAutocorPerSecond > maxPerformance){
			maxPerformance = megaAutocorPerSecond;
		}

    }

    // write Sums[*] vs the shift result file for the hidden sine-wave in the signal
	FILE* shift_res = fopen("simd_sums_result.txt", "a");
	for (int i = 1; i < 513; i++){
		fprintf (shift_res, "%d\t%f\n", i, Sums[i]);
	}

	fclose(shift_res);

    // print result and write result.txt file of openMP
    printf("(1) the maxPerformance : %8.2lf MegaMults/Sec\n", maxPerformance);

    FILE *res = fopen("simd_result.txt", "a");
    fprintf(res, "%d\t%f\n", NUMT, maxPerformance);
    fclose(res);

    delete [] A;
    delete [] Sums;

}


// from slide #21 SIMD: multiplication with Intel intrinsics
void SimdMul( float *a, float *b, float *c, int len ){
        int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
        register float *pa = a;
        register float *pb = b;
        register float *pc = c;

        for( int i = 0; i < limit; i += SSE_WIDTH ){
                _mm_storeu_ps( pc, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
                pa += SSE_WIDTH;
                pb += SSE_WIDTH;
                pc += SSE_WIDTH;
        }

        for( int i = limit; i < len; i++ ){
                c[i] = a[i] * b[i];
        }

}

float SimdMulSum( float *a, float *b, int len ) {
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}