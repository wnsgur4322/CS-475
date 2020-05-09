/*
CS 475 - Project #4
Vectorized Array Multiplication/Reduction using SSE
author: Junhyeok Jeong
email: jeongju@oregonstate.edu
*/

#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <xmmintrin.h>

using namespace std;

// setting the number of threads (1,2, and 4):
#ifndef NUMT
#define NUMT		4
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	100
#endif

// setting the array size (len)
#ifndef ARR_SIZE
#define ARR_SIZE		1024
#endif

#ifndef SSE_WIDTH
#define SSE_WIDTH		4
#endif

#ifndef NUM_ELEMENTS_PER_CORE
#define NUM_ELEMENTS_PER_CORE   ARR_SIZE/NUMT
#endif

//prototypes
void SimdMul(float*, float*, float*, int);
float SimdMulSum(float*, float*, int);
void non_SimdMul(float*, float*, float*, int);
float non_SimdMulSum(float*, float*, int);
float Ranf( float, float); 

int main( int argc, char *argv[ ] ) {

        #ifndef _OPENMP
	        fprintf( stderr, "No OpenMP support!\n" );
	        return 1;
        #endif
        
        // store sum
        float Simd_sum, non_Simd_sum, threads_sum, threads_SIMD_sum = 0.;

        // get ready to record the maximum performance
        double maxPerformance = 0.;      // must be declared outside the NUMTRIES loop
        double maxPerformance_nonSimd = 0.;      // must be declared outside the NUMTRIES loop
        double maxPerformance_threads = 0.;
        double maxPerformance_threads_SIMD = 0.;
        double megaMultsPerSecond, megaMultsPerSecond_nonSimd, megaMultsPerSecond_threads, megaMultsPerSecond_threads_SIMD;

        // initialize arrays
        float* a = new float[ARR_SIZE];
        float* b = new float[ARR_SIZE];
        float* c = new float[ARR_SIZE];

        double time0, time1;

        // fill up arrays with random values
        for (int i = 0; i < ARR_SIZE; i++){
                a[i] = Ranf(-1.f, 1.f);
                b[i] = Ranf(-1.f, 1.f);
        }

        omp_set_num_threads( NUMT );

        for (int i = 0; i < NUMTRIES; i++){
                // SIMD multiplication/reduction
                // openmp timer start
                time0 = omp_get_wtime( );
                //Simd Multiplication Sum
                Simd_sum = SimdMulSum(a, b, ARR_SIZE);
                //Simd only Multiplication
                //SimdMul(a, b, c, ARR_SIZE);
                // opemp timer end
                time1 = omp_get_wtime( );

                // calculate performance
		megaMultsPerSecond = (double)ARR_SIZE / ( time1 - time0 ) / 1000000.;

                if (megaMultsPerSecond > maxPerformance){
                        maxPerformance = megaMultsPerSecond;
                }

                // non-SIMD multiplication/reduction
                // openmp timer start
                time0 = omp_get_wtime( );
                //non Simd Multiplication Sum
                non_Simd_sum = non_SimdMulSum(a, b, ARR_SIZE);
                // non Simd only Multiplication
                //non_SimdMul(a, b, c, ARR_SIZE);
                // opemp timer end
                time1 = omp_get_wtime( );

                // calculate performance
		megaMultsPerSecond_nonSimd = (double)ARR_SIZE / ( time1 - time0 ) / 1000000.;

                if (megaMultsPerSecond_nonSimd > maxPerformance_nonSimd){
                        maxPerformance_nonSimd = megaMultsPerSecond_nonSimd;
                }

                // threads multiplication/reduction
                time0 = omp_get_wtime( );
                #pragma omp parallel reduction(+:threads_sum)
                {
                        int first = omp_get_thread_num() * NUM_ELEMENTS_PER_CORE;
                        threads_sum += non_SimdMulSum(&a[first], &b[first], NUM_ELEMENTS_PER_CORE);
                }
                time1 = omp_get_wtime( );

                megaMultsPerSecond_threads = (double)ARR_SIZE / ( time1 - time0 ) / 1000000.;
                
                if (megaMultsPerSecond_threads > maxPerformance_threads){
                        maxPerformance_threads = megaMultsPerSecond_threads;
                }

                // SIMD + multithreading multiplication/reduction
                time0 = omp_get_wtime( );
                #pragma omp parallel reduction(+:threads_SIMD_sum)
                {
                        int first = omp_get_thread_num() * NUM_ELEMENTS_PER_CORE;
                        threads_SIMD_sum += SimdMulSum(&a[first], &b[first], NUM_ELEMENTS_PER_CORE);
                }
                time1 = omp_get_wtime( );

                megaMultsPerSecond_threads_SIMD = (double)ARR_SIZE / ( time1 - time0 ) / 1000000.;

                if (megaMultsPerSecond_threads_SIMD > maxPerformance_threads_SIMD){
                        maxPerformance_threads_SIMD = megaMultsPerSecond_threads_SIMD;
                }

                // dummy variable to prevent error from -O3 optimizer
                fprintf(stderr, "%8.2lf\n", Simd_sum + non_Simd_sum + threads_SIMD_sum + threads_sum);


        }

        // print results
        printf("(1) Array size: %d\n(2) MegaMults_SIMD: %8.2lf MegaMults/Sec\n(3) MegaMults_non_SIMD: %8.2lf MegaMults/Sec\n(4) Speed-Up: %8.2lf\n", ARR_SIZE, maxPerformance, maxPerformance_nonSimd, maxPerformance/maxPerformance_nonSimd);
        printf("(5) MegaMults_threads: %8.2lf MegaMults/Sec, Speed-Up: %8.2lf\n(6) MegaMults_threads_SIMD: %8.2lf MegaMults/Sec, Speed-Up: %8.2lf\n", maxPerformance_threads, maxPerformance_threads/maxPerformance_nonSimd, maxPerformance_threads_SIMD, maxPerformance_threads_SIMD/maxPerformance_nonSimd);
        //write a result record file
        FILE *result;
        result = fopen("ec_result.txt", "a");
        fprintf(result, "%d\t%8.2lf\t%8.2lf\t%8.2lf\t%8.2lf\t%8.2lf\t%8.2lf\t%8.2lf\n", ARR_SIZE, maxPerformance, maxPerformance_nonSimd, maxPerformance_threads, maxPerformance_threads_SIMD, maxPerformance/maxPerformance_nonSimd, maxPerformance_threads/maxPerformance_nonSimd, maxPerformance_threads_SIMD/maxPerformance_nonSimd);
        fclose(result);

        // free allocated memories from array
        delete [] a;
        delete [] b;
        delete [] c;

        return 0;
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

void non_SimdMul(float *a, float *b, float *c, int len) {
        
        for (int i = 0; i < len; i++){
                
                // non-SIMD array multiplication
                c[i] = a[i] * b[i];
        }

}

float non_SimdMulSum(float *a, float *b, int len){
        float sum[4] = { 0., 0., 0., 0. };

	for( int i = 0; i < len; i++){
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}

// Helper function
float Ranf( float low, float high ) {
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}
