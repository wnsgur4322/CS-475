/*
CS 475 - Project #0
Simple OpenMP Experiment
author: Junhyeok Jeong
email: jeongju@oregonstate.edu
*/

#include <omp.h>
#include <stdio.h>
#include <math.h>

#define NUMT		4
#define SIZE		100000	//you decide
#define NUMTRIES	100	// you decide

float A[SIZE];
float B[SIZE];
float C[SIZE];

int main(){
#ifndef _OPENMP
	fprintf( stderr, "OpenMp is not supported here -- sorry.\n");
	return 1;
#endif
	//check the number of my system cores
	int numprocs = omp_get_num_procs();
	fprintf("the number of cores: %d\n" % numprocs);

	// inialize the arrays:
	for( int i = 0; i < SIZE; i++ )
	{
		A[ i ] = 1.;
		B[ i ] = 2.;
	}

        omp_set_num_threads( NUMT );
        fprintf( stderr, "Using %d threads\n", NUMT );

        double maxMegaMults = 0.;

        for( int t = 0; t < NUMTRIES; t++ )
        {
                double time0 = omp_get_wtime( );

                #pragma omp parallel for
                for( int i = 0; i < SIZE; i++ )
                {
                        C[i] = A[i] * B[i];
                }

                double time1 = omp_get_wtime( );
                double megaMults = (double)SIZE/(time1-time0)/1000000.;
                if( megaMults > maxMegaMults )
                        maxMegaMults = megaMults;
        }

        printf( "Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults );

	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"

        return 0;
}

