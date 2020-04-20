/*
CS 475 - Project #2
Numeric Integration with OpenMP Reduction
author: Junhyeok Jeong
email: jeongju@oregonstate.edu
*/

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

// setting the number of threads (1,2, and 4):
#ifndef NUMT
#define NUMT		1
#endif

// setting the number of subdivisions (NUMNODES):
#ifndef NUMNODES
#define NUMNODES	500
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

// set N value
#ifndef N
#define N		4
#endif

#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

// function prototype
float Height( int, int );

int main( int argc, char *argv[ ] ) {

#ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
#endif

	omp_set_num_threads( NUMT );	// set the number of threads to use in the for-loop:`

	// the area of a single full-sized tile:

	float fullTileArea = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
				( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );

    // get ready to record the maximum performance and the volume:
    float maxPerformance = 0.;      // must be declared outside the NUMTRIES loop
	float avgMegaHeights = 0.;
    double execution_time = 0.;     // delcare execution time
    double volume = 0.;				// volume of the superquadric

    // looking for the maximum performance:
    for( int t = 0; t < NUMTRIES; t++ ){
        double time0 = omp_get_wtime( );

		volume = 0.;
		
	// sum up the weighted heights into the variable "volume"
	// using an OpenMP for loop and a reduction:
    	#pragma omp parallel for default(none) shared(fullTileArea) reduction(+:volume)
    	for( int i = 0; i < NUMNODES*NUMNODES; i++ ){
		    int iu = i % NUMNODES;
		    int iv = i / NUMNODES;
		    float z = Height( iu, iv );
			//printf("iu: %d  iv: %d  Height: %f\n", iu, iv, z);

			// corner: quarter tile of the full tile area
			if ((iu == 0 || iu == NUMNODES - 1) && (iv == 0 || iv == NUMNODES - 1)){
				volume += z * (fullTileArea/(float)4.);
			}
			// edge: half tile of the full tile area
			else if (iu == 0 || iu == NUMNODES - 1 || iv == 0 || iv == NUMNODES -1){
				volume += z * (fullTileArea/(float)2.);
			}
			// full tile area
			else{
				volume += (z * fullTileArea);
			}
    	}

		double time1 = omp_get_wtime( );
		double megaHeightsPerSecond = (double)(NUMNODES * NUMNODES) / ( time1 - time0 ) / 1000000.;
		avgMegaHeights += megaHeightsPerSecond;
		if( megaHeightsPerSecond > maxPerformance ){
			maxPerformance = megaHeightsPerSecond;
		}
        //caculate execution time
        execution_time = time1 - time0;

	}

	avgMegaHeights /= (float) NUMTRIES;
	volume *= 2.;

	// Print out: (1) the number of threads, (2) the number of trials, (3) the probability of hitting the plate, and (4) the MegaTrialsPerSecond.
    printf("(1) the number of threads: %d \n(2) NUMNODES: %d \n(3) volume: %f \n(4) the MegaHeightsPersecond: %8.2lf MegaHeights/Sec\n(5) Average Megaheights: %f\n", NUMT, NUMNODES, volume, maxPerformance, avgMegaHeights );
    printf("(6) execution time: %f \n", execution_time);

	//write a result record file
    ofstream result;
    result.open("result.txt", ios::app);
    result << NUMT << "\t" << NUMNODES << "\t" << volume << "\t" << maxPerformance << "\t" << avgMegaHeights << endl;
    result.close();
    return 0;
}

// functions

float Height( int iu, int iv )	// iu,iv = 0 .. NUMNODES-1
{
	float x = -1.  +  2.*(float)iu /(float)(NUMNODES-1);	// -1. to +1.
	float y = -1.  +  2.*(float)iv /(float)(NUMNODES-1);	// -1. to +1.

	float xn = pow( fabs(x), (double)N );
	float yn = pow( fabs(y), (double)N );
	float r = 1. - xn - yn;
	if( r < 0. )
	        return 0.;
	float height = pow( 1. - xn - yn, 1./(float)N );
	return height;
}
