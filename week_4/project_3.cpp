/*
CS 475 - Project #3
Functional Decomposition
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

int	NowYear;		// 2020 - 2025
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population
int     NowNumAlien;            // In addition to this, you must add in some other phenomenon that directly or 
                                // indirectly controls the growth of the grain and/or the graindeer population

const float GRAIN_GROWS_PER_MONTH =		9.0;
const float ONE_DEER_EATS_PER_MONTH =		1.0;
const float ONE_ALIEN_EATS_PER_MONTH =          0.5;    // speical agent

const float AVG_PRECIP_PER_MONTH =		7.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				60.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

unsigned int seed = 0;

//function prototypes

void GrainDeer();
void Grain();
void Watcher();
void Alien ();
float SQR(float);
float Ranf( unsigned int *, float, float);
int Ranf( unsigned int *,  int, int);
float inch_to_cm(float);
float F_to_C(float);

int main( int argc, char *argv[ ] ) {

#ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
#endif

        // starting date and time:
        NowMonth =    0;
        NowYear  = 2020;

        // starting state (feel free to change this if you want):
        NowNumDeer = 1;
        NowNumAlien = 1;
        NowHeight =  1.;

        omp_set_num_threads( 4 );	// same as # of sections
        #pragma omp parallel sections
        {
                #pragma omp section
                {
                        GrainDeer( );
                }

                #pragma omp section
                {
                        Grain( );
                }

                #pragma omp section
                {
                        Watcher( );
                }

                #pragma omp section
                {
                        Alien( );	// your own
                }
        }       // implied barrier -- all functions must return in order
                // to allow any of them to get past here

        return 0;
}

// functions
void GrainDeer(){
        while(NowYear < 2026){
        // compute a temporary next-value for this quantity
	// based on the current state of the simulation:
                int NextNumDeer = NowNumDeer;
        
        // The Carrying Capacity of the graindeer is the number of inches of height of the grain.
        // If the number of graindeer exceeds this value at the end of a month,
        // decrease the number of graindeer by one.
                if (NextNumDeer >= 0){ 
                        if (NowNumDeer > NowHeight){
                                NextNumDeer = NowNumDeer - 1;
                        }

        // If the number of graindeer is less than this value at the end of a month,
        // increase the number of graindeer by one.
                        if (NowNumDeer < NowHeight){
                                NextNumDeer = NowNumDeer + 1;
                        }
                }
        // if the number of deer is negative, then set to 0
                else{
                        NextNumDeer = 0;
                }

	// 1st DoneComputing barrier:
	        #pragma omp barrier

        // after first barrier got through, copy the next state into the now variable
                NowNumDeer = NextNumDeer;

	// 2nd DoneAssigning barrier:
	        #pragma omp barrier
	// according to the barrier schema, empty here 

	// 3rd DonePrinting barrier:
	        #pragma omp barrier
	// go back to 1st barrier
        }
}

void Grain(){
        while(NowYear < 2026){
        // compute a temporary next-value for this quantity
	// based on the current state of the simulation:
                float NextHeight = NowHeight;
        // Note that there is a standard math function, exp( x ), to compute e-to-the-x:
                float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
                float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

        //compute next height
        // add grown height
                float factor_res = tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
                NextHeight = NowHeight + factor_res;
                        
        // take the eaten height by deers and aliens
                NextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
                NextHeight -= (float)NowNumAlien * ONE_ALIEN_EATS_PER_MONTH;
        // Be sure to clamp NowHeight against zero.        
                if (NextHeight < 0.){
                        NextHeight = 0.;
                }
	// 1st DoneComputing barrier:
	        #pragma omp barrier

        // after first barrier got through, copy the next state into the now variable
                NowHeight = NextHeight;

	// 2nd DoneAssigning barrier:
	        #pragma omp barrier
	// according to the barrier schema, empty here 

	// 3rd DonePrinting barrier:
	        #pragma omp barrier
	// go back to 1st barrier

        }
}

void Watcher(){
        while(NowYear < 2026){
	// 1st DoneComputing barrier:
	        #pragma omp barrier

	// 2nd DoneAssigning barrier:
	        #pragma omp barrier
        // The temperature and precipitation are a function of the particular month:
                float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

                float temp = AVG_TEMP - AMP_TEMP * cos( ang );
                NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

                float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
                NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
                if( NowPrecip < 0. )
                        NowPrecip = 0.;

        // Note: if you change the units to °C and centimeters, the quantities might fit better on the same set of axes.
                float c_temp = F_to_C(NowTemp);
                float cm_precip = inch_to_cm(NowPrecip);
                float cm_height = inch_to_cm(NowHeight);
	// according to the barrier schema, Watcher should 
        // 1. print results and increment time
                printf("Date: %d-%d\ttemperature: %f C\tprecipitation: %fcm\tgrain height: %fcm\tthe number of deer: %d\n", NowMonth + 1, NowYear, c_temp, cm_precip, cm_height, NowNumDeer);
        
        //write a result record file
                FILE *result;
                result = fopen("result.txt", "a");
                fprintf(result, "%d-%d\t%f\t%f\t%f\t%d\n", NowMonth + 1, NowYear, c_temp, cm_precip, cm_height, NowNumDeer);
                fclose(result);

                if (NowMonth == 11){
                        //rollback month and increase year
                        NowMonth = 0;
                        NowYear += 1;
                }
                else{
                        NowMonth += 1;
                }
        // 2, Calculate new Environmental Parameters

	// 3rd DonePrinting barrier:
	        #pragma omp barrier
	// go back to 1st barrier

        }
}

void Alien(){
        while(NowYear < 2026){
        // compute a temporary next-value for this quantity
	// based on the current state of the simulation:
                int NextNumAlien = NowNumAlien;
        
        // The Carrying Capacity of the graindeer is the number of inches of height of the grain.
        // If the number of graindeer exceeds 2 times the number of alien at the end of a month,
        // decrease the number of graindeer by one
        // And then increase the number of graindeer by one.
                if (NextNumAlien >= 0){ 
                        if (NowNumDeer > NowNumAlien * 2){
                                NextNumDeer = NowNumDeer - 1;
                                NextNumAlien = NowNumAlien + 1;
                        }

        // If the number of graindeer is less than the number of alien at the end of a month,
        // decrease the number of alien by one.
                        if (NowNumDeer < NowNumAlien){
                                NextNumAlien = NowNumAlien - 1;
                        }
                }
        // if the number of deer is negative, then set to 0
                else{
                        NextNumAlien = 0;
                }
        // 1st DoneComputing barrier:
	        #pragma omp barrier

        // after first barrier got through, copy the next state into the now variable
                NowNumAlien = NextNumAlien;
	// 2nd DoneAssigning barrier:
	        #pragma omp barrier
	// according to the barrier schema, empty here 

	// 3rd DonePrinting barrier:
	        #pragma omp barrier
	// go back to 1st barrier
        }
}

//helper functions

float SQR( float x ){
        return x*x;
}

float Ranf( unsigned int *seedp,  float low, float high ){
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int Ranf( unsigned int *seedp, int ilow, int ihigh ){
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}

float inch_to_cm(float inch){
        return inch * 2.54;
}

float F_to_C(float F_temp){
        return (5./9.) * (F_temp - 32.);
}

