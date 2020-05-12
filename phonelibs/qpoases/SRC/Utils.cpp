/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2008 by Hans Joachim Ferreau et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *	Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file SRC/Utils.cpp
 *	\author Hans Joachim Ferreau, Eckhard Arnold
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of some inlined utilities for working with the different QProblem
 *  classes.
 */


#include <math.h>

#if defined(__WIN32__) || defined(WIN32)
  #include <windows.h>
#elif defined(LINUX)
  #include <sys/stat.h>
  #include <sys/time.h>
#endif

#ifdef __MATLAB__
  #include <mex.h>
#endif


#include <Utils.hpp>



#ifdef PC_DEBUG  /* Define print functions only for debugging! */
/*
 *	p r i n t
 */
returnValue print( const real_t* const v, int n )
{
	int i;
	char myPrintfString[160];

	/* Print a vector. */
	myPrintf( "[\t" );
	for( i=0; i<n; ++i )
	{
		sprintf( myPrintfString," %.16e\t", v[i] );
		myPrintf( myPrintfString );
	}
	myPrintf( "]\n" );

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t
 */
returnValue print(	const real_t* const v, int n,
					const int* const V_idx
					)
{
	int i;
	char myPrintfString[160];

	/* Print a permuted vector. */
	myPrintf( "[\t" );
	for( i=0; i<n; ++i )
	{
		sprintf( myPrintfString," %.16e\t", v[ V_idx[i] ] );
		myPrintf( myPrintfString );
	}
	myPrintf( "]\n" );

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t
 */
returnValue print(	const real_t* const v, int n,
					const char* name
					)
{
	char myPrintfString[160];

	/* Print vector name ... */
	sprintf( myPrintfString,"%s = ", name );
	myPrintf( myPrintfString );

	/* ... and the vector itself. */
	return print( v, n );
}


/*
 *	p r i n t
 */
returnValue print( const real_t* const M, int nrow, int ncol )
{
	int i;

	/* Print a matrix as a collection of row vectors. */
	for( i=0; i<nrow; ++i )
		print( &(M[i*ncol]), ncol );
	myPrintf( "\n" );

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t
 */
returnValue print(	const real_t* const M, int nrow, int ncol,
					const int* const ROW_idx, const int* const COL_idx
					)
{
	int i;

	/* Print a permuted matrix as a collection of permuted row vectors. */
	for( i=0; i<nrow; ++i )
		print( &( M[ ROW_idx[i]*ncol ] ), ncol, COL_idx );
	myPrintf( "\n" );

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t
 */
returnValue print(	const real_t* const M, int nrow, int ncol,
					const char* name
					)
{
	char myPrintfString[160];

	/* Print matrix name ... */
	sprintf( myPrintfString,"%s = ", name );
	myPrintf( myPrintfString );

	/* ... and the matrix itself. */
	return print( M, nrow, ncol );
}


/*
 *	p r i n t
 */
returnValue print( const int* const index, int n )
{
	int i;
	char myPrintfString[160];

	/* Print a indexlist. */
	myPrintf( "[\t" );
	for( i=0; i<n; ++i )
	{
		sprintf( myPrintfString," %d\t", index[i] );
		myPrintf( myPrintfString );
	}
	myPrintf( "]\n" );

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t
 */
returnValue print(	const int* const index, int n,
					const char* name
					)
{
	char myPrintfString[160];

	/* Print indexlist name ... */
	sprintf( myPrintfString,"%s = ", name );
	myPrintf( myPrintfString );

	/* ... and the indexlist itself. */
	return print( index, n );
}


/*
 *	m y P r i n t f
 */
returnValue myPrintf( const char* s )
{
	#ifdef __MATLAB__
	mexPrintf( s );
	#else
	myFILE* outputfile = getGlobalMessageHandler( )->getOutputFile( );
	if ( outputfile == 0 )
		return THROWERROR( RET_NO_GLOBAL_MESSAGE_OUTPUTFILE );

	fprintf( outputfile, "%s", s );
	#endif

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t C o p y r i g h t N o t i c e
 */
returnValue printCopyrightNotice( )
{
	return myPrintf( "\nqpOASES -- An Implementation of the Online Active Set Strategy.\nCopyright (C) 2007-2008 by Hans Joachim Ferreau et al. All rights reserved.\n\nqpOASES is distributed under the terms of the \nGNU Lesser General Public License 2.1 in the hope that it will be \nuseful, but WITHOUT ANY WARRANTY; without even the implied warranty \nof MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. \nSee the GNU Lesser General Public License for more details.\n\n" );
}


/*
 *	r e a d F r o m F i l e
 */
returnValue readFromFile(	real_t* data, int nrow, int ncol,
							const char* datafilename
							)
{
	int i, j;
	float float_data;
	myFILE* datafile;

	/* 1) Open file. */
	if ( ( datafile = fopen( datafilename, "r" ) ) == 0 )
	{
		char errstr[80];
		sprintf( errstr,"(%s)",datafilename );
		return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_OPEN_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
	}

	/* 2) Read data from file. */
	for( i=0; i<nrow; ++i )
	{
		for( j=0; j<ncol; ++j )
		{
			if ( fscanf( datafile, "%f ", &float_data ) == 0 )
			{
				fclose( datafile );
				char errstr[80];
				sprintf( errstr,"(%s)",datafilename );
				return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_READ_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
			}
			data[i*ncol + j] = ( (real_t) float_data );
		}
	}

	/* 3) Close file. */
	fclose( datafile );

	return SUCCESSFUL_RETURN;
}


/*
 *	r e a d F r o m F i l e
 */
returnValue readFromFile(	real_t* data, int n,
							const char* datafilename
							)
{
	return readFromFile( data, n, 1, datafilename );
}



/*
 *	r e a d F r o m F i l e
 */
returnValue readFromFile(	int* data, int n,
							const char* datafilename
							)
{
	int i;
	myFILE* datafile;

	/* 1) Open file. */
	if ( ( datafile = fopen( datafilename, "r" ) ) == 0 )
	{
		char errstr[80];
		sprintf( errstr,"(%s)",datafilename );
		return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_OPEN_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
	}

	/* 2) Read data from file. */
	for( i=0; i<n; ++i )
	{
		if ( fscanf( datafile, "%d\n", &(data[i]) ) == 0 )
		{
			fclose( datafile );
			char errstr[80];
			sprintf( errstr,"(%s)",datafilename );
			return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_READ_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		}
	}

	/* 3) Close file. */
	fclose( datafile );

	return SUCCESSFUL_RETURN;
}


/*
 *	w r i t e I n t o F i l e
 */
returnValue writeIntoFile(	const real_t* const data, int nrow, int ncol,
							const char* datafilename, BooleanType append
							)
{
	int i, j;
	myFILE* datafile;

	/* 1) Open file. */
	if ( append == BT_TRUE )
	{
		/* append data */
		if ( ( datafile = fopen( datafilename, "a" ) ) == 0 )
		{
			char errstr[80];
			sprintf( errstr,"(%s)",datafilename );
			return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_OPEN_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		}
	}
	else
	{
		/* do not append data */
		if ( ( datafile = fopen( datafilename, "w" ) ) == 0 )
		{
			char errstr[80];
			sprintf( errstr,"(%s)",datafilename );
			return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_OPEN_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		}
	}

	/* 2) Write data into file. */
	for( i=0; i<nrow; ++i )
	{
		for( j=0; j<ncol; ++j )
		 	fprintf( datafile, "%.16e ", data[i*ncol+j] );

		fprintf( datafile, "\n" );
	}

	/* 3) Close file. */
	fclose( datafile );

	return SUCCESSFUL_RETURN;
}


/*
 *	w r i t e I n t o F i l e
 */
returnValue writeIntoFile(	const real_t* const data, int n,
							const char* datafilename, BooleanType append
							)
{
	return writeIntoFile( data,1,n,datafilename,append );
}


/*
 *	w r i t e I n t o F i l e
 */
returnValue writeIntoFile(	const int* const data, int n,
							const char* datafilename, BooleanType append
							)
{
	int i;

	myFILE* datafile;

	/* 1) Open file. */
	if ( append == BT_TRUE )
	{
		/* append data */
		if ( ( datafile = fopen( datafilename, "a" ) ) == 0 )
		{
			char errstr[80];
			sprintf( errstr,"(%s)",datafilename );
			return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_OPEN_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		}
	}
	else
	{
		/* do not append data */
		if ( ( datafile = fopen( datafilename, "w" ) ) == 0 )
		{
			char errstr[80];
			sprintf( errstr,"(%s)",datafilename );
			return getGlobalMessageHandler( )->throwError( RET_UNABLE_TO_OPEN_FILE,errstr,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		}
	}

	/* 2) Write data into file. */
	for( i=0; i<n; ++i )
		fprintf( datafile, "%d\n", data[i] );

	/* 3) Close file. */
	fclose( datafile );

	return SUCCESSFUL_RETURN;
}
#endif  /* PC_DEBUG */


/*
 *	g e t C P U t i m e
 */
real_t getCPUtime( )
{
	real_t current_time = -1.0;

	#if defined(__WIN32__) || defined(WIN32)
	LARGE_INTEGER counter, frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&counter);
	current_time = ((real_t) counter.QuadPart) / ((real_t) frequency.QuadPart);
	#elif defined(LINUX)
	struct timeval theclock;
	gettimeofday( &theclock,0 );
	current_time = 1.0*theclock.tv_sec + 1.0e-6*theclock.tv_usec;
	#endif

	return current_time;
}


/*
 *	g e t N o r m
 */
real_t getNorm( const real_t* const v, int n )
{
	int i;

	real_t norm = 0.0;

	for( i=0; i<n; ++i )
		norm += v[i]*v[i];

	return sqrt( norm );
}



/*
 *	end of file
 */
