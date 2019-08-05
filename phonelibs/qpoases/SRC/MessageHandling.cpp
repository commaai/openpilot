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
 *	\file SRC/MessageHandling.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the MessageHandling class including global return values.
 *
 */



#include <MessageHandling.hpp>
#include <Utils.hpp>




/** Defines pairs of global return values and messages. */
MessageHandling::ReturnValueList returnValueList[] =
{
/* miscellaneous */
{ SUCCESSFUL_RETURN, "Successful return", VS_VISIBLE },
{ RET_DIV_BY_ZERO, "Division by zero", VS_VISIBLE },
{ RET_INDEX_OUT_OF_BOUNDS, "Index out of bounds", VS_VISIBLE },
{ RET_INVALID_ARGUMENTS, "At least one of the arguments is invalid", VS_VISIBLE },
{ RET_ERROR_UNDEFINED, "Error number undefined", VS_VISIBLE },
{ RET_WARNING_UNDEFINED, "Warning number undefined", VS_VISIBLE },
{ RET_INFO_UNDEFINED, "Info number undefined", VS_VISIBLE },
{ RET_EWI_UNDEFINED, "Error/warning/info number undefined", VS_VISIBLE },
{ RET_AVAILABLE_WITH_LINUX_ONLY, "This function is available under Linux only", VS_HIDDEN },
{ RET_UNKNOWN_BUG, "The error occured is not yet known", VS_VISIBLE },
{ RET_PRINTLEVEL_CHANGED, "Print level changed", VS_VISIBLE },
{ RET_NOT_YET_IMPLEMENTED, "Requested function is not yet implemented.", VS_VISIBLE },
/* Indexlist */
{ RET_INDEXLIST_MUST_BE_REORDERD, "Index list has to be reordered", VS_VISIBLE },
{ RET_INDEXLIST_EXCEEDS_MAX_LENGTH, "Index list exceeds its maximal physical length", VS_VISIBLE },
{ RET_INDEXLIST_CORRUPTED, "Index list corrupted", VS_VISIBLE },
{ RET_INDEXLIST_OUTOFBOUNDS, "Physical index is out of bounds", VS_VISIBLE },
{ RET_INDEXLIST_ADD_FAILED, "Adding indices from another index set failed", VS_VISIBLE },
{ RET_INDEXLIST_INTERSECT_FAILED, "Intersection with another index set failed", VS_VISIBLE },
/* SubjectTo / Bounds / Constraints */
{ RET_INDEX_ALREADY_OF_DESIRED_STATUS, "Index is already of desired status", VS_VISIBLE },
{ RET_SWAPINDEX_FAILED, "Cannot swap between different indexsets", VS_VISIBLE },
{ RET_ADDINDEX_FAILED, "Adding index to index set failed", VS_VISIBLE },
{ RET_NOTHING_TO_DO, "Nothing to do", VS_VISIBLE },
{ RET_SETUP_BOUND_FAILED, "Setting up bound index failed", VS_VISIBLE },
{ RET_SETUP_CONSTRAINT_FAILED, "Setting up constraint index failed", VS_VISIBLE },
{ RET_MOVING_BOUND_FAILED, "Moving bound between index sets failed", VS_VISIBLE },
{ RET_MOVING_CONSTRAINT_FAILED, "Moving constraint between index sets failed", VS_VISIBLE },
/* QProblem */
{ RET_QP_ALREADY_INITIALISED, "QProblem has already been initialised", VS_VISIBLE },
{ RET_NO_INIT_WITH_STANDARD_SOLVER, "Initialisation via extern QP solver is not yet implemented", VS_VISIBLE },
{ RET_RESET_FAILED, "Reset failed", VS_VISIBLE },
{ RET_INIT_FAILED, "Initialisation failed", VS_VISIBLE },
{ RET_INIT_FAILED_TQ, "Initialisation failed due to TQ factorisation", VS_VISIBLE },
{ RET_INIT_FAILED_CHOLESKY, "Initialisation failed due to Cholesky decomposition", VS_VISIBLE },
{ RET_INIT_FAILED_HOTSTART, "Initialisation failed! QP could not be solved!", VS_VISIBLE },
{ RET_INIT_FAILED_INFEASIBILITY, "Initial QP could not be solved due to infeasibility!", VS_VISIBLE },
{ RET_INIT_FAILED_UNBOUNDEDNESS, "Initial QP could not be solved due to unboundedness!", VS_VISIBLE },
{ RET_INIT_SUCCESSFUL, "Initialisation done", VS_VISIBLE },
{ RET_OBTAINING_WORKINGSET_FAILED, "Failed to obtain working set for auxiliary QP", VS_VISIBLE },
{ RET_SETUP_WORKINGSET_FAILED, "Failed to setup working set for auxiliary QP", VS_VISIBLE },
{ RET_SETUP_AUXILIARYQP_FAILED, "Failed to setup auxiliary QP for initialised homotopy", VS_VISIBLE },
{ RET_NO_EXTERN_SOLVER, "No extern QP solver available", VS_VISIBLE },
{ RET_QP_UNBOUNDED, "QP is unbounded", VS_VISIBLE },
{ RET_QP_INFEASIBLE, "QP is infeasible", VS_VISIBLE },
{ RET_QP_NOT_SOLVED, "Problems occured while solving QP with standard solver", VS_VISIBLE },
{ RET_QP_SOLVED, "QP successfully solved", VS_VISIBLE },
{ RET_UNABLE_TO_SOLVE_QP, "Problems occured while solving QP", VS_VISIBLE },
{ RET_INITIALISATION_STARTED, "Starting problem initialisation...", VS_VISIBLE },
{ RET_HOTSTART_FAILED, "Unable to perform homotopy due to internal error", VS_VISIBLE },
{ RET_HOTSTART_FAILED_TO_INIT, "Unable to initialise problem", VS_VISIBLE },
{ RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED, "Unable to perform homotopy as previous QP is not solved", VS_VISIBLE },
{ RET_ITERATION_STARTED, "Iteration", VS_VISIBLE },
{ RET_SHIFT_DETERMINATION_FAILED, "Determination of shift of the QP data failed", VS_VISIBLE },
{ RET_STEPDIRECTION_DETERMINATION_FAILED, "Determination of step direction failed", VS_VISIBLE },
{ RET_STEPLENGTH_DETERMINATION_FAILED, "Determination of step direction failed", VS_VISIBLE },
{ RET_OPTIMAL_SOLUTION_FOUND, "Optimal solution of neighbouring QP found", VS_VISIBLE },
{ RET_HOMOTOPY_STEP_FAILED, "Unable to perform homotopy step", VS_VISIBLE },
{ RET_HOTSTART_STOPPED_INFEASIBILITY, "Premature homotopy termination because QP is infeasible", VS_VISIBLE },
{ RET_HOTSTART_STOPPED_UNBOUNDEDNESS, "Premature homotopy termination because QP is unbounded", VS_VISIBLE },
{ RET_WORKINGSET_UPDATE_FAILED, "Unable to update working sets according to initial guesses", VS_VISIBLE },
{ RET_MAX_NWSR_REACHED, "Maximum number of working set recalculations performed", VS_VISIBLE },
{ RET_CONSTRAINTS_NOT_SPECIFIED, "Problem does comprise constraints! You have to specify new constraints' bounds", VS_VISIBLE },
{ RET_INVALID_FACTORISATION_FLAG, "Invalid factorisation flag", VS_VISIBLE },
{ RET_UNABLE_TO_SAVE_QPDATA, "Unable to save QP data", VS_VISIBLE },
{ RET_STEPDIRECTION_FAILED_TQ, "Abnormal termination due to TQ factorisation", VS_VISIBLE },
{ RET_STEPDIRECTION_FAILED_CHOLESKY, "Abnormal termination due to Cholesky factorisation", VS_VISIBLE },
{ RET_CYCLING_DETECTED, "Cycling detected", VS_VISIBLE },
{ RET_CYCLING_NOT_RESOLVED, "Cycling cannot be resolved, QP is probably infeasible", VS_VISIBLE },
{ RET_CYCLING_RESOLVED, "Cycling probably resolved", VS_VISIBLE },
{ RET_STEPSIZE, "", VS_VISIBLE },
{ RET_STEPSIZE_NONPOSITIVE, "", VS_VISIBLE },
{ RET_SETUPSUBJECTTOTYPE_FAILED, "Setup of SubjectToTypes failed", VS_VISIBLE },
{ RET_ADDCONSTRAINT_FAILED, "Addition of constraint to working set failed", VS_VISIBLE },
{ RET_ADDCONSTRAINT_FAILED_INFEASIBILITY, "Addition of constraint to working set failed", VS_VISIBLE },
{ RET_ADDBOUND_FAILED, "Addition of bound to working set failed", VS_VISIBLE },
{ RET_ADDBOUND_FAILED_INFEASIBILITY, "Addition of bound to working set failed", VS_VISIBLE },
{ RET_REMOVECONSTRAINT_FAILED, "Removal of constraint from working set failed", VS_VISIBLE },
{ RET_REMOVEBOUND_FAILED, "Removal of bound from working set failed", VS_VISIBLE },
{ RET_REMOVE_FROM_ACTIVESET, "Removing from active set:", VS_VISIBLE },
{ RET_ADD_TO_ACTIVESET, "Adding to active set:", VS_VISIBLE },
{ RET_REMOVE_FROM_ACTIVESET_FAILED, "Removing from active set failed", VS_VISIBLE },
{ RET_ADD_TO_ACTIVESET_FAILED, "Adding to active set failed", VS_VISIBLE },
{ RET_CONSTRAINT_ALREADY_ACTIVE, "Constraint is already active", VS_VISIBLE },
{ RET_ALL_CONSTRAINTS_ACTIVE, "All constraints are active, no further constraint can be added", VS_VISIBLE },
{ RET_LINEARLY_DEPENDENT, "New bound/constraint is linearly dependent", VS_VISIBLE },
{ RET_LINEARLY_INDEPENDENT, "New bound/constraint is linearly independent", VS_VISIBLE },
{ RET_LI_RESOLVED, "Linear independence of active contraint matrix successfully resolved", VS_VISIBLE },
{ RET_ENSURELI_FAILED, "Failed to ensure linear indepence of active contraint matrix", VS_VISIBLE },
{ RET_ENSURELI_FAILED_TQ, "Abnormal termination due to TQ factorisation", VS_VISIBLE },
{ RET_ENSURELI_FAILED_NOINDEX, "No index found, QP is probably infeasible", VS_VISIBLE },
{ RET_ENSURELI_FAILED_CYCLING, "Cycling detected, QP is probably infeasible", VS_VISIBLE },
{ RET_BOUND_ALREADY_ACTIVE, "Bound is already active", VS_VISIBLE },
{ RET_ALL_BOUNDS_ACTIVE, "All bounds are active, no further bound can be added", VS_VISIBLE },
{ RET_CONSTRAINT_NOT_ACTIVE, "Constraint is not active", VS_VISIBLE },
{ RET_BOUND_NOT_ACTIVE, "Bound is not active", VS_VISIBLE },
{ RET_HESSIAN_NOT_SPD, "Projected Hessian matrix not positive definite", VS_VISIBLE },
{ RET_MATRIX_SHIFT_FAILED, "Unable to update matrices or to transform vectors", VS_VISIBLE },
{ RET_MATRIX_FACTORISATION_FAILED, "Unable to calculate new matrix factorisations", VS_VISIBLE },
{ RET_PRINT_ITERATION_FAILED, "Unable to print information on current iteration", VS_VISIBLE },
{ RET_NO_GLOBAL_MESSAGE_OUTPUTFILE, "No global message output file initialised", VS_VISIBLE },
/* Utils */
{ RET_UNABLE_TO_OPEN_FILE, "Unable to open file", VS_VISIBLE },
{ RET_UNABLE_TO_WRITE_FILE, "Unable to write into file", VS_VISIBLE },
{ RET_UNABLE_TO_READ_FILE, "Unable to read from file", VS_VISIBLE },
{ RET_FILEDATA_INCONSISTENT, "File contains inconsistent data", VS_VISIBLE },
/* SolutionAnalysis */
{ RET_NO_SOLUTION, "QP solution does not satisfy KKT optimality conditions", VS_VISIBLE },
{ RET_INACCURATE_SOLUTION, "KKT optimality conditions not satisfied to sufficient accuracy", VS_VISIBLE },
{ TERMINAL_LIST_ELEMENT, "", VS_HIDDEN } /* IMPORTANT: Terminal list element! */
};



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	M e s s a g e H a n d l i n g
 */
MessageHandling::MessageHandling( ) :	errorVisibility( VS_VISIBLE ),
										warningVisibility( VS_VISIBLE ),
										infoVisibility( VS_VISIBLE ),
										outputFile( myStdout ),
										errorCount( 0 )
{
}

/*
 *	M e s s a g e H a n d l i n g
 */
MessageHandling::MessageHandling( myFILE* _outputFile ) :
										errorVisibility( VS_VISIBLE ),
										warningVisibility( VS_VISIBLE ),
										infoVisibility( VS_VISIBLE ),
										outputFile( _outputFile ),
										errorCount( 0 )
{
}

/*
 *	M e s s a g e H a n d l i n g
 */
MessageHandling::MessageHandling(	VisibilityStatus _errorVisibility,
									VisibilityStatus _warningVisibility,
		 							VisibilityStatus _infoVisibility
									) :
										errorVisibility( _errorVisibility ),
										warningVisibility( _warningVisibility ),
										infoVisibility( _infoVisibility ),
										outputFile( myStderr ),
										errorCount( 0 )
{
}

/*
 *	M e s s a g e H a n d l i n g
 */
MessageHandling::MessageHandling( 	myFILE* _outputFile,
									VisibilityStatus _errorVisibility,
									VisibilityStatus _warningVisibility,
		 							VisibilityStatus _infoVisibility
									) :
										errorVisibility( _errorVisibility ),
										warningVisibility( _warningVisibility ),
										infoVisibility( _infoVisibility ),
										outputFile( _outputFile ),
										errorCount( 0 )
{
}



/*
 *	M e s s a g e H a n d l i n g
 */
MessageHandling::MessageHandling( const MessageHandling& rhs ) :
										errorVisibility( rhs.errorVisibility ),
										warningVisibility( rhs.warningVisibility ),
										infoVisibility( rhs.infoVisibility ),
										outputFile( rhs.outputFile ),
										errorCount( rhs.errorCount )
{
}


/*
 *	~ M e s s a g e H a n d l i n g
 */
MessageHandling::~MessageHandling( )
{
	#ifdef PC_DEBUG
	if ( outputFile != 0 )
		fclose( outputFile );
	#endif
}


/*
 *	o p e r a t o r =
 */
MessageHandling& MessageHandling::operator=( const MessageHandling& rhs )
{
	if ( this != &rhs )
	{
		errorVisibility = rhs.errorVisibility;
		warningVisibility = rhs.warningVisibility;
		infoVisibility = rhs.infoVisibility;
		outputFile = rhs.outputFile;
		errorCount = rhs.errorCount;
	}

	return *this;
}


/*
 *	t h r o w E r r o r
 */
returnValue MessageHandling::throwError(
	returnValue Enumber,
	const char* additionaltext,
	const char* functionname,
	const char* filename,
	const unsigned long linenumber,
	VisibilityStatus localVisibilityStatus
	)
{
	/* consistency check */
	if ( Enumber <= SUCCESSFUL_RETURN )
		return throwError( RET_ERROR_UNDEFINED,0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );

	/* Call to common throwMessage function if error shall be displayed. */
	if ( errorVisibility == VS_VISIBLE )
		return throwMessage( Enumber,additionaltext,functionname,filename,linenumber,localVisibilityStatus,"ERROR" );
	else
		return Enumber;
}


/*
 *	t h r o w W a r n i n g
 */
returnValue MessageHandling::throwWarning(
	returnValue Wnumber,
	const char* additionaltext,
	const char* functionname,
	const char* filename,
	const unsigned long linenumber,
	VisibilityStatus localVisibilityStatus
  	)
{
	/* consistency check */
  	if ( Wnumber <= SUCCESSFUL_RETURN )
		return throwError( RET_WARNING_UNDEFINED,0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );

	/* Call to common throwMessage function if warning shall be displayed. */
	if ( warningVisibility == VS_VISIBLE )
		return throwMessage( Wnumber,additionaltext,functionname,filename,linenumber,localVisibilityStatus,"WARNING" );
  	else
  		return Wnumber;
}


/*
 *	t h r o w I n f o
 */
returnValue MessageHandling::throwInfo(
  	returnValue Inumber,
	const char* additionaltext,
  	const char* functionname,
	const char* filename,
	const unsigned long linenumber,
	VisibilityStatus localVisibilityStatus
 	)
{
	/* consistency check */
	if ( Inumber < SUCCESSFUL_RETURN )
		return throwError( RET_INFO_UNDEFINED,0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );

	/* Call to common throwMessage function if info shall be displayed. */
	if ( infoVisibility == VS_VISIBLE )
		return throwMessage( Inumber,additionaltext,functionname,filename,linenumber,localVisibilityStatus,"INFO" );
	else
		return Inumber;
}


/*
 *	r e s e t
 */
returnValue MessageHandling::reset( )
{
	setErrorVisibilityStatus(   VS_VISIBLE );
	setWarningVisibilityStatus( VS_VISIBLE );
	setInfoVisibilityStatus(    VS_VISIBLE );

	setOutputFile( myStderr );
	setErrorCount( 0 );

	return SUCCESSFUL_RETURN;
}


/*
 *	l i s t A l l M e s s a g e s
 */
returnValue MessageHandling::listAllMessages( )
{
	#ifdef PC_DEBUG
	int keypos = 0;
	char myPrintfString[160];

	/* Run through whole returnValueList and print each item. */
	while ( returnValueList[keypos].key != TERMINAL_LIST_ELEMENT )
	{
		sprintf( myPrintfString," %d - %s \n",keypos,returnValueList[keypos].data );
		myPrintf( myPrintfString );

		++keypos;
	}
	#endif

	return SUCCESSFUL_RETURN;
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/


#ifdef PC_DEBUG  /* Re-define throwMessage function for embedded code! */

/*
 *	t h r o w M e s s a g e
 */
returnValue MessageHandling::throwMessage(
	returnValue RETnumber,
	const char* additionaltext,
	const char* functionname,
	const char* filename,
	const unsigned long linenumber,
	VisibilityStatus localVisibilityStatus,
	const char* RETstring
 	)
{
	int i;

	int keypos = 0;
	char myPrintfString[160];

	/* 1) Determine number of whitespace for output. */
	char whitespaces[41];
	int numberOfWhitespaces = (errorCount-1)*2;

	if ( numberOfWhitespaces < 0 )
		numberOfWhitespaces = 0;

	if ( numberOfWhitespaces > 40 )
		numberOfWhitespaces = 40;

	for( i=0; i<numberOfWhitespaces; ++i )
		whitespaces[i] = ' ';
	whitespaces[numberOfWhitespaces] = '\0';

	/* 2) Find error/warning/info in list. */
	while ( returnValueList[keypos].key != TERMINAL_LIST_ELEMENT )
	{
		if ( returnValueList[keypos].key == RETnumber )
			break;
		else
			++keypos;
	}

	if ( returnValueList[keypos].key == TERMINAL_LIST_ELEMENT )
	{
		throwError( RET_EWI_UNDEFINED,0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		return RETnumber;
	}

	/* 3) Print error/warning/info. */
	if ( ( returnValueList[keypos].globalVisibilityStatus == VS_VISIBLE ) && ( localVisibilityStatus == VS_VISIBLE ) )
	{
		if ( errorCount > 0 )
		{
			sprintf( myPrintfString,"%s->", whitespaces );
			myPrintf( myPrintfString );
		}

		if ( additionaltext == 0 )
		{
			sprintf(	myPrintfString,"%s (%s, %s:%d): \t%s\n",
						RETstring,functionname,filename,(int)linenumber,returnValueList[keypos].data
						);
			myPrintf( myPrintfString );
		}
		else
		{
			sprintf(	myPrintfString,"%s (%s, %s:%d): \t%s %s\n",
						RETstring,functionname,filename,(int)linenumber,returnValueList[keypos].data,additionaltext
						);
			myPrintf( myPrintfString );
		}

		/* take care of proper indention for subsequent error messages */
		if ( RETstring[0] == 'E' )
		{
			++errorCount;
		}
		else
		{
			if ( errorCount > 0 )
				myPrintf( "\n" );
			errorCount = 0;
		}
	}

	return RETnumber;
}

#else  /* = PC_DEBUG not defined */

/*
 *	t h r o w M e s s a g e
 */
returnValue MessageHandling::throwMessage(
	returnValue RETnumber,
	const char* additionaltext,
	const char* functionname,
	const char* filename,
	const unsigned long linenumber,
	VisibilityStatus localVisibilityStatus,
	const char* RETstring
 	)
{
	/* DUMMY CODE FOR PRETENDING USE OF ARGUMENTS
	 * FOR SUPPRESSING COMPILER WARNINGS! */
	int i = 0;
	if ( additionaltext == 0 ) i++;
	if ( functionname == 0 ) i++;
	if ( filename == 0 ) i++;
	if ( linenumber == 0 ) i++;
	if ( localVisibilityStatus == VS_VISIBLE ) i++;
	if ( RETstring == 0 ) i++;
	/* END OF DUMMY CODE */

	return RETnumber;
}

#endif  /* PC_DEBUG */



/*****************************************************************************
 *  G L O B A L  M E S S A G E  H A N D L E R                                *
 *****************************************************************************/


/** Global message handler for all qpOASES modules.*/
MessageHandling globalMessageHandler( myStderr,VS_VISIBLE,VS_VISIBLE,VS_VISIBLE );


/*
 *	g e t G l o b a l M e s s a g e H a n d l e r
 */
MessageHandling* getGlobalMessageHandler( )
{
	return &globalMessageHandler;
}

const char* MessageHandling::getErrorString(int error)
{
	return returnValueList[ error ].data;
}



/*
 *	end of file
 */
