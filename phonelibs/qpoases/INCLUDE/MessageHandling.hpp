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
 *	\file INCLUDE/MessageHandling.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the MessageHandling class including global return values.
 */


#ifndef QPOASES_MESSAGEHANDLING_HPP
#define QPOASES_MESSAGEHANDLING_HPP

// #define PC_DEBUG

#ifdef PC_DEBUG
  #include <stdio.h>

  /** Defines an alias for  FILE  from stdio.h. */
  #define myFILE FILE
  /** Defines an alias for  stderr  from stdio.h. */
  #define myStderr stderr
  /** Defines an alias for  stdout  from stdio.h. */
  #define myStdout stdout
#else
  /** Defines an alias for  FILE  from stdio.h. */
  #define myFILE int
  /** Defines an alias for  stderr  from stdio.h. */
  #define myStderr 0
  /** Defines an alias for  stdout  from stdio.h. */
  #define myStdout 0
#endif


#include <Types.hpp>
#include <Constants.hpp>


/** Defines symbols for global return values. \n
 *  Important: All return values are assumed to be nonnegative! */
enum returnValue
{
TERMINAL_LIST_ELEMENT = -1,						/**< Terminal list element, internal usage only! */
/* miscellaneous */
SUCCESSFUL_RETURN = 0,							/**< Successful return. */
RET_DIV_BY_ZERO,		   						/**< Division by zero. */
RET_INDEX_OUT_OF_BOUNDS,						/**< Index out of bounds. */
RET_INVALID_ARGUMENTS,							/**< At least one of the arguments is invalid. */
RET_ERROR_UNDEFINED,							/**< Error number undefined. */
RET_WARNING_UNDEFINED,							/**< Warning number undefined. */
RET_INFO_UNDEFINED,								/**< Info number undefined. */
RET_EWI_UNDEFINED,								/**< Error/warning/info number undefined. */
RET_AVAILABLE_WITH_LINUX_ONLY,					/**< This function is available under Linux only. */
RET_UNKNOWN_BUG,								/**< The error occured is not yet known. */
RET_PRINTLEVEL_CHANGED,							/**< 10 Print level changed. */
RET_NOT_YET_IMPLEMENTED,						/**< Requested function is not yet implemented in this version of qpOASES. */
/* Indexlist */
RET_INDEXLIST_MUST_BE_REORDERD,					/**< Index list has to be reordered. */
RET_INDEXLIST_EXCEEDS_MAX_LENGTH,				/**< Index list exceeds its maximal physical length. */
RET_INDEXLIST_CORRUPTED,						/**< Index list corrupted. */
RET_INDEXLIST_OUTOFBOUNDS,						/**< Physical index is out of bounds. */
RET_INDEXLIST_ADD_FAILED,						/**< Adding indices from another index set failed. */
RET_INDEXLIST_INTERSECT_FAILED,					/**< Intersection with another index set failed. */
/* SubjectTo / Bounds / Constraints */
RET_INDEX_ALREADY_OF_DESIRED_STATUS,			/**< Index is already of desired status. */
RET_ADDINDEX_FAILED,							/**< Cannot swap between different indexsets. */
RET_SWAPINDEX_FAILED,							/**< 20 Adding index to index set failed. */
RET_NOTHING_TO_DO,								/**< Nothing to do. */
RET_SETUP_BOUND_FAILED,							/**< Setting up bound index failed. */
RET_SETUP_CONSTRAINT_FAILED,					/**< Setting up constraint index failed. */
RET_MOVING_BOUND_FAILED,						/**< Moving bound between index sets failed. */
RET_MOVING_CONSTRAINT_FAILED,					/**< Moving constraint between index sets failed. */
/* QProblem */
RET_QP_ALREADY_INITIALISED,						/**< QProblem has already been initialised. */
RET_NO_INIT_WITH_STANDARD_SOLVER,				/**< Initialisation via extern QP solver is not yet implemented. */
RET_RESET_FAILED,								/**< Reset failed. */
RET_INIT_FAILED,								/**< Initialisation failed. */
RET_INIT_FAILED_TQ,								/**< 30 Initialisation failed due to TQ factorisation. */
RET_INIT_FAILED_CHOLESKY,						/**< Initialisation failed due to Cholesky decomposition. */
RET_INIT_FAILED_HOTSTART,						/**< Initialisation failed! QP could not be solved! */
RET_INIT_FAILED_INFEASIBILITY,					/**< Initial QP could not be solved due to infeasibility! */
RET_INIT_FAILED_UNBOUNDEDNESS,					/**< Initial QP could not be solved due to unboundedness! */
RET_INIT_SUCCESSFUL,							/**< Initialisation done. */
RET_OBTAINING_WORKINGSET_FAILED,				/**< Failed to obtain working set for auxiliary QP. */
RET_SETUP_WORKINGSET_FAILED,					/**< Failed to setup working set for auxiliary QP. */
RET_SETUP_AUXILIARYQP_FAILED,					/**< Failed to setup auxiliary QP for initialised homotopy. */
RET_NO_EXTERN_SOLVER,							/**< No extern QP solver available. */
RET_QP_UNBOUNDED,								/**< 40 QP is unbounded. */
RET_QP_INFEASIBLE,								/**< QP is infeasible. */
RET_QP_NOT_SOLVED,								/**< Problems occured while solving QP with standard solver. */
RET_QP_SOLVED,									/**< QP successfully solved. */
RET_UNABLE_TO_SOLVE_QP,							/**< Problems occured while solving QP. */
RET_INITIALISATION_STARTED,						/**< Starting problem initialisation. */
RET_HOTSTART_FAILED,							/**< Unable to perform homotopy due to internal error. */
RET_HOTSTART_FAILED_TO_INIT,					/**< Unable to initialise problem. */
RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED,		/**< Unable to perform homotopy as previous QP is not solved. */
RET_ITERATION_STARTED,							/**< Iteration... */
RET_SHIFT_DETERMINATION_FAILED,					/**< 50 Determination of shift of the QP data failed. */
RET_STEPDIRECTION_DETERMINATION_FAILED,			/**< Determination of step direction failed. */
RET_STEPLENGTH_DETERMINATION_FAILED,			/**< Determination of step direction failed. */
RET_OPTIMAL_SOLUTION_FOUND,						/**< Optimal solution of neighbouring QP found. */
RET_HOMOTOPY_STEP_FAILED,						/**< Unable to perform homotopy step. */
RET_HOTSTART_STOPPED_INFEASIBILITY,				/**< Premature homotopy termination because QP is infeasible. */
RET_HOTSTART_STOPPED_UNBOUNDEDNESS,				/**< Premature homotopy termination because QP is unbounded. */
RET_WORKINGSET_UPDATE_FAILED,					/**< Unable to update working sets according to initial guesses. */
RET_MAX_NWSR_REACHED,							/**< Maximum number of working set recalculations performed. */
RET_CONSTRAINTS_NOT_SPECIFIED,					/**< Problem does comprise constraints! You also have to specify new constraints' bounds. */
RET_INVALID_FACTORISATION_FLAG,					/**< 60 Invalid factorisation flag. */
RET_UNABLE_TO_SAVE_QPDATA,						/**< Unable to save QP data. */
RET_STEPDIRECTION_FAILED_TQ,					/**< Abnormal termination due to TQ factorisation. */
RET_STEPDIRECTION_FAILED_CHOLESKY,				/**< Abnormal termination due to Cholesky factorisation. */
RET_CYCLING_DETECTED,							/**< Cycling detected. */
RET_CYCLING_NOT_RESOLVED,						/**< Cycling cannot be resolved, QP probably infeasible. */
RET_CYCLING_RESOLVED,							/**< Cycling probably resolved. */
RET_STEPSIZE,									/**< For displaying performed stepsize. */
RET_STEPSIZE_NONPOSITIVE,						/**< For displaying non-positive stepsize. */
RET_SETUPSUBJECTTOTYPE_FAILED,					/**< Setup of SubjectToTypes failed. */
RET_ADDCONSTRAINT_FAILED,						/**< 70 Addition of constraint to working set failed. */
RET_ADDCONSTRAINT_FAILED_INFEASIBILITY,			/**< Addition of constraint to working set failed (due to QP infeasibility). */
RET_ADDBOUND_FAILED,							/**< Addition of bound to working set failed. */
RET_ADDBOUND_FAILED_INFEASIBILITY,				/**< Addition of bound to working set failed (due to QP infeasibility). */
RET_REMOVECONSTRAINT_FAILED,					/**< Removal of constraint from working set failed. */
RET_REMOVEBOUND_FAILED,							/**< Removal of bound from working set failed. */
RET_REMOVE_FROM_ACTIVESET,						/**< Removing from active set... */
RET_ADD_TO_ACTIVESET,							/**< Adding to active set... */
RET_REMOVE_FROM_ACTIVESET_FAILED,				/**< Removing from active set failed. */
RET_ADD_TO_ACTIVESET_FAILED,					/**< Adding to active set failed. */
RET_CONSTRAINT_ALREADY_ACTIVE,					/**< 80 Constraint is already active. */
RET_ALL_CONSTRAINTS_ACTIVE,						/**< All constraints are active, no further constraint can be added. */
RET_LINEARLY_DEPENDENT,							/**< New bound/constraint is linearly dependent. */
RET_LINEARLY_INDEPENDENT,						/**< New bound/constraint is linearly independent. */
RET_LI_RESOLVED,								/**< Linear independence of active contraint matrix successfully resolved. */
RET_ENSURELI_FAILED,							/**< Failed to ensure linear indepence of active contraint matrix. */
RET_ENSURELI_FAILED_TQ,							/**< Abnormal termination due to TQ factorisation. */
RET_ENSURELI_FAILED_NOINDEX,					/**< No index found, QP probably infeasible. */
RET_ENSURELI_FAILED_CYCLING,					/**< Cycling detected, QP probably infeasible. */
RET_BOUND_ALREADY_ACTIVE,						/**< Bound is already active. */
RET_ALL_BOUNDS_ACTIVE,							/**< 90 All bounds are active, no further bound can be added. */
RET_CONSTRAINT_NOT_ACTIVE,						/**< Constraint is not active. */
RET_BOUND_NOT_ACTIVE,							/**< Bound is not active. */
RET_HESSIAN_NOT_SPD,							/**< Projected Hessian matrix not positive definite. */
RET_MATRIX_SHIFT_FAILED,						/**< Unable to update matrices or to transform vectors. */
RET_MATRIX_FACTORISATION_FAILED,				/**< Unable to calculate new matrix factorisations. */
RET_PRINT_ITERATION_FAILED,						/**< Unable to print information on current iteration. */
RET_NO_GLOBAL_MESSAGE_OUTPUTFILE,				/**< No global message output file initialised. */
/* Utils */
RET_UNABLE_TO_OPEN_FILE,						/**< Unable to open file. */
RET_UNABLE_TO_WRITE_FILE,						/**< Unable to write into file. */
RET_UNABLE_TO_READ_FILE,						/**< 100 Unable to read from file. */
RET_FILEDATA_INCONSISTENT,						/**< File contains inconsistent data. */
/* SolutionAnalysis */
RET_NO_SOLUTION, 								/**< QP solution does not satisfy KKT optimality conditions. */
RET_INACCURATE_SOLUTION							/**< KKT optimality conditions not satisfied to sufficient accuracy. */
};



/** This class handles all kinds of messages (errors, warnings, infos) initiated
 *  by qpOASES modules and stores the correspoding global preferences.
 *
 *	\author Hans Joachim Ferreau (special thanks to Leonard Wirsching)
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class MessageHandling
{
	/*
	 *	INTERNAL DATA STRUCTURES
	 */
	public:
		/** Data structure for entries in global message list. */
		typedef struct {
			returnValue key;							/**< Global return value. */
			const char* data;							/**< Corresponding message. */
			VisibilityStatus globalVisibilityStatus; 	/**< Determines if message can be printed.
														* 	 If this value is set to VS_HIDDEN, no message is printed! */
		} ReturnValueList;


	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		MessageHandling( );

		/** Constructor which takes the desired output file. */
		MessageHandling(  myFILE* _outputFile					/**< Output file. */
						  );

		/** Constructor which takes the desired visibility states. */
		MessageHandling(	VisibilityStatus _errorVisibility,	/**< Visibility status for error messages. */
							VisibilityStatus _warningVisibility,/**< Visibility status for warning messages. */
							VisibilityStatus _infoVisibility	/**< Visibility status for info messages. */
							);

		/** Constructor which takes the desired output file and desired visibility states. */
		MessageHandling(	myFILE* _outputFile,				/**< Output file. */
							VisibilityStatus _errorVisibility,	/**< Visibility status for error messages. */
							VisibilityStatus _warningVisibility,/**< Visibility status for warning messages. */
							VisibilityStatus _infoVisibility	/**< Visibility status for info messages. */
							);

		/** Copy constructor (deep copy). */
		MessageHandling(	const MessageHandling& rhs	/**< Rhs object. */
							);

		/** Destructor. */
		~MessageHandling( );

		/** Assignment operator (deep copy). */
		MessageHandling& operator=(	const MessageHandling& rhs	/**< Rhs object. */
									);


		/** Prints an error message(a simplified macro THROWERROR is also provided). \n
		 *  Errors are definied as abnormal events which cause an immediate termination of the current (sub) function.
		 *  Errors of a sub function should be commented by the calling function by means of a warning message
		 *  (if this error does not cause an error of the calling function, either)!
		 *  \return Error number returned by sub function call
		 */
		returnValue throwError(
			returnValue Enumber,					/**< Error number returned by sub function call. */
			const char* additionaltext,				/**< Additional error text (0, if none). */
			const char* functionname,				/**< Name of function which caused the error. */
			const char* filename,					/**< Name of file which caused the error. */
			const unsigned long linenumber,			/**< Number of line which caused the error.incompatible binary file */
			VisibilityStatus localVisibilityStatus	/**< Determines (locally) if error message can be printed to myStderr.
			   									 *   If GLOBAL visibility status of the message is set to VS_HIDDEN,
	   			 								 *   no message is printed, anyway! */
		);

		/** Prints a warning message (a simplified macro THROWWARNING is also provided).
		 *  Warnings are definied as abnormal events which does NOT cause an immediate termination of the current (sub) function.
		 *  \return Warning number returned by sub function call
		 */
		returnValue throwWarning(
			returnValue Wnumber,	 				/**< Warning number returned by sub function call. */
			const char* additionaltext,				/**< Additional warning text (0, if none). */
			const char* functionname,				/**< Name of function which caused the warning. */
			const char* filename,   				/**< Name of file which caused the warning. */
			const unsigned long linenumber,	 		/**< Number of line which caused the warning. */
			VisibilityStatus localVisibilityStatus	/**< Determines (locally) if warning message can be printed to myStderr.
		   		  									 *   If GLOBAL visibility status of the message is set to VS_HIDDEN,
					 								 *   no message is printed, anyway! */
			);

	   /** Prints a info message (a simplified macro THROWINFO is also provided).
		 *  \return Info number returned by sub function call
		 */
		returnValue throwInfo(
			returnValue Inumber,	 				/**< Info number returned by sub function call. */
			const char* additionaltext,	 			/**< Additional warning text (0, if none). */
			const char* functionname,				/**< Name of function which submitted the info. */
			const char* filename,   				/**< Name of file which submitted the info. */
			const unsigned long linenumber,			/**< Number of line which submitted the info. */
			VisibilityStatus localVisibilityStatus	/**< Determines (locally) if info message can be printed to myStderr.
				  									 *   If GLOBAL visibility status of the message is set to VS_HIDDEN,
				   									 *   no message is printed, anyway! */
			);


		/** Resets all preferences to default values.
		 *	\return SUCCESSFUL_RETURN */
		returnValue reset( );


		/** Prints a complete list of all messages to output file.
		 *	\return SUCCESSFUL_RETURN */
		returnValue listAllMessages( );


		/** Returns visibility status for error messages.
		 *	\return Visibility status for error messages. */
		inline VisibilityStatus getErrorVisibilityStatus( ) const;

		/** Returns visibility status for warning messages.
		 *	\return Visibility status for warning messages. */
		inline VisibilityStatus getWarningVisibilityStatus( ) const;

		/** Returns visibility status for info messages.
		 *	\return Visibility status for info messages. */
		inline VisibilityStatus getInfoVisibilityStatus( ) const;

		/** Returns pointer to output file.
		 *	\return Pointer to output file. */
		inline myFILE* getOutputFile( ) const;

		/** Returns error count value.
		 *	\return Error count value. */
		inline int getErrorCount( ) const;


		/** Changes visibility status for error messages. */
		inline void setErrorVisibilityStatus(	VisibilityStatus _errorVisibility	/**< New visibility status for error messages. */
												);

		/** Changes visibility status for warning messages. */
		inline void setWarningVisibilityStatus(	VisibilityStatus _warningVisibility	/**< New visibility status for warning messages. */
												);

		/** Changes visibility status for info messages. */
		inline void setInfoVisibilityStatus(	VisibilityStatus _infoVisibility	/**< New visibility status for info messages. */
												);

		/** Changes output file for messages. */
		inline void setOutputFile(	myFILE* _outputFile	/**< New output file for messages. */
									);

		/** Changes error count.
		 * \return SUCCESSFUL_RETURN \n
		 *		   RET_INVALID_ARGUMENT */
		inline returnValue setErrorCount(	int _errorCount	/**< New error count value. */
											);

		/** Return the error code string. */
		static const char* getErrorString(int error);

	/*
	 *	PROTECTED MEMBER FUNCTIONS
	 */
	protected:
		/** Prints a info message to myStderr (auxiliary function).
		 *  \return Error/warning/info number returned by sub function call
		 */
		returnValue throwMessage(
			returnValue RETnumber,	 				/**< Error/warning/info number returned by sub function call. */
			const char* additionaltext,				/**< Additional warning text (0, if none). */
			const char* functionname,				/**< Name of function which caused the error/warning/info. */
			const char* filename,   				/**< Name of file which caused the error/warning/info. */
			const unsigned long linenumber,			/**< Number of line which caused the error/warning/info. */
			VisibilityStatus localVisibilityStatus,	/**< Determines (locally) if info message can be printed to myStderr.
					  								 *   If GLOBAL visibility status of the message is set to VS_HIDDEN,
				   									 *   no message is printed, anyway! */
			const char* RETstring					/**< Leading string of error/warning/info message. */
			);


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		VisibilityStatus errorVisibility;		/**< Error messages visible? */
		VisibilityStatus warningVisibility;		/**< Warning messages visible? */
		VisibilityStatus infoVisibility;		/**< Info messages visible? */

		myFILE* outputFile;						/**< Output file for messages. */

		int errorCount; 						/**< Counts number of errors (for nicer output only). */
};


#ifndef __FUNCTION__
  /** Ensures that __FUNCTION__ macro is defined. */
  #define __FUNCTION__ 0
#endif

#ifndef __FILE__
  /** Ensures that __FILE__ macro is defined. */
  #define __FILE__ 0
#endif

#ifndef __LINE__
  /** Ensures that __LINE__ macro is defined. */
  #define __LINE__ 0
#endif


/** Short version of throwError with default values, only returnValue is needed */
#define THROWERROR(retval) ( getGlobalMessageHandler( )->throwError((retval),0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE) )

/** Short version of throwWarning with default values, only returnValue is needed */
#define THROWWARNING(retval) ( getGlobalMessageHandler( )->throwWarning((retval),0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE) )

/** Short version of throwInfo with default values, only returnValue is needed */
#define THROWINFO(retval) ( getGlobalMessageHandler( )->throwInfo((retval),0,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE) )


/** Returns a pointer to global message handler.
 *  \return Pointer to global message handler.
 */
MessageHandling* getGlobalMessageHandler( );


#include <MessageHandling.ipp>

#endif /* QPOASES_MESSAGEHANDLING_HPP */


/*
 *	end of file
 */
