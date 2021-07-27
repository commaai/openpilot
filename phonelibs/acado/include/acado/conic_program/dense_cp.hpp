/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *    \file include/acado/conic_program/dense_cp.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_DENSE_CP_HPP
#define ACADO_TOOLKIT_DENSE_CP_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Data class for storing generic conic programs.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class DenseCP (dense conic program) is a data class
 *  to store generic conic programs in a convenient format.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */

class DenseCP{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        DenseCP( );

        /** Copy constructor (deep copy). */
        DenseCP( const DenseCP& rhs );

        /** Destructor. */
        virtual ~DenseCP( );

        /** Assignment operator (deep copy). */
        DenseCP& operator=( const DenseCP& rhs );


        /** Constructs an empty QP with dimensions nV and nC */
        returnValue init( uint nV_, uint nC_ );



//         returnValue setBounds( const DVector &lb,
//                                const DVector &ub  );


        /** Returns whether or not the conic program is an LP */
        inline BooleanType isLP () const;

        /** Returns whether or not the conic program is an LP */
        inline BooleanType isQP () const;

        /** Returns whether or not the conic program is an SDP */
        inline BooleanType isSDP() const;



        /** Returns the number of variables */
        inline uint getNV() const;

        /** Returns the number of linear constraints */
        inline uint getNC() const;


        /** Sets the primal and dual solution converting the dual solution   \n
         *  into the internal format (this routine expects a vector y of     \n
         *  dimension nV + nC, where nC is number of linear constraints and  \n
         *  nV the number of variables (= number of bounds) ).               \n
         *                                                                   \n
         *  \return SUCCESSFUL_RETURN                                        \n
         */
        returnValue setQPsolution( const DVector &x_, const DVector &y_ );


        /** Sets the primal and dual solution converting the dual solution   \n
         *  into the internal format (this routine expects a vector y of     \n
         *  dimension nV + nC, where nC is number of linear constraints and  \n
         *  nV the number of variables (= number of bounds) ).               \n
         *                                                                   \n
         *  \return SUCCESSFUL_RETURN                                        \n
         */
        DVector getMergedDualSolution( ) const;


		/** Prints CP to standard ouput stream. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] startString		Prefix before printing the numerical values.
		 *	@param[in] endString		Suffix after printing the numerical values.
		 *	@param[in] width			Total number of digits per single numerical value.
		 *	@param[in] precision		Number of decimals per single numerical value.
		 *	@param[in] colSeparator		Separator between the columns of the numerical values.
		 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue print(	const char* const name         = DEFAULT_LABEL,
							const char* const startString  = DEFAULT_START_STRING,
							const char* const endString    = DEFAULT_END_STRING,
							uint width                     = DEFAULT_WIDTH,
							uint precision                 = DEFAULT_PRECISION,
							const char* const colSeparator = DEFAULT_COL_SEPARATOR,
							const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
							) const;

		/** Prints CP to standard ouput stream. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue print(	const char* const name,
							PrintScheme printScheme
							) const;

							
		/** Prints CP to file with given name. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] filename			Filename for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] startString		Prefix before printing the numerical values.
		 *	@param[in] endString		Suffix after printing the numerical values.
		 *	@param[in] width			Total number of digits per single numerical value.
		 *	@param[in] precision		Number of decimals per single numerical value.
		 *	@param[in] colSeparator		Separator between the columns of the numerical values.
		 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printToFile(	const char* const filename,
									const char* const name         = DEFAULT_LABEL,
									const char* const startString  = DEFAULT_START_STRING,
									const char* const endString    = DEFAULT_END_STRING,
									uint width                     = DEFAULT_WIDTH,
									uint precision                 = DEFAULT_PRECISION,
									const char* const colSeparator = DEFAULT_COL_SEPARATOR,
									const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
									) const;

		/** Prints CP to given file. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] stream			File for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] startString		Prefix before printing the numerical values.
		 *	@param[in] endString		Suffix after printing the numerical values.
		 *	@param[in] width			Total number of digits per single numerical value.
		 *	@param[in] precision		Number of decimals per single numerical value.
		 *	@param[in] colSeparator		Separator between the columns of the numerical values.
		 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printToFile(	std::ostream& stream,
									const char* const name         = DEFAULT_LABEL,
									const char* const startString  = DEFAULT_START_STRING,
									const char* const endString    = DEFAULT_END_STRING,
									uint width                     = DEFAULT_WIDTH,
									uint precision                 = DEFAULT_PRECISION,
									const char* const colSeparator = DEFAULT_COL_SEPARATOR,
									const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
									) const;

		/** Prints CP to file with given name. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] filename			Filename for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printToFile(	const char* const filename,
									const char* const name,
									PrintScheme printScheme
									) const;

		/** Prints CP to given file. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] stream			File for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printToFile(	std::ostream& stream,
									const char* const name,
									PrintScheme printScheme
									) const;


		/** Prints CP solution to standard ouput stream. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] startString		Prefix before printing the numerical values.
		 *	@param[in] endString		Suffix after printing the numerical values.
		 *	@param[in] width			Total number of digits per single numerical value.
		 *	@param[in] precision		Number of decimals per single numerical value.
		 *	@param[in] colSeparator		Separator between the columns of the numerical values.
		 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printSolution(	const char* const name         = DEFAULT_LABEL,
									const char* const startString  = DEFAULT_START_STRING,
									const char* const endString    = DEFAULT_END_STRING,
									uint width                     = DEFAULT_WIDTH,
									uint precision                 = DEFAULT_PRECISION,
									const char* const colSeparator = DEFAULT_COL_SEPARATOR,
									const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
									) const;

		/** Prints CP solution to standard ouput stream. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printSolution(	const char* const name,
									PrintScheme printScheme
									) const;

		/** Prints CP solution to file with given name. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] filename			Filename for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] startString		Prefix before printing the numerical values.
		 *	@param[in] endString		Suffix after printing the numerical values.
		 *	@param[in] width			Total number of digits per single numerical value.
		 *	@param[in] precision		Number of decimals per single numerical value.
		 *	@param[in] colSeparator		Separator between the columns of the numerical values.
		 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printSolutionToFile(	const char* const filename,
											const char* const name         = DEFAULT_LABEL,
											const char* const startString  = DEFAULT_START_STRING,
											const char* const endString    = DEFAULT_END_STRING,
											uint width                     = DEFAULT_WIDTH,
											uint precision                 = DEFAULT_PRECISION,
											const char* const colSeparator = DEFAULT_COL_SEPARATOR,
											const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
											) const;

		/** Prints CP solution to given file. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] stream			File for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] startString		Prefix before printing the numerical values.
		 *	@param[in] endString		Suffix after printing the numerical values.
		 *	@param[in] width			Total number of digits per single numerical value.
		 *	@param[in] precision		Number of decimals per single numerical value.
		 *	@param[in] colSeparator		Separator between the columns of the numerical values.
		 *	@param[in] rowSeparator		Separator between the rows of the numerical values.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printSolutionToFile(	std::ostream& stream,
											const char* const name         = DEFAULT_LABEL,
											const char* const startString  = DEFAULT_START_STRING,
											const char* const endString    = DEFAULT_END_STRING,
											uint width                     = DEFAULT_WIDTH,
											uint precision                 = DEFAULT_PRECISION,
											const char* const colSeparator = DEFAULT_COL_SEPARATOR,
											const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
											) const;

		/** Prints CP solution to file with given name. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] filename			Filename for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printSolutionToFile(	const char* const filename,
											const char* const name,
											PrintScheme printScheme
											) const;

		/** Prints CP solution to given file. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] stream			File for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue printSolutionToFile(	std::ostream& stream,
											const char* const name,
											PrintScheme printScheme
											) const;


    //
    // PUBLIC DATA MEMBERS:
    //
    public:


    // DIMENSIONS OF THE CP:
    // ---------------------

    uint         nS;    /**< Number of SDP constraints      */


    // DENSE CP IN MATRIX-VECTOR FORMAT:
    // -------------------------------------------------------

    DMatrix        H;    /**< The Hessian matrix             */
    DVector        g;    /**< The objective gradient         */

    DVector       lb;    /**< Simple lower bounds            */
    DVector       ub;    /**< Simple upper bounds            */

    DMatrix        A;    /**< Constraint matrix              */
    DVector      lbA;    /**< Constraint lower bounds        */
    DVector      ubA;    /**< Constraint upper bounds        */

    DMatrix      **B;    /**< SDP constraint tensor          */
    DVector     *lbB;    /**< SDP lower bounds               */
    DVector     *ubB;    /**< SDP upper bounds               */


    // SOLUTION OF THE DENSE CP:
    // -------------------------------------------------------
    DVector       *x;    /**< Primal Solution                */

    DVector     *ylb;    /**< Dual solution, lower bound     */
    DVector     *yub;    /**< Dual solution, upper bound     */

    DVector    *ylbA;    /**< Dual solution, LP lower bound  */
    DVector    *yubA;    /**< Dual solution, LP upper bound  */

    DVector   **ylbB;    /**< Dual solution, SDB lower bound */
    DVector   **yubB;    /**< Dual solution, SDP upper bound */



    // PROTECTED MEMBER FUNCTIONS:
    // ---------------------------
    protected:

    void copy (const DenseCP& rhs);
    void clean();
};


CLOSE_NAMESPACE_ACADO


#include <acado/conic_program/dense_cp.ipp>


#endif  // ACADO_TOOLKIT_DENSE_CP_HPP

/*
 *  end of file
 */
