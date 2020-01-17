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
 *    \file include/acado/function/ocp_iterate.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_OCP_ITERATE_HPP
#define ACADO_TOOLKIT_OCP_ITERATE_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/variables_grid/variables_grid.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Data class for storing generic optimization variables.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class OCPiterate is a data class
 *  to store generic optimization variables as arising in the
 *  optimal control context. (Philosophy: This class is used liked
 *  a struct with some add-on functionality.)
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */

class OCPiterate{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        OCPiterate( );

		OCPiterate(	const VariablesGrid* const _x,
					const VariablesGrid* const _xa,
					const VariablesGrid* const _p,
					const VariablesGrid* const _u,
					const VariablesGrid* const _w
					);

        /** Copy constructor (deep copy). */
        OCPiterate( const OCPiterate& rhs );

        /** Destructor. */
        virtual ~OCPiterate( );

        /** Assignment operator (deep copy). */
        OCPiterate& operator=( const OCPiterate& rhs );


		returnValue allocateAll( );
		
		returnValue init(	const VariablesGrid* const _x,
							const VariablesGrid* const _xa,
							const VariablesGrid* const _p,
							const VariablesGrid* const _u,
							const VariablesGrid* const _w
							);

		returnValue clear( );

        /** Assignment operator (deep copy). */
        //OCPiterate& operator+=( const OCPiterate& rhs );


        /** Returns the dimension of the differential state vector. */
        inline uint getNX( ) const;

        /** Returns the dimension of the algebraic state vector. */
        inline uint getNXA( ) const;

        /** Returns the dimension of the parameter vector. */
        inline uint getNP( ) const;

        /** Returns the dimension of the control vector. */
        inline uint getNU( ) const;

        /** Returns the dimension of the disturbance vector. */
        inline uint getNW( ) const;


		uint getNumPoints( ) const;


		returnValue print( ) const;

        inline double getTime( const uint &idx ) const;

        inline DVector getX   ( const uint &idx ) const;
        inline DVector getXA  ( const uint &idx ) const;
        inline DVector getP   ( const uint &idx ) const;
        inline DVector getU   ( const uint &idx ) const;
        inline DVector getW   ( const uint &idx ) const;


        inline Grid getGrid() const;


        /** Returns the union grid of all members, i.e. x, xa, p, u, and w. \n
         *                                                                  \n
         *  \return The requested union grid.                               \n
         */
        Grid getUnionGrid( ) const;


		BooleanType areGridsConsistent( );


        returnValue getInitialData( DVector &x_  = emptyVector,
                                    DVector &xa_ = emptyVector,
                                    DVector &p_  = emptyVector,
                                    DVector &u_  = emptyVector,
                                    DVector &w_  = emptyVector  ) const;


        returnValue updateData(  double t                 ,
                                 DVector &x_  = emptyVector,
                                 DVector &xa_ = emptyVector,
                                 DVector &p_  = emptyVector,
                                 DVector &u_  = emptyVector,
                                 DVector &w_  = emptyVector  );


		returnValue applyStep(	const BlockMatrix& bm,
        						double alpha
								);


        returnValue enableSimulationMode( );

		inline BooleanType isInSimulationMode( ) const;


		/** Shifts all grids by given amount.
		 *
		 *	\return SUCCESSFUL_RETURN \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		virtual returnValue shift(	double timeShift = -1.0,
									DVector  lastX    =  emptyVector,
									DVector lastXA    =  emptyVector,
									DVector lastP     =  emptyVector,
									DVector lastU     =  emptyVector,
									DVector lastW     =  emptyVector );


    //
    // PUBLIC DATA MEMBERS:
    //
    public:

        VariablesGrid  *x ;  // the differential state variables
        VariablesGrid  *xa;  // the algebraic state variables
        VariablesGrid  *p ;  // the parameter
        VariablesGrid  *u ;  // the control variables
        VariablesGrid  *w ;  // the disturbances



    // PROTECTED MEMBER FUNCTIONS:
    // ---------------------------
    protected:

		void copy (const OCPiterate& rhs);

		inline DVector copy ( const VariablesGrid *z, const uint &idx ) const;

		inline uint getDim( VariablesGrid *z ) const;


        void update( double t, VariablesGrid &z1, DVector &z2 ) const;


    //
    // PUBLIC DATA MEMBERS:
    //
    protected:
		
		BooleanType inSimulationMode;
};


CLOSE_NAMESPACE_ACADO


#include <acado/function/ocp_iterate.ipp>

#endif  // ACADO_TOOLKIT_OCP_ITERATE_HPP

/*
 *  end of file
 */
