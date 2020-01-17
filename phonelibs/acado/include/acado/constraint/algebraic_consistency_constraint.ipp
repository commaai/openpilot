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
 *    \file include/acado/constraint/algebraic_consistency_constraint.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


//
// PUBLIC MEMBER FUNCTIONS:
//



BEGIN_NAMESPACE_ACADO



inline int AlgebraicConsistencyConstraint::getNC() const{

    int tmp = 0;
    int run1;

    for( run1 = 0; run1 < counter; run1++ )
        tmp += (breakPoints[run1+1] - breakPoints[run1])*numberOfAlgebraicStates[run1];

    return tmp;
}


inline int AlgebraicConsistencyConstraint::getDim( const int& idx_ ){

    ASSERT( counter == numberOfStages );

    int run1     = 0;
    int stageIdx = 0;

    while( run1 < numberOfStages ){
        if( breakPoints[run1] <= idx_ && breakPoints[run1+1] > idx_ ){
            stageIdx = run1;
            break;
        }
        run1++;
    }
    ASSERT( run1 < numberOfStages );

    return numberOfAlgebraicStates[stageIdx];
}


inline returnValue AlgebraicConsistencyConstraint::add( const uint& endOfStage_  ,
                                                        const DifferentialEquation& dae  ){

    ASSERT( counter < numberOfStages );

    if( fcn == 0 )
        return ACADOERROR(RET_MEMBER_NOT_INITIALISED);

    fcn[counter] = dae;

    numberOfDifferentialStates[counter] = dae.getNumDynamicEquations  ();
    numberOfAlgebraicStates   [counter] = dae.getNumAlgebraicEquations();

    if( counter == 0 ) breakPoints[counter] = 0;

    breakPoints[counter+1] = endOfStage_;

    counter++;
    return SUCCESSFUL_RETURN;
}


// PROTECTED FUNCTIONS:
// --------------------

inline returnValue AlgebraicConsistencyConstraint::computeForwardSensitivityBlock( int offset1, int offset2, int offset3,
                                                                                   int stageIdx, DMatrix *seed ){

    int run1,run2;
    returnValue returnvalue;

    int nc = numberOfAlgebraicStates[stageIdx];

    double* dresult1 = new double[fcn[stageIdx].getDim()                ];
    double*   fseed1 = new double[fcn[stageIdx].getNumberOfVariables()+1];

    if( seed != 0 ){
        int nFDirs = seed->getNumCols();
        DMatrix tmp( nc, nFDirs );
        for( run1 = 0; run1 < nFDirs; run1++ ){

            for( run2 = 0; run2 < fcn[stageIdx].getNumberOfVariables()+1; run2++ )
                        fseed1[run2] = 0.0;

            for( run2 = 0; run2 < (int) seed->getNumRows(); run2++ )
                        fseed1[y_index[stageIdx][offset3+run2]] = seed->operator()(run2,run1);

            returnvalue = fcn[stageIdx].AD_forward( offset1, fseed1, dresult1 );

            if( returnvalue != SUCCESSFUL_RETURN ){
                if( dresult1 != 0 ) delete[] dresult1;
                if( fseed1   != 0 ) delete[] fseed1  ;
                return ACADOERROR(returnvalue);
            }

            for( run2 = 0; run2 < nc; run2++ )
                tmp( run2, run1 ) = dresult1[fcn[stageIdx].getDim() - nc + run2];
        }
        dForward.setDense( offset1, offset2, tmp );
    }

    delete[] dresult1;
    delete[] fseed1  ;

    return SUCCESSFUL_RETURN;
}





CLOSE_NAMESPACE_ACADO

// end of file.
