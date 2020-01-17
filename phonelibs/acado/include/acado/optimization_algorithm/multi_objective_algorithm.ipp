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
 *    \file include/acado/optimization_algorithm/multi_objective_algorithm.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 2009
 */



BEGIN_NAMESPACE_ACADO




inline returnValue MultiObjectiveAlgorithm::setParetoFrontDiscretization( const int &N_ ){

    N = N_;
    return SUCCESSFUL_RETURN;
}

inline returnValue MultiObjectiveAlgorithm::getParetoFront( VariablesGrid &paretoFront ) const{

    int run1, run2;

    if( result.getDim() == 0 )
        return ACADOERROR(RET_MEMBER_NOT_INITIALISED);

    DVector firstColumn(count);
    for( run1 = 0; run1 < count; run1++ )
        firstColumn(run1) = result(run1,0);

    Grid tmp(firstColumn);
    paretoFront.init( m-1, tmp );

    for( run1 = 0; run1 < count; run1++ )
        for( run2 = 0; run2 < m-1; run2++ )
            paretoFront(run1,run2) = result(run1,run2+1);

    return SUCCESSFUL_RETURN;
}



inline returnValue MultiObjectiveAlgorithm::getParetoFrontWithFilter( VariablesGrid &paretoFront ) const{

    if( result.getDim() == 0 )
        return ACADOERROR(RET_MEMBER_NOT_INITIALISED);

    int run1, run2, run3;

    int  nP, check1, check2;
    int *Pp = new int[count];

    nP = 0;
    for( run1 = 0; run1 < count; run1++ ){
        check1 = 0;
        for( run2 = 0; run2 < count; run2++ ){
            check2 = 0;
            for( run3 = 0; run3 < m; run3++ )
                if( result(run2,run3) >= result(run1,run3)-EPS )
                    check2 = 1;
            if( check2 == 0 ) check1 = 1;
        }
        if( check1 == 0 ){
            Pp[nP] = run1;
            nP++;
        }
    }

    DVector firstColumn(nP);
    for( run1 = 0; run1 < nP; run1++ )
        firstColumn(run1) = result(Pp[run1],0);

    Grid tmp(firstColumn);
    paretoFront.init( m-1, tmp );

    for( run1 = 0; run1 < nP; run1++ )
        for( run2 = 0; run2 < m-1; run2++ )
            paretoFront(run1,run2) = result(Pp[run1],run2+1);

    delete[] Pp;

    return SUCCESSFUL_RETURN;
}


inline DMatrix MultiObjectiveAlgorithm::getPayOffMatrix( ) const{

    int run1, run2;
    DMatrix tmp(m,m);

    for( run1 = 0; run1 < m; run1++ )
        for( run2 = 0; run2 < m; run2++ )
            tmp(run1,run2) = vertices(run2,run1) - vertices(run1,run1);

    return tmp;
}


inline DMatrix MultiObjectiveAlgorithm::getNormalizedPayOffMatrix( ) const{

    int run1, run2;
    DMatrix tmp = getPayOffMatrix();
    DVector L   = getNormalizationVector();

    for( run1 = 0; run1 < m; run1++ )
        for( run2 = 0; run2 < m; run2++ )
            tmp(run1,run2) /= L(run1);

    return tmp;
}


inline DVector MultiObjectiveAlgorithm::getUtopiaVector( ) const{

    int run1;
    DVector tmp(m);

    for( run1 = 0; run1 < m; run1++ )
        tmp(run1) = vertices(run1,run1);

    return tmp;
}


inline DVector MultiObjectiveAlgorithm::getNadirVector( ) const{

    int run1,run2;
    DVector tmp(m);
    double max;

    for( run1 = 0; run1 < m; run1++ ){
        max = 0.0;
        for( run2 = 0; run2 < m; run2++ )
            max = acadoMax( max, vertices(run2,run1) );
        tmp(run1) = max;
    }
    return tmp;
}


inline DVector MultiObjectiveAlgorithm::getNormalizationVector( ) const{

    return (getNadirVector()-getUtopiaVector());
}


inline DMatrix MultiObjectiveAlgorithm::getUtopiaPlaneVectors( ) const{

    int run1,run2;
    DMatrix tmp = getNormalizedPayOffMatrix();

    for( run1 = 0; run1 < m; run1++ )
        for( run2 = 0; run2 < m; run2++ )
            tmp(run1,run2) = tmp(run1,m-1) - tmp(run1,run2);

    return tmp;
}



inline DMatrix MultiObjectiveAlgorithm::getWeights() const{

   int tmp = N;
    if( tmp == 0 )
        get( PARETO_FRONT_DISCRETIZATION, tmp );

    WeightGeneration generator;
    DMatrix Weights;
    DVector formers;
    DVector  lb(m);
    DVector  ub(m);
    lb.setZero();
    ub.setAll(1.0);
    generator.getWeights( m, tmp, lb, ub, Weights, formers );

    return Weights;
}



inline returnValue MultiObjectiveAlgorithm::getWeights( const char*fileName ) const{

    DMatrix Weights;
    Weights = getWeights();
    
    Weights.print( fileName );

    return SUCCESSFUL_RETURN;
}



inline returnValue MultiObjectiveAlgorithm::getWeightsWithFilter( const char*fileName ) const{

    if( result.getDim() == 0 )
        return ACADOERROR(RET_MEMBER_NOT_INITIALISED);

    int run1, run2, run3;

    int  nP, check1, check2;
    int *Pp = new int[count];

    nP = 0;
    for( run1 = 0; run1 < count; run1++ ){
        check1 = 0;
        for( run2 = 0; run2 < count; run2++ ){
            check2 = 0;
            for( run3 = 0; run3 < m; run3++ )
                if( result(run2,run3) >= result(run1,run3) - 1000.0*EPS )
                    check2 = 1;
            if( check2 == 0 ) check1 = 1;
        }
        if( check1 == 0 ){
            Pp[nP] = run1;
            nP++;
        }
    }

    DMatrix Weights;
    Weights = getWeights();

    DMatrix FilteredWeights( Weights.getNumRows(), nP );

    for( run1 = 0; run1 < nP; run1++ )
        for( run2 = 0; run2 < (int) Weights.getNumRows(); run2++ )
            FilteredWeights( run2, run1 ) = Weights( run2, Pp[run1] );

	FilteredWeights.print( fileName );

    delete[] Pp;

    return SUCCESSFUL_RETURN;
}


inline returnValue MultiObjectiveAlgorithm::printAuxiliaryRoutine( const char*fileName, VariablesGrid *x_ ) const{

    int run1;

    DMatrix Weights;
    Weights = getWeights();

    for( run1 = 0; run1 < (int) Weights.getNumCols(); run1++ ){
        char *tmp = new char[MAX_LENGTH_STRING];
        sprintf( tmp, "MO%d%s", run1, fileName );
        if( x_[run1].getNumPoints() != 0 ){
        	x_[run1].print( tmp );
        }
        else{
            FILE *file = fopen(tmp,"w");
            fprintf( file, "nan \n" );
            fclose(file);
        }
        delete[] tmp;
    }
    return SUCCESSFUL_RETURN;
}


inline returnValue MultiObjectiveAlgorithm::getAllDifferentialStates( const char*fileName ) const{

    return printAuxiliaryRoutine( fileName, xResults );
}


inline returnValue MultiObjectiveAlgorithm::getAllAlgebraicStates( const char*fileName ) const{

    return printAuxiliaryRoutine( fileName, xaResults );
}


inline returnValue MultiObjectiveAlgorithm::getAllParameters( const char*fileName ) const{

    return printAuxiliaryRoutine( fileName, pResults );
}


inline returnValue MultiObjectiveAlgorithm::getAllControls( const char*fileName ) const{

    return printAuxiliaryRoutine( fileName, uResults );
}


inline returnValue MultiObjectiveAlgorithm::getAllDisturbances( const char*fileName ) const{

    return printAuxiliaryRoutine( fileName, wResults );
}



inline returnValue MultiObjectiveAlgorithm::printInfo()
{
    std::cout << "\n\n--------------- INFO: ------------------------\n";
    std::cout << "\n    Total number of SQP iterations:  " <<  totalNumberOfSQPiterations << std::endl;
    std::cout << "    Total CPU time                :    " << totalCPUtime << " sec" << std::endl;
    std::cout << "\n\n----------------------------------------------\n\n";

    return SUCCESSFUL_RETURN;
}



//
// PROTECTED MEMBER FUNCTIONS:
//



CLOSE_NAMESPACE_ACADO

// end of file.
