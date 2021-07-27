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
*    \file include/acado/symbolic_operator/symbolic_index_list.ipp
*    \author Boris Houska, Hans Joachim Ferreau
*    \date 2008
*/



BEGIN_NAMESPACE_ACADO



inline int SymbolicIndexList::getNumberOfOperators(){

    return numberOfOperators;
}


inline SymbolicIndexList* SymbolicIndexList::substitute( VariableType variableType_, int index_ ){

    SymbolicIndexList tmp(*this);

    int       run1, run2                 ;
    const int numberOfVariableTypes = 11 ;
    int       type                  = -1 ;

    if(variableType_ == VT_DIFFERENTIAL_STATE )  type = 0;
    if(variableType_ == VT_ALGEBRAIC_STATE    )  type = 1;
    if(variableType_ == VT_CONTROL            )  type = 2;
    if(variableType_ == VT_INTEGER_CONTROL    )  type = 3;
    if(variableType_ == VT_PARAMETER          )  type = 4;
    if(variableType_ == VT_INTEGER_PARAMETER  )  type = 5;
    if(variableType_ == VT_DISTURBANCE        )  type = 6;
    if(variableType_ == VT_TIME               )  type = 7;
    if(variableType_ == VT_INTERMEDIATE_STATE )  type = 8;
    if(variableType_ == VT_DDIFFERENTIAL_STATE)  type = 9;
    if(variableType_ == VT_ONLINE_DATA        )  type = 10;

    if( type == -1 ) return this;



    for( run1 = 0; run1 < numberOfVariableTypes; run1++ ){

        if( tmp.maxNumberOfEntries[run1] > 0 ){

            free(tmp.entryExists  [run1]);
            free(tmp.variableIndex[run1]);
            free(tmp.variableScale[run1]);
        }
    }

    delete[] tmp.maxNumberOfEntries;
    delete[] tmp.variableIndex;
    delete[] tmp.variableScale;
    delete[] tmp.entryExists;


    tmp.entryExists        = new BooleanType*[numberOfVariableTypes];
    tmp.variableIndex      = new int        *[numberOfVariableTypes];
    tmp.variableScale      = new double     *[numberOfVariableTypes];
    tmp.maxNumberOfEntries = new int         [numberOfVariableTypes];


    for( run1 = 0; run1 < numberOfVariableTypes; run1++ ){

        if( run1 != type ){
            tmp.maxNumberOfEntries[run1] = maxNumberOfEntries[run1];
            if( maxNumberOfEntries[run1] > 0 ){
                tmp.entryExists       [run1] = (BooleanType*)calloc(maxNumberOfEntries[run1],
                                               sizeof(BooleanType));
                tmp.variableIndex     [run1] = (int*)calloc(maxNumberOfEntries[run1],
                                               sizeof(int));
                tmp.variableScale     [run1] = (double*)calloc(maxNumberOfEntries[run1],
                                               sizeof(double));
            }
            for( run2 = 0; run2 < maxNumberOfEntries[run1]; run2++ ){
                tmp.entryExists  [run1][run2] = entryExists  [run1][run2];
                tmp.variableIndex[run1][run2] = variableIndex[run1][run2];
                tmp.variableScale[run1][run2] = variableScale[run1][run2];
            }
        }
        else{

            if( index_ < maxNumberOfEntries[run1] ) tmp.maxNumberOfEntries[run1] = maxNumberOfEntries[run1]-1;
            else                                    tmp.maxNumberOfEntries[run1] = maxNumberOfEntries[run1]  ;

            if( tmp.maxNumberOfEntries[run1] > 0 ){
                tmp.entryExists       [run1] = (BooleanType*)calloc(tmp.maxNumberOfEntries[run1],
                                               sizeof(BooleanType));
                tmp.variableIndex     [run1] = (int*)calloc(tmp.maxNumberOfEntries[run1],
                                               sizeof(int));
                tmp.variableScale     [run1] = (double*)calloc(tmp.maxNumberOfEntries[run1],
                                               sizeof(double));
            }
            for( run2 = 0; run2 < tmp.maxNumberOfEntries[run1]; run2++ ){

                if( run2 < index_ ){
                    tmp.entryExists  [run1][run2] = entryExists  [run1][run2];
                    tmp.variableIndex[run1][run2] = variableIndex[run1][run2];
                    tmp.variableScale[run1][run2] = variableScale[run1][run2];
                }
                else{
                    tmp.entryExists  [run1][run2] = entryExists  [run1][run2+1];
                    tmp.variableIndex[run1][run2] = variableIndex[run1][run2+1];
                    tmp.variableScale[run1][run2] = variableScale[run1][run2+1];
                }
            }
        }
    }
    tmp.variableCounter = variableCounter;

    return new SymbolicIndexList(tmp);
}



inline int SymbolicIndexList::index( VariableType variableType_, int index_ ) const{

    switch(variableType_){

        case VT_DIFFERENTIAL_STATE:
             if( index_ < maxNumberOfEntries[0] ){
                 if(  entryExists[0][index_] == BT_TRUE ){
                      return variableIndex[0][index_];
                 }
             }
             return variableCounter;

        case VT_ALGEBRAIC_STATE:
             if( index_ < maxNumberOfEntries[1] ){
                 if(  entryExists[1][index_] == BT_TRUE ){
                      return variableIndex[1][index_];
                 }
             }
             return variableCounter;

        case VT_CONTROL:
             if( index_ < maxNumberOfEntries[2] ){
                 if(  entryExists[2][index_] == BT_TRUE ){
                      return variableIndex[2][index_];
                 }
             }
             return variableCounter;

        case VT_INTEGER_CONTROL:
             if( index_ < maxNumberOfEntries[3] ){
                 if(  entryExists[3][index_] == BT_TRUE ){
                      return variableIndex[3][index_];
                 }
             }
             return variableCounter;

        case VT_PARAMETER:
             if( index_ < maxNumberOfEntries[4] ){
                 if(  entryExists[4][index_] == BT_TRUE ){
                      return variableIndex[4][index_];
                 }
             }
             return variableCounter;

        case VT_INTEGER_PARAMETER:
             if( index_ < maxNumberOfEntries[5] ){
                 if(  entryExists[5][index_] == BT_TRUE ){
                      return variableIndex[5][index_];
                 }
             }
             return variableCounter;

        case VT_DISTURBANCE:
             if( index_ < maxNumberOfEntries[6] ){
                 if(  entryExists[6][index_] == BT_TRUE ){
                      return variableIndex[6][index_];
                 }
             }
             return variableCounter;

        case VT_TIME:
             if( maxNumberOfEntries[7] > 0 ){
                 if(  entryExists[7][index_] == BT_TRUE ){
                      return variableIndex[7][index_];
                 }
             }
             return variableCounter;

        case VT_INTERMEDIATE_STATE:
             if( index_ < maxNumberOfEntries[8] ){
                 if(  entryExists[8][index_] == BT_TRUE ){
                      return variableIndex[8][index_];
                 }
             }
             return variableCounter;

        case VT_DDIFFERENTIAL_STATE:
             if( index_ < maxNumberOfEntries[9] ){
                 if(  entryExists[9][index_] == BT_TRUE ){
                      return variableIndex[9][index_];
                 }
             }
             return variableCounter;
             
        case VT_ONLINE_DATA:
             if( index_ < maxNumberOfEntries[10] ){
                 if(  entryExists[10][index_] == BT_TRUE ){
                      return variableIndex[10][index_];
                 }
             }
             return variableCounter;

        default:
			return -1;
    }
}



inline double SymbolicIndexList::scale( VariableType variableType_, int index_ ) const{

    switch(variableType_){

        case VT_DIFFERENTIAL_STATE:
             if( index_ < maxNumberOfEntries[0] )
                 return variableScale[0][index_];
             return 1.0;

        case VT_ALGEBRAIC_STATE:
             if( index_ < maxNumberOfEntries[1] )
                 return variableScale[1][index_];
             return 1.0;

        case VT_CONTROL:
             if( index_ < maxNumberOfEntries[2] )
                 return variableScale[2][index_];
             return 1.0;

        case VT_INTEGER_CONTROL:
             if( index_ < maxNumberOfEntries[3] )
                 return variableScale[3][index_];
             return 1.0;

        case VT_PARAMETER:
             if( index_ < maxNumberOfEntries[4] )
                 return variableScale[4][index_];
             return 1.0;

        case VT_INTEGER_PARAMETER:
             if( index_ < maxNumberOfEntries[5] )
                 return variableScale[5][index_];
             return 1.0;

        case VT_DISTURBANCE:
             if( index_ < maxNumberOfEntries[6] )
                 return variableScale[6][index_];
             return 1.0;

        case VT_TIME:
             if( maxNumberOfEntries[7] > 0 )
                 return variableScale[7][index_];
             return 1.0;

        case VT_INTERMEDIATE_STATE:
             if( index_ < maxNumberOfEntries[8] )
                 return variableScale[8][index_];
             return 1.0;

        case VT_DDIFFERENTIAL_STATE:
             if( index_ < maxNumberOfEntries[9] )
                 return variableScale[9][index_];
             return 1.0;
             
        case VT_ONLINE_DATA:
             if( index_ < maxNumberOfEntries[10] )
                 return variableScale[10][index_];
             return 1.0;

        default:
			return 1.0;

    }
}


inline int SymbolicIndexList::makeImplicit( int dim ){

    if( maxNumberOfEntries[9] != 0 ){

        return -1;
    }

    entryExists  [9] = (BooleanType*)calloc(dim,sizeof(BooleanType));
    variableIndex[9] = (int*)        calloc(dim,sizeof(int)        );
    variableScale[9] = (double*)     calloc(dim,sizeof(double)     );

    int run1;
    for( run1 = 0; run1 < dim; run1++ ){
         entryExists[9]  [run1] = BT_TRUE        ;
         variableIndex[9][run1] = variableCounter;
         variableScale[9][run1] = 1.0            ;
         variableCounter++;
    }

    maxNumberOfEntries[9] = dim;

    return variableCounter;
}



inline int SymbolicIndexList::getNumberOfVariables() const{

    return variableCounter;
}

inline int SymbolicIndexList::getNX   () const{

    return maxNumberOfEntries[0];
}

inline int SymbolicIndexList::getNXA  () const{

    return maxNumberOfEntries[1];
}

inline int SymbolicIndexList::getNDX  () const{

    return maxNumberOfEntries[9];
}

inline int SymbolicIndexList::getNU   () const{

    return maxNumberOfEntries[2];
}

inline int SymbolicIndexList::getNUI   () const{

    return maxNumberOfEntries[3];
}

inline int SymbolicIndexList::getNP   () const{

    return maxNumberOfEntries[4];
}

inline int SymbolicIndexList::getNPI   () const{

    return maxNumberOfEntries[5];
}

inline int SymbolicIndexList::getNW   () const{

    return maxNumberOfEntries[6];
}

inline int SymbolicIndexList::getNT   () const{

    return maxNumberOfEntries[7];
}

inline int SymbolicIndexList::getOD   () const{

    return maxNumberOfEntries[10];
}



CLOSE_NAMESPACE_ACADO

// end of file.
