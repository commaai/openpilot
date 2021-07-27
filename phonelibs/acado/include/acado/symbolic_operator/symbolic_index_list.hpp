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
*    \file include/acado/symbolic_operator/symbolic_index_list.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*    \date 2008
*/


#ifndef ACADO_TOOLKIT_SYMBOLIC_INDEX_LIST_HPP
#define ACADO_TOOLKIT_SYMBOLIC_INDEX_LIST_HPP


#include <acado/symbolic_operator/symbolic_operator_fwd.hpp>


BEGIN_NAMESPACE_ACADO


// Forward Declarations:
   class Operator;


/** 
 *	\brief Manages the indices of SymbolicVariables.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class SymbolicIndexList manages the indices of SymbolicVariables.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */

class SymbolicIndexList{

public:

    /** Default constructor. */
    SymbolicIndexList();

    /** Default destructor. */
    ~SymbolicIndexList();

    /** Copy constructor (deep copy). */
    SymbolicIndexList( const SymbolicIndexList &arg );

    /** Assignment Operator (deep copy). */
    SymbolicIndexList& operator=( const SymbolicIndexList &arg );



//
//  PUBLIC MEMBER FUNCTIONS:
//  ------------------------

    /** Adds a new element to the list.                           \n
     *  \return BT_TRUE  if the element is successfully added     \n
     *          BT_FALSE if the element has already been added    \n
     *          before.                                           \n
     */
    BooleanType addNewElement( VariableType variableType_  /**< The type of the
                                                             *  element to add.  */,
                               int index_                  /**< The index of the
                                                             *  element to add   */ );


    /** Determines the variableIndex of a given variable.         \n
     *  If variableIndex[variableType_][index_] already exists    \n
     *  (i.e. the index is not equal to the default value -1).    \n
     *  the returned index is equal to the variableCounter        \n
     *  and this counter will be increased by one. Otherwise, if  \n
     *  the variable has already an index this index will be      \n
     *  returned (without increasing the counter)                 \n
     *  \return The new variableIndex                             \n
     *          or -1 if the index is out of range.               \n
     */
    int determineVariableIndex( VariableType variableType_  /**< The type of
                                                             *    the variable  */,
                                 int index_                  /**< The index of the
                                                             *    variable      */,
                                 double scale               /**< The scale of the
                                                             *    variable      */ );


    /** This routine is for the communication with the C expressions. \n
     *  It determines the index list for the C expression and returns \n
     *  whether the current call is the first.                        \n
     */
    BooleanType determineCExpressionIndices( uint  dimension,  /**< The dimension of the C expression */
                                             uint  ID       ,  /**< The ID of the C expression        */
                                              int *idx         /**< The index list to be returned     */ );


    /** Sets all variable indices to -1 (default) and the variable \n
     *  counter to 0.                                              \n
     *  \return SUCCESSFUL_RETURN                                  \n
     */
    returnValue clearVariableIndexList();


//
//  INLINE FUNCTIONS:
//  -----------------

    /** Returns the index of a given variable.   \n
     *  \return The requested index      or      \n
     *          The indexCounter of the index is \n
     *          out of range.                    \n
     */
    inline int index( VariableType variableType_, int index_ ) const;


    /** Returns the scale of a given variable.   \n
     *  \return The requested scale      or      \n
     *          1.0  if index is out of range    \n
     */
    inline double scale( VariableType variableType_, int index_ ) const;


    /** Returns the variable counter.           \n
     *  \return The number of variables         \n
     */
    inline int getNumberOfVariables() const;


    /** Returns the number of differential states                 \n
     *  \return The requested number of differential states.      \n
     */
    inline int getNX    () const;

    /** Returns the number of algebraic states                    \n
     *  \return The requested number of algebraic states.         \n
     */
    inline int getNXA   () const;

    /** Returns the number of d-differential states               \n
     *  \return The requested number of d-differential states.    \n
     */
    inline int getNDX   () const;

    /** Returns the number of controls                            \n
     *  \return The requested number of controls.                 \n
     */
    inline int getNU   () const;

    /** Returns the number of integer controls                    \n
     *  \return The requested number of integer controls.         \n
     */
    inline int getNUI  () const;

    /** Returns the number of parameters                          \n
     *  \return The requested number of parameters.               \n
     */
    inline int getNP   () const;

    /** Returns the number of integer parameters                  \n
     *  \return The requested number of integer parameters.       \n
     */
    inline int getNPI  () const;

    /** Returns the number of disturbances                        \n
     *  \return The requested number of disturbances.             \n
     */
    inline int getNW  () const;

    /** Returns the number of time variables                        \n
     *  \return The requested number of time variables.             \n
     */
    inline int getNT  () const;

    /** Return number of "online data" objects. */
    inline int getOD  () const;


    inline int makeImplicit( int dim );



    /** Adds new Intermediate Operators to the stack.           \n
     *  \return SUCCESSFUL_RETURN                                 \n
     *
     */
    int addOperatorPointer( Operator* intermediateOperator
                                                           /**< The intermediate
                                                             *  OperatorPointer
                                                             *  to be added.      */,
                                      int comp_            /**< The corresponding
                                                             *  index             */ );


    /** Returns the number of new Operators            \n
     *  \return SUCCESSFUL_RETURN                        \n
     *
     */
    inline int getNumberOfOperators();


    /** Returns a copy of all intermediate Operators   \n
     *  \return SUCCESSFUL_RETURN                        \n
     *
     */
    returnValue getOperators( Operator **sub, int *comp_, int *n );


    /** Optimizes the index list for the case that a variable has \n
     *  has been substituted.                                     \n
     *  \return The new index list.                               \n
     */
    inline SymbolicIndexList* substitute( VariableType variableType_, int index_ );


//
//  PROTECTED MEMBERS:
//  ------------------

protected:

    Operator  **expression     ;   /**< Pointer to the expressions    \n
                                       *  of the intermediate states.   */

    int          *comp           ;   /**< The components of the intermediate
                                       *  states.                       */

    int numberOfOperators      ;   /**< The number of Intermediate    \n
                                       *  expressions.                  */


    BooleanType **entryExists    ;   /**< Indicates whether the entry   \n
                                       *  has already been added.       */

    int           nC             ;   /**< Maximum number of existing C  \n
                                       *  expressions                   */

    BooleanType  *cExist         ;   /**< Indicates whether a C         \n
                                       *  expression is already         \n
                                       *  regirstered.                  */

    int         **cIdx           ;   /**< Index list of existing C      \n
                                       *  expressions                   */

    uint         *cDim           ;   /**< dimesions of C expressions    */

    int         **variableIndex  ;   /**< The indices of the variables. */
    double      **variableScale  ;   /**< The scales  of the variables. */

    // INTERNAL USE / ONLY FOR MEMORY ALLOCATION:
    int       *maxNumberOfEntries;   /**< Maximum number of entries    \n
                                       *  in the lists.                */

    int           variableCounter;   /**< Counter for the variables    \n
                                       *  that are added               */



private:


};


CLOSE_NAMESPACE_ACADO



#include <acado/symbolic_operator/symbolic_index_list.ipp>


#endif
