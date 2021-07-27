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
 *    \file include/acado/symbolic_operator/tree_projection.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_TREE_PROJECTION_HPP
#define ACADO_TOOLKIT_TREE_PROJECTION_HPP


#include <acado/symbolic_operator/symbolic_operator_fwd.hpp>


BEGIN_NAMESPACE_ACADO


class Expression;
class ConstraintComponent;


/**
 *	\brief Implements the tree-projection operator within the family of SymbolicOperators.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class TreeProjection implements a tree projection within the 
 *	family of SymbolicOperators.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class TreeProjection : public Projection{

public:

    /** Default constructor */
    TreeProjection();

    /** Default constructor */
    TreeProjection( const std::string& name_ );

    /** Copy constructor (deep copy). */
    TreeProjection( const TreeProjection &arg );

    /** Default destructor. */
    virtual ~TreeProjection();

    /** Assignment Operator (deep copy). */
    Operator& operator=( const Operator &arg );


    /** Sets the argument (note that arg should have dimension 1). */
    virtual Operator& operator=( const Expression& arg );
    virtual Operator& operator=( const double& arg );



     /** Provides a deep copy of the expression. \n
      *  \return a clone of the expression.      \n
      */
     virtual TreeProjection* clone() const;


     /** Provides a deep copy of a tree projection. \n
      *  \return a clone of the TreeProjection.     \n
      */
     virtual TreeProjection* cloneTreeProjection() const;



     /** The function loadIndices passes an IndexList through    \n
      *  the whole expression tree. Whenever a variable gets the \n
      *  IndexList it tries to make an entry. However if a       \n
      *  variable recognices that it has already been added      \n
      *  before it will not be allowed to make a second entry.   \n
      *  Note that all variables, in paticular the intermediate  \n
      *  states, will keep in mind whether they were allowed     \n
      *  to make an entry or not. This guarantees that           \n
      *  intermediate states are never evaluated twice if they   \n
      *  occur at several knots of the tree.                     \n
      *                                                          \n
      *  THIS FUNCTION IS FOR INTERNAL USE ONLY.                 \n
      *                                                          \n
      *  PLEASE CALL THIS FUNTION AT MOST ONES FOR AN EXPRESSION \n
      *  AS A KIND OF INIT ROUTINE.                              \n
      *                                                          \n
      *  \return the name of the expression.                     \n
      */
     virtual returnValue loadIndices( SymbolicIndexList *indexList
                                                           /**< The index list to be
                                                             *  filled with entries  */ );


    /** Checks whether the expression is depending on a variable  \n
     *  \return BT_FALSE if no dependence is detected             \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isDependingOn( int           dim      ,    /**< number of directions  */
                                          VariableType *varType  ,    /**< the variable types    */
                                          int          *component,    /**< and their components  */
                                          BooleanType   *implicit_dep /**< implicit dependencies */ );




    /** Checks whether the expression is linear in                \n
     *  (or not depending on) a variable                          \n
     *  \return BT_FALSE if no linearity is                       \n
     *                detected                                    \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isLinearIn( int           dim      ,    /**< number of directions  */
                                       VariableType *varType  ,    /**< the variable types    */
                                       int          *component,    /**< and their components  */
                                       BooleanType  *implicit_dep  /**< implicit dependencies */ );



    /** Checks whether the expression is polynomial in            \n
     *  the specified variables                                   \n
     *  \return BT_FALSE if the expression is not  polynomial     \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isPolynomialIn( int           dim      ,    /**< number of directions  */
                                           VariableType *varType  ,    /**< the variable types    */
                                           int          *component,    /**< and their components  */
                                           BooleanType  *implicit_dep  /**< implicit dependencies */ );



    /** Checks whether the expression is rational in              \n
     *  the specified variables                                   \n
     *  \return BT_FALSE if the expression is not rational        \n
     *          BT_TRUE  otherwise                                \n
     *
     */
     virtual BooleanType isRationalIn( int           dim      ,    /**< number of directions  */
                                         VariableType *varType  ,    /**< the variable types    */
                                         int          *component,    /**< and their components  */
                                         BooleanType  *implicit_dep  /**< implicit dependencies */ );


     /** This function clears all static counters. Although this \n
      *  function is public it should never be used in C-code.   \n
      *  It is necessary for some Matlab-specific interfaces.    \n
      *  Please have a look into the header file                 \n
      *  include/utils/matlab_acado_utils.hpp                    \n
      *  for more details.                                       \n
      */
     returnValue clearStaticCounters();



     /** Get argument from intermediate state */
     Operator *getArgument() const;



    /** Checks whether the expression is zero or one              \n
     *  \return NE_ZERO                                           \n
     *          NE_ONE                                            \n
     *          NE_NEITHER_ONE_NOR_ZERO                           \n
     *
     */
     virtual NeutralElement isOneOrZero() const;



     /** Returns the argument or NULL if no intermediate argument available */
     virtual Operator* passArgument() const;


     virtual BooleanType isTrivial() const;

     virtual returnValue initDerivative();

//
//  PROTECTED FUNCTIONS:
//

protected:



    /** Automatic Differentiation in forward mode on the symbolic \n
     *  level. This function generates an expression for a        \n
     *  forward derivative                                        \n
     *  \return SUCCESSFUL_RETURN                                 \n
     */
     virtual Operator* ADforwardProtected( int                dim      , /**< dimension of the seed */
                                                  VariableType      *varType  , /**< the variable types    */
                                                  int               *component, /**< and their components  */
                                                  Operator  **seed     , /**< the forward seed      */
                                                  int               &nNewIS   , /**< the number of new IS  */
                                                  TreeProjection  ***newIS      /**< the new IS-pointer    */ );



    /** Automatic Differentiation in backward mode on the symbolic \n
     *  level. This function generates an expression for a         \n
     *  backward derivative                                        \n
     *  \return SUCCESSFUL_RETURN                                  \n
     */
     virtual returnValue ADbackwardProtected( int            dim      , /**< number of directions  */
                                              VariableType  *varType  , /**< the variable types    */
                                              int           *component, /**< and their components  */
                                              Operator      *seed     , /**< the backward seed     */
                                              Operator     **df       , /**< the result            */
                                              int            &nNewIS  , /**< the number of new IS  */
                                              TreeProjection ***newIS    /**< the new IS-pointer   */ );


    /** Automatic Differentiation in symmetric mode on the symbolic \n
     *  level. This function generates an expression for a          \n
     *  second order derivative.                                    \n
     *  \return SUCCESSFUL_RETURN                                   \n
     */
     virtual returnValue ADsymmetricProtected( int            dim       , /**< number of directions  */
                                               VariableType  *varType   , /**< the variable types    */
                                               int           *component , /**< and their components  */
                                               Operator      *l         , /**< the backward seed     */
                                               Operator     **S         , /**< forward seed matrix   */
                                               int            dimS      , /**< dimension of forward seed             */
                                               Operator     **dfS       , /**< first order foward result             */
                                               Operator     **ldf       , /**< first order backward result           */
                                               Operator     **H         , /**< upper trianglular part of the Hessian */
                                               int            &nNewLIS  , /**< the number of newLIS  */
                                               TreeProjection ***newLIS , /**< the new LIS-pointer   */
                                               int            &nNewSIS  , /**< the number of newSIS  */
                                               TreeProjection ***newSIS , /**< the new SIS-pointer   */
                                               int            &nNewHIS  , /**< the number of newHIS  */
                                               TreeProjection ***newHIS   /**< the new HIS-pointer   */ );
       
       
       
    /** Copy function. */
    virtual void copy( const Projection &arg );


	/** Sets the name of the variable that is used for code export.   \n
	 *  \return SUCCESSFUL_RETURN                                     \n
	 */
    virtual returnValue setVariableExportName(	const VariableType &_type,
        										const std::vector< std::string >& _name
        										);

    //
    //  PROTECTED MEMBERS:
    //
    protected:

        Operator   *argument;
        static int  count   ;
        NeutralElement    ne;
};


static TreeProjection emptyTreeProjection;


CLOSE_NAMESPACE_ACADO



#endif
