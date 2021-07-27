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
 *    \file include/acado/variables_grid/matrix_variables_grid.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_MATRIX_VARIABLES_GRID_HPP
#define ACADO_TOOLKIT_MATRIX_VARIABLES_GRID_HPP

#include <acado/variables_grid/grid.hpp>

BEGIN_NAMESPACE_ACADO

const Grid emptyGrid;
const Grid trivialGrid( 1 );

class VariablesGrid;
class MatrixVariable;

/**
 *	\brief Provides a time grid consisting of matrix-valued optimization variables at each grid point.
 *
 *	\ingroup BasicDataStructures
 *	
 *  The class MatrixVariablesGrid provides a time grid consisting of 
 *	matrix-valued optimization variables at each grid point, as they 
 *	usually occur when discretizing optimal control problems.
 *
 *	The class inherits from the Grid class and stores the matrix-valued
 *	optimization variables in an re-allocatable array of MatrixVariables.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class MatrixVariablesGrid : public Grid
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor.
		 */
        MatrixVariablesGrid( );

		/** Constructor that takes the dimensions of each MatrixVariable as 
		 *	well as the grid on which they are defined. Further information
		 *	can optionally be specified.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _grid		Grid on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing name labels for each component of the variable(s).
		 *	@param[in] _units		Array containing unit labels for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 */
        MatrixVariablesGrid(	uint _nRows,
								uint _nCols,
								const Grid& _grid,
								VariableType _type = VT_UNKNOWN,
								const char** const _names = 0,
								const char** const _units = 0,
								const DVector* const _scaling = 0,
								const DVector* const _lb = 0,
								const DVector* const _ub = 0,
								const BooleanType* const _autoInit = 0
								);

		/** Constructor that takes the dimensions of each MatrixVariable as 
		 *	well as the number of grid points on which they are defined. 
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _nPoints		Number of grid points on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing name labels for each component of the variable(s).
		 *	@param[in] _units		Array containing unit labels for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 */
        MatrixVariablesGrid(	uint _nRows,
								uint _nCols,
								uint _nPoints,
								VariableType _type = VT_UNKNOWN,
								const char** const _names = 0,
								const char** const _units = 0,
								const DVector* const _scaling = 0,
								const DVector* const _lb = 0,
								const DVector* const _ub = 0,
								const BooleanType* const _autoInit = 0
								);

		/** Constructor that takes the dimensions of each MatrixVariable as 
		 *	well as the number of grid points on which they are defined. Moreover,
		 *	it takes the time of the first and the last grid point; all intermediate
		 *	grid points are setup to form a equidistant grid of time points.
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 *	@param[in] _nPoints		Number of grid points on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing name labels for each component of the variable(s).
		 *	@param[in] _units		Array containing unit labels for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 */
		MatrixVariablesGrid(	uint _nRows,
								uint _nCols,
								double _firstTime,
								double _lastTime,
								uint _nPoints,
								VariableType _type = VT_UNKNOWN,
								const char** const _names = 0,
								const char** const _units = 0,
								const DVector* const _scaling = 0,
								const DVector* const _lb = 0,
								const DVector* const _ub = 0,
								const BooleanType* const _autoInit = 0
								);

		/** Constructor that creates a variables grid on a given grid with given type.
		 *	At each grid point, the MatrixVariable is constructed from the matrix passed.
		 *
		 *	@param[in] arg			DMatrix to be assign at each point of the grid.
		 *	@param[in] _grid		Grid on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 */
        MatrixVariablesGrid(	const DMatrix& arg,
								const Grid& _grid = trivialGrid,
								VariableType _type = VT_UNKNOWN
								);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        MatrixVariablesGrid(	const MatrixVariablesGrid& rhs
								);

        /** Destructor.
		 */
        virtual ~MatrixVariablesGrid( );

        /** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        MatrixVariablesGrid& operator=(	const MatrixVariablesGrid& rhs
										);




		/** Assignment operator which reads data from a matrix. The data is interpreted 
		 *	as follows: the first entry of each row is taken as time of the grid point 
		 *	to be added, all remaining entries of each row are taken as numerical values 
		 *	of a MatrixVariable with exactly one column. In effect, a MatrixVariablesGrid
		 *	consisting of <number of columns - 1>-by-1 MatrixVariables defined on 
		 *	<number of rows> grid points is setup.
		 *	
		 *	@param[in] rhs		DMatrix to be read.
		 *
		 *	\note The file is closed at the end of routine.
		 */
		MatrixVariablesGrid& operator=(	const DMatrix& rhs
										);


        /** Returns the value of a certain component
         *  at a certain grid point.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *	@param[in] colIdx		Column index of the component to be returned.
		 *
         *  \return Value of component 'valueIdx' at grid point 'pointIdx' 
		 */
        double& operator()(	uint pointIdx,
									uint rowIdx,
									uint colIdx
									);

        /** Returns the value of a certain component
         *  at a certain grid point (const variant).
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *	@param[in] colIdx		Column index of the component to be returned.
		 *
         *  \return Value of component 'valueIdx' at grid point 'pointIdx' 
		 */
        double operator()(	uint pointIdx,
									uint rowIdx,
									uint colIdx
									) const;

        /** Returns a MatrixVariablesGrid consisting only of the given row.
		 *
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *
         *  \return MatrixVariablesGrid consisting only of the given row
         */
        MatrixVariablesGrid operator()(	const uint rowIdx
												) const;

        /** Returns a MatrixVariablesGrid consisting only of the values at 
		 *	given grid point.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
         *  \return MatrixVariablesGrid consisting only of the values at given grid point
         */
        MatrixVariablesGrid operator[](	const uint pointIdx
												) const;


		/** Adds (element-wise) two MatrixVariablesGrid into a temporary object.
		 *
		 *	@param[in] arg		Second summand.
		 *
		 *  \return Temporary object containing sum of MatrixVariablesGrids.
		 */
		MatrixVariablesGrid operator+(	const MatrixVariablesGrid& arg
												) const;

		/** Adds (element-wise) a MatrixVariablesGrid to object.
		 *
		 *	@param[in] arg		Second summand.
		 *
		 *  \return Reference to object after addition
		 */
		MatrixVariablesGrid& operator+=(	const MatrixVariablesGrid& arg
												);


		/** Subtracts (element-wise) a MatrixVariablesGrid from the object and stores
		 *  result into a temporary object.
		 *
		 *	@param[in] arg		Subtrahend.
		 *
		 *  \return Temporary object containing the difference of the MatrixVariablesGrids
		 */
		MatrixVariablesGrid operator-(	const MatrixVariablesGrid& arg
												) const;

		/** Subtracts (element-wise) a MatrixVariablesGrid from the object.
		 *
		 *	@param[in] arg		Subtrahend.
		 *
		 *  \return Reference to object after subtraction
		 */
		MatrixVariablesGrid& operator-=(	const MatrixVariablesGrid& arg
												);


		/** Initializes an empty MatrixVariablesGrid.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue init( );

		/** Initializes the MatrixVariablesGrid on a given grid with given dimensions 
		 * 	of each MatrixVariable. Further information can optionally be specified.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _grid		Grid on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing name labels for each component of the variable(s).
		 *	@param[in] _units		Array containing unit labels for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue init(	uint _nRows,
							uint _nCols,
							const Grid& _grid,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							const DVector* const _scaling = 0,
							const DVector* const _lb = 0,
							const DVector* const _ub = 0,
							const BooleanType* const _autoInit = 0
							);

		/** Initializes the MatrixVariablesGrid taking the dimensions of each 
		 *	MatrixVariable as well as the number of grid points on which they are defined. 
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _nPoints		Number of grid points on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing name labels for each component of the variable(s).
		 *	@param[in] _units		Array containing unit labels for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue init(	uint _nRows,
							uint _nCols,
							uint _nPoints,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							const DVector* const _scaling = 0,
							const DVector* const _lb = 0,
							const DVector* const _ub = 0,
							const BooleanType* const _autoInit = 0
							);

		/** Initializes the MatrixVariablesGrid taking the dimensions of each MatrixVariable 
		 *	as well as the number of grid points on which they are defined. Moreover,
		 *	it takes the time of the first and the last grid point; all intermediate
		 *	grid points are setup to form a equidistant grid of time points.
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 *	@param[in] _nPoints		Number of grid points on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing name labels for each component of the variable(s).
		 *	@param[in] _units		Array containing unit labels for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue init(	uint _nRows,
							uint _nCols,
							double _firstTime,
							double _lastTime,
							uint _nPoints,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							const DVector* const _scaling = 0,
							const DVector* const _lb = 0,
							const DVector* const _ub = 0,
							const BooleanType* const _autoInit = 0
							);

		/** Initializes the MatrixVariablesGrid on a given grid with given type.
		 *	At each grid point, the MatrixVariable is constructed from the matrix passed.
		 *
		 *	@param[in] arg			DMatrix to be assign at each point of the grid.
		 *	@param[in] _grid		Grid on which the MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue init(	const DMatrix& arg,
							const Grid& _grid = trivialGrid,
							VariableType _type = VT_UNKNOWN
							);

		/** Adds a new grid point with given matrix and time to grid.
		 *
		 *	@param[in] newMatrix	DMatrix of grid point to be added.
		 *	@param[in] newTime		Time of grid point to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue addMatrix(	const DMatrix& newMatrix,
								double newTime = -INFTY
								);

		/** Assigns new matrix to grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _value		New matrix.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setMatrix(	uint pointIdx,
								const DMatrix& _value
								) const;

		/** Assigns new matrix to all grid points.
		 *
		 *	@param[in] _value		New matrix.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setAllMatrices(	const DMatrix& _values
									);


		/** Returns matrix at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return DMatrix at grid point with given index (empty if index is out of bounds)
		 */
		DMatrix getMatrix(	uint pointIdx
							) const;

		/** Returns matrix at first grid point.
		 *
		 *  \return DMatrix at first grid point
		 */
		DMatrix getFirstMatrix( ) const;

		/** Returns matrix at last grid point.
		 *
		 *  \return DMatrix at last grid point
		 */
		DMatrix getLastMatrix( ) const;


		/** Returns total dimension of MatrixVariablesGrid, i.e. the sum
		 *	of dimensions of matrices at all grid point.
		 *
		 *  \return Total dimension of MatrixVariablesGrid
		 */
		uint getDim( ) const;


		/** Returns number of rows of matrix at first grid point.
		 *
		 *  \return Number of rows of matrix at first grid point
		 */
		uint getNumRows( ) const;

		/** Returns number of columns of matrix at first grid point.
		 *
		 *  \return Number of columns of matrix at first grid point
		 */
		uint getNumCols( ) const;

		/** Returns number of values of matrix at first grid point.
		 *
		 *  \return Number of values of matrix at first grid point
		 */
		uint getNumValues( ) const;


		/** Returns number of rows of matrix at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Number of rows of matrix at grid point with given index
		 */
		uint getNumRows(	uint pointIdx
								) const;

		/** Returns number of columns of matrix at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Number of columns of matrix at grid point with given index
		 */
		uint getNumCols(	uint pointIdx
								) const;

		/** Returns number of values of matrix at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Number of values of matrix at grid point with given index
		 */
		uint getNumValues(	uint pointIdx
									) const;


		/** Returns variable type of MatrixVariable at first grid point.
		 *
		 *  \return Variable type of MatrixVariable at first grid point
		 */
        VariableType getType( ) const;

		/** Assigns new variable type at all grid points.
		 *
		 *	@param[in] _type		Type of the variable(s).
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
        returnValue setType(	VariableType _type
									);

		/** Returns variable type of MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Variable type of MatrixVariable at grid point with given index
		 */
		VariableType getType(	uint pointIdx
										) const;

		/** Assigns new variable type to MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _type		New type of the variable(s).
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
        returnValue setType(	uint pointIdx,
									VariableType _type
									);

		/** Returns name label of given component of MatrixVariable at grid point with given index.
		 *
		 *	@param[in]  pointIdx	Index of grid point.
		 *	@param[in]  idx			Index of component.
		 *	@param[out] _name		Name label of given component at given grid point.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue getName(	uint pointIdx,
									uint idx,
									char* const _name
									) const;

		/** Assigns new name label to given component of MatrixVariable at grid point with given index.
		 *
		 *	@param[in]  pointIdx	Index of grid point.
		 *	@param[in]  idx			Index of component.
		 *	@param[in]  _name		New name label of given component at given grid point.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setName(	uint pointIdx,
									uint idx,
									const char* const _name
									);


		/** Returns current unit label of given component of MatrixVariable at grid point with given index.
		 *
		 *	@param[in]  pointIdx	Index of grid point.
		 *	@param[in]  idx			Index of component.
		 *	@param[out] _unit		Unit label of given component at given grid point.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue getUnit(	uint pointIdx,
									uint idx,
									char* const _unit
									) const;

		/** Assigns new name label to given component of MatrixVariable at grid point with given index.
		 *
		 *	@param[in]  pointIdx	Index of grid point.
		 *	@param[in]  idx			Index of component.
		 *	@param[in]  _unit		New unit label of given component at given grid point.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setUnit(	uint pointIdx,
									uint idx,
									const char* const _unit
									);

		/** Returns scaling of MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Scaling of MatrixVariable at given grid point
		 */
		DVector getScaling(	uint pointIdx
												) const;

		/** Assigns new scaling to MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _scaling		New scaling.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setScaling(	uint pointIdx,
										const DVector& _scaling
										);

		/** Returns scaling of given component of MatrixVariable at grid point 
		 *	with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] valueIdx		Index of component.
		 *
		 *  \return > 0.0: Scaling of given component of MatrixVariable at given grid point, \n
		 *	         -1.0: Index out of bounds
		 */
		double getScaling(	uint pointIdx,
									uint valueIdx
									) const;

		/** Assigns new scaling to given component of MatrixVariable at 
		 *	grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] valueIdx		Index of component.
		 *	@param[in] _scaling		New scaling.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setScaling(	uint pointIdx,
										uint valueIdx,
										double _scaling
										);

		/** Returns lower bounds of MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Lower bounds of MatrixVariable at given grid point
		 */
		DVector getLowerBounds(	uint pointIdx
													) const;

		/** Assigns new lower bounds to MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _lb			New lower bounds.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setLowerBounds(	uint pointIdx,
											const DVector& _lb
											);

		/** Returns lower bound of given component of MatrixVariable at grid point 
		 *	with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] valueIdx		Index of component.
		 *
		 *  \return < INFTY: Lower bound of given component of MatrixVariable at given grid point, \n
		 *	          INFTY: Index out of bounds
		 */
		double getLowerBound(	uint pointIdx,
										uint valueIdx
										) const;

		/** Assigns new lower bound to given component of MatrixVariable at 
		 *	grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] valueIdx		Index of component.
		 *	@param[in] _lb			New lower bound.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setLowerBound(	uint pointIdx,
											uint valueIdx,
											double _lb
											);

		/** Returns upper bounds of MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Upper bounds of MatrixVariable at given grid point
		 */
		DVector getUpperBounds(	uint pointIdx
													) const;

		/** Assigns new upper bounds to MatrixVariable at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _ub			New upper bounds.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setUpperBounds(	uint pointIdx,
											const DVector& _ub
											);

		/** Returns upper bound of given component of MatrixVariable at grid point 
		 *	with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] valueIdx		Index of component.
		 *
		 *  \return > -INFTY: Upper bound of given component of MatrixVariable at given grid point, \n
		 *	          -INFTY: Index out of bounds
		 */
		double getUpperBound(	uint pointIdx,
										uint valueIdx
										) const;

		/** Assigns new upper bound to given component of MatrixVariable at 
		 *	grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] valueIdx		Index of component.
		 *	@param[in] _ub			New upper bound.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setUpperBound(	uint pointIdx,
											uint valueIdx,
											double _ub
											);


		/** Returns whether MatrixVariable at grid point with given index 
		 *	will be automatically initialized.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return BT_TRUE  iff MatrixVariable at given grid point will be automatically initialized, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType getAutoInit(	uint pointIdx
										) const;

		/** Assigns new auto initialization flag to MatrixVariable at grid point 
		 *	with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _autoInit	New auto initialization flag.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setAutoInit(	uint pointIdx,
										BooleanType _autoInit
										);

		/** Enables auto initialization at all grid points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue disableAutoInit( );

		/** Disables auto initialization at all grid points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue enableAutoInit( );


		/** Returns whether MatrixVariablesGrid comprises (non-empty) name labels
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff MatrixVariablesGrid comprises name labels, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType hasNames( ) const;

		/** Returns whether MatrixVariablesGrid comprises (non-empty) unit labels
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff MatrixVariablesGrid comprises unit labels, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType hasUnits( ) const;

		/** Returns whether scaling is set (at at least one grid point).
		 *
		 *  \return BT_TRUE  iff scaling is set, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType hasScaling( ) const;

		/** Returns whether MatrixVariablesGrid comprises lower bounds 
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff MatrixVariablesGrid comprises lower bounds, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType hasLowerBounds( ) const;

		/** Returns whether MatrixVariablesGrid comprises upper bounds
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff MatrixVariablesGrid comprises upper bounds, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType hasUpperBounds( ) const;


		/** Returns maximum value over all matrices at all grid points.
		 *
		 *  \return Maximum value over all matrices at all grid points
		 */
		double getMax( ) const;

		/** Returns minimum value over all matrices at all grid points.
		 *
		 *  \return Minimum value over all matrices at all grid points
		 */
		double getMin( ) const;

		/** Returns mean value over all matrices at all grid points.
		 *
		 *  \return Mean value over all matrices at all grid points
		 */
		double getMean( ) const;

		/** Assigns zero to all components of all matrices at all grid points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setZero( );

		/** Assigns given value to all components of all matrices at all grid points.
		 *
		 *	@param[in] _value		New value.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setAll(	double _value
									);

		/** Appends grid point of given grid to object. A merge
		 *	method defines the way duplicate entries are handled.
		 *
		 *	@param[in] arg				Grid to append.
		 *	@param[in] _mergeMethod		Merge method, see documentation of MergeMethod for details.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue appendTimes(	const MatrixVariablesGrid& arg,
									MergeMethod _mergeMethod = MM_DUPLICATE
									);

		/** Appends values at all grid points of given grid to object.
		 *	Both grids need to be defined over identical grid points.
		 *
		 *	@param[in] arg				Grid whose values are to appended.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue appendValues(	const MatrixVariablesGrid& arg
									);

		/** Constructs the set union in time of current and given grid. A merge
		 *	method defines the way duplicate entries are handled. Moreover,
		 *	it can be specified whether an overlap in time of both grids shall
		 *	be kept or if only the entries of one of them shall be kept according
		 *	to the merge method.
		 *
		 *	@param[in] arg				Grid to append.
		 *	@param[in] _mergeMethod		Merge method, see documentation of MergeMethod for details.
		 *	@param[in] keepOverlap		Flag indicating whether overlap shall be kept.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue merge(	const MatrixVariablesGrid& arg,
							MergeMethod _mergeMethod = MM_DUPLICATE,
							BooleanType keepOverlap = BT_TRUE
							);

		/** Returns the time grid of MatrixVariablesGrid.
		 *
		 *	@param[out] grid_		Time grid.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue getGrid(	Grid& _grid
									) const;

		/** Returns (deep-copy of) time grid of MatrixVariablesGrid.
		 *
		 *	\note This routine is only introduced for user-convenience and
		 *	should not be used by developers aiming for maximum efficiency. 
		 *	Use routine getGrid() instead if efficiency is crucial.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		Grid getTimePoints( ) const;

		/** Returns the sub grid in time starting and ending at given 
		 *	indices.
		 *
		 *	@param[in] startIdx		Index of first grid point to be included in sub grid.
		 *	@param[in] endIdx		Index of last grid point to be included in sub grid.
		 *
		 *	\return Sub grid in time
		 */
        MatrixVariablesGrid getTimeSubGrid(	uint startIdx,
											uint endIdx
											) const;

		/** Returns the sub grid of values. It comprises all grid points of the 
		 *	object, but comprises at each grid point only the compenents starting and 
		 *	ending at given indices.
		 *
		 *	@param[in] startIdx		Index of first compenent to be included in sub grid.
		 *	@param[in] endIdx		Index of last compenent to be included in sub grid.
		 *
		 *	\note This function implicitly assumes that matrices at all grid points 
		 *	      have same number of components (or at least more than 'endIdx').
		 *
		 *	\return Sub grid of values
		 */
        MatrixVariablesGrid getValuesSubGrid(	uint startIdx,
												uint endIdx
												) const;

		/** Refines the grid by adding all grid points of given grid that are not
		 *	not yet included. For doing so, the given grid has to be a superset of 
		 *  the current grid. Values at newly added grid points are obtained by
		 *  the (optionally) specified interpolation mode.
		 *
		 *	@param[in] arg			Grid to be used for refinement.
		 *	@param[in] mode			Interpolation mode, see documentation of InterpolationMode.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue refineGrid(	const Grid& arg,
								InterpolationMode mode = IM_CONSTANT
								);

		/** Coarsens the grid by removing all grid points of current grid that are 
		 *	not included in given grid. For doing so, the given grid has to be a 
		 *  subset of the current grid.
		 *
		 *	@param[in] arg			Grid to be used for coarsening.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue coarsenGrid(	const Grid& arg
									);

		/** Returns a refined grid by adding all grid points of given grid that are 
		 *	not yet included. For doing so, the given grid has to be a superset of the
		 *  current grid. Values at newly added grid points are obtained by
		 *  the (optionally) specified interpolation mode.
		 *
		 *	@param[in] arg			Grid to be used for refinement.
		 *	@param[in] mode			Interpolation mode, see documentation of InterpolationMode.
		 *
		 *	\return Refined grid
		 */
		MatrixVariablesGrid getRefinedGrid(	const Grid& arg,
											InterpolationMode mode = IM_CONSTANT
											) const;

		/** Returns a coarsened grid by removing all grid points of current grid that are 
		 *	not included in given grid. For doing so, the given grid has to be a subset of 
		 *  the current grid.
		 *
		 *	@param[in] arg			Grid to be used for coarsening.
		 *
		 *	\return Coarsened grid
		 */
		MatrixVariablesGrid getCoarsenedGrid(	const Grid& arg
												) const;

		/** Shifts times at all grid points by a given offset.
		 *
		 *	@param[in] timeShift	Time offset for shifting.
		 *
		 *  \return Reference to object with shifted times
		 */
		MatrixVariablesGrid& shiftTimes(	double timeShift
											);

		/** Shifts all grid points backwards by one grid point, 
		 *	deleting the first one and doubling the value at 
		 *	last grid point.
		 *
		 *  \return Reference to object with shifted points
		 */
		MatrixVariablesGrid& shiftBackwards( DMatrix lastValue = emptyMatrix );

		/** Returns a vector with interpolated values of the MatrixVariablesGrid 
		 *	at given time. If given time lies in between two grid points, the 
		 *	value of the vector will be determined by linear interpolation between 
		 *	these grid points. If given time is smaller than the smallest time 
		 *	of the grid, the value of the first grid point will be returned. 
		 *	Analoguosly, if given time is larger than the largest time of the grid, 
		 *	the vector at the last grid point will be returned.
		 *
		 *	@param[in] time			Time for evaluation.
		 *
		 *  \return DVector with interpolated values at given time
		 */
		DVector linearInterpolation(	double time
									) const;

		/** Prints object to standard ouput stream. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] stream			Output stream for printing.
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
		returnValue print(	std::ostream& stream           = std::cout,
							const char* const name         = DEFAULT_LABEL,
							const char* const startString  = DEFAULT_START_STRING,
							const char* const endString    = DEFAULT_END_STRING,
							uint width                     = DEFAULT_WIDTH,
							uint precision                 = DEFAULT_PRECISION,
							const char* const colSeparator = DEFAULT_COL_SEPARATOR,
							const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
							) const;

		/** Prints object to file with given name. Various settings can
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
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue print(	const char* const filename,
							const char* const name         = DEFAULT_LABEL,
							const char* const startString  = DEFAULT_START_STRING,
							const char* const endString    = DEFAULT_END_STRING,
							uint width                     = DEFAULT_WIDTH,
							uint precision                 = DEFAULT_PRECISION,
							const char* const colSeparator = DEFAULT_COL_SEPARATOR,
							const char* const rowSeparator = DEFAULT_ROW_SEPARATOR
							) const;

		/** Prints object to file with given name. Various settings can
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
		returnValue print(	const char* const filename,
							const char* const name,
							PrintScheme printScheme
							) const;

		/** Prints object to given file. Various settings can
		 *	be specified defining its output format. 
		 *
		 *	@param[in] stream			Output stream for printing.
		 *	@param[in] name				Name label to be printed before the numerical values.
		 *	@param[in] printScheme		Print scheme defining the output format of the information.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue print(	std::ostream& stream,
							const char* const name,
							PrintScheme printScheme
							) const;

		/** A fuction that reads data from a file. The data is expected
		 *	to be in matrix format and is interpreted as follows: the first entry
		 *	of each row is taken as time of the grid point to be added, all
		 *	remaining entries of each row are taken as numerical values of a
		 *	MatrixVariable with exactly one column. In effect, a MatrixVariablesGrid
		 *	consisting of <number of columns - 1>-by-1 MatrixVariables defined on
		 *	<number of rows> grid points is setup. Note that all rows are expected
		 *	to have equal number of columns.
		 *
		 *	@param[in] stream An input stream to be read.
		 *
		 *	\note The routine is significantly different from the constructor that
		 *	      takes a single matrix.
		 */
		returnValue read(	std::istream& stream
							);

        /** A function that reads data from a file with given name. The data is expected
		 *	to be in matrix format and is interpreted as follows: the first entry
		 *	of each row is taken as time of the grid point to be added, all
		 *	remaining entries of each row are taken as numerical values of a
		 *	MatrixVariable with exactly one column. In effect, a MatrixVariablesGrid
		 *	consisting of <number of columns - 1>-by-1 MatrixVariables defined on
		 *	<number of rows> grid points is setup. Note that all rows are expected
		 *	to have equal number of columns.
		 *
		 *	@param[in] filename		Name of file to be read.
		 *
		 *	\note The routine is significantly different from the constructor that
		 *	      takes a single matrix.
		 */
		returnValue read(	const char* const filename
							);

		/**  Output streaming operator. */
		friend std::ostream& operator<<(	std::ostream& stream,
											const MatrixVariablesGrid& arg
											);

		/** Read a MatrixVariablesGrid from an input stream. */
		friend std::istream& operator>>(	std::istream& stream,
		        							MatrixVariablesGrid& arg
		        							);

		/** A printing function needed for plotting. */
		returnValue sprint(	std::ostream& stream
							);

    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Clears all MatrixVariables on the grid. Note that the grid itself
		 *	is not cleared.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue clearValues( );


		/** Initializes array of MatrixVariables with given information.
		 *	Note that this function assumes that the grid has already been setup.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing names (labels) for each component of the variable(s).
		 *	@param[in] _units		Array containing units for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue initMatrixVariables(	uint _nRows,
											uint _nCols,
											VariableType _type = VT_UNKNOWN,
											const char** const _names = 0,
											const char** const _units = 0,
											const DVector* const _scaling = 0,
											const DVector* const _lb = 0,
											const DVector* const _ub = 0,
											const BooleanType* const _autoInit = 0
											);

		/** Adds a new grid point with given MatrixVariable and time to grid.
		 *
		 *	@param[in] newMatrix	MatrixVariable of grid point to be added.
		 *	@param[in] newTime		Time of grid point to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue addMatrix(	const MatrixVariable& newMatrix,
								double newTime = -INFTY
								);

    //
    // DATA MEMBERS:
    //
    protected:

		/** DMatrix-valued optimization variable at all grid points. */
		MatrixVariable** values;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_MATRIX_VARIABLES_GRID_HPP

/*
 *	end of file
 */
