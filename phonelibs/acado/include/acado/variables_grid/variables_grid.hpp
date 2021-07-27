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
 *    \file include/acado/variables_grid/variables_grid.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_VARIABLES_GRID_HPP
#define ACADO_TOOLKIT_VARIABLES_GRID_HPP

#include <acado/variables_grid/matrix_variables_grid.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Provides a time grid consisting of vector-valued optimization variables at each grid point.
 *
 *	\ingroup BasicDataStructures
 *	
 *  The class VariablesGrid provides a time grid consisting of vector-valued 
 *	optimization variables at each grid point, as they usually occur when 
 *	discretizing optimal control problems.
 *
 *	The class specalizes the MatrixVariablesGrid class to vectors represented internally 
 *	as matrices with exactly one column.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class VariablesGrid : public MatrixVariablesGrid
{
	friend class OptimizationAlgorithmBase;
	friend class OptimizationAlgorithm;
	friend class RealTimeAlgorithm;

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor.
		 */
        VariablesGrid( );

		/** Constructor that takes the dimension of each vector-valued MatrixVariable 
		 *	as well as the grid on which they are defined. Further information
		 *	can optionally be specified.
		 *
		 *	@param[in] _dim			Dimension of each vector.
		 *	@param[in] _grid		Grid on which the vector-valued MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing names (labels) for each component of the variable(s).
		 *	@param[in] _units		Array containing units for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 */
        VariablesGrid(	uint _dim,
						const Grid& _grid,
						VariableType _type = VT_UNKNOWN,
						const char** const _names = 0,
						const char** const _units = 0,
						const DVector* const _scaling = 0,
						const DVector* const _lb = 0,
						const DVector* const _ub = 0,
						const BooleanType* const _autoInit = 0
						);


		/** Constructor that takes the dimension of each vector-valued MatrixVariable 
		 *	as well as the number of grid points on which they are defined. 
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _dim			Dimension of each vector.
		 *	@param[in] _nPoints		Number of grid points on which the vector-valued MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing names (labels) for each component of the variable(s).
		 *	@param[in] _units		Array containing units for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 */
        VariablesGrid(	uint _dim,
						uint _nPoints,
						VariableType _type = VT_UNKNOWN,
						const char** const _names = 0,
						const char** const _units = 0,
						const DVector* const _scaling = 0,
						const DVector* const _lb = 0,
						const DVector* const _ub = 0,
						const BooleanType* const _autoInit = 0
						);

		/** Constructor that takes the dimensions of each vector-valued MatrixVariable 
		 *	as well as the number of grid points on which they are defined. Moreover,
		 *	it takes the time of the first and the last grid point; all intermediate
		 *	grid points are setup to form a equidistant grid of time points.
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _dim			Dimension of each vector.
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 *	@param[in] _nPoints		Number of grid points on which the vector-valued MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *	@param[in] _names		Array containing names (labels) for each component of the variable(s).
		 *	@param[in] _units		Array containing units for each component of the variable(s).
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable(s).
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable(s).
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable(s).
		 *	@param[in] _autoInit	Array defining if each component of the variable(s) is to be automatically initialized.
		 */
        VariablesGrid(	uint _dim,
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

		/** Constructor that creates a VariablesGrid on a given grid with given type.
		 *	At each grid point, the vector-valued MatrixVariable is constructed from the 
		 *	vector passed.
		 *
		 *	@param[in] arg			DVector to be assign at each point of the grid.
		 *	@param[in] _grid		Grid on which the vector-valued MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 */
        VariablesGrid(	const DVector& arg,
						const Grid& _grid = trivialGrid,
						VariableType _type = VT_UNKNOWN
						);

		/** Constructor which reads data from a matrix. The data is expected 
		 *	to be in matrix format and is interpreted as follows: the first entry 
		 *	of each row is taken as time of the grid point to be added, all
		 *	remaining entries of each row are taken as numerical values of a vector-valued
		 *	MatrixVariable with exactly one column. In effect, a MatrixVariablesGrid
		 *	consisting of <number of columns - 1>-by-1 MatrixVariables defined on 
		 *	<number of rows> grid points is setup. Note that all rows are expected
		 *	to have equal number of columns.
		 *	
		 *	@param[in] file		File to be read.
		 *
		 *	\note The file is closed at the end of routine.
		 */
        VariablesGrid(	const DMatrix& arg,
						VariableType _type = VT_UNKNOWN
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        VariablesGrid(	const VariablesGrid& rhs
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        VariablesGrid(	const MatrixVariablesGrid& rhs
						);


        /** Destructor.
		 */
        ~VariablesGrid( );


		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        VariablesGrid& operator=(	const VariablesGrid& rhs
									);

        /** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        VariablesGrid& operator=(	const MatrixVariablesGrid& rhs
									);

        operator DMatrix() const;

		/** Assignment operator which reads data from a matrix. The data is interpreted 
		 *	as follows: the first entry of each row is taken as time of the grid point 
		 *	to be added, all remaining entries of each row are taken as numerical values 
		 *	of a vector-valued MatrixVariable with exactly one column. In effect, a 
		 *	MatrixVariablesGrid consisting of <number of columns - 1>-by-1 MatrixVariables 
		 *	defined on <number of rows> grid points is setup.
		 *	
		 *	@param[in] rhs		DMatrix to be read.
		 *
		 *	\note The file is closed at the end of routine.
		 */
        VariablesGrid& operator=(	const DMatrix& rhs
									);


        /** Tests for equality,
         *
         *	@param[in] rhs	Object of comparison.
         *
         *  \return BT_TRUE  iff both objects are equal, \n
         *	        BT_FALSE otherwise
         */
        inline BooleanType operator==(	const VariablesGrid& arg
        								) const;

        /** Returns the value of a certain component at a certain grid point.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *
         *  \return Value of component 'rowIdx' at grid point 'pointIdx' 
		 */
        inline double& operator()(	uint pointIdx,
									uint rowIdx
									);

        /** Returns the value of a certain component at a certain grid point.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *
         *  \return Value of component 'rowIdx' at grid point 'pointIdx' 
		 */
        inline double operator()(	uint pointIdx,
									uint rowIdx
									) const;


        /** Returns a VariablesGrid consisting only of the given row.
		 *
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *
         *  \return VariablesGrid consisting only of the given row
         */
        VariablesGrid operator()(	const uint rowIdx
									) const;

        /** Returns a VariablesGrid consisting only of the values at 
		 *	given grid point.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
         *  \return VariablesGrid consisting only of the values at given grid point
         */
        VariablesGrid operator[](	const uint pointIdx
									) const;


		/** Adds (element-wise) two VariablesGrid into a temporary object.
		 *
		 *	@param[in] arg		Second summand.
		 *
		 *  \return Temporary object containing sum of VariablesGrids.
		 */
		inline VariablesGrid operator+(	const VariablesGrid& arg
										) const;

		/** Adds (element-wise) a VariablesGrid to object.
		 *
		 *	@param[in] arg		Second summand.
		 *
		 *  \return Reference to object after addition
		 */
		inline VariablesGrid& operator+=(	const VariablesGrid& arg
											);


		/** Subtracts (element-wise) a VariablesGrid from the object and stores
		 *  result into a temporary object.
		 *
		 *	@param[in] arg		Subtrahend.
		 *
		 *  \return Temporary object containing the difference of the VariablesGrids
		 */
		inline VariablesGrid operator-(	const VariablesGrid& arg
										) const;

		/** Subtracts (element-wise) a VariablesGrid from the object.
		 *
		 *	@param[in] arg		Subtrahend.
		 *
		 *  \return Reference to object after subtraction
		 */
		inline VariablesGrid& operator-=(	const VariablesGrid& arg
											);


		/** Initializes an empty VariablesGrid.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue init( );

		/** Initializes the VariablesGrid on a given grid with given dimension
		 * 	of each vector-valued MatrixVariable. Further information can optionally be specified.
		 *
		 *	@param[in] _dim			Dimension of each vector.
		 *	@param[in] _grid		Grid on which the vector-valued MatrixVariable(s) are defined.
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
		returnValue	init(	uint _dim,
							const Grid& _grid,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							const DVector* const _scaling = 0,
							const DVector* const _lb = 0,
							const DVector* const _ub = 0,
							const BooleanType* const _autoInit = 0
							);

		/** Initializes the VariablesGrid taking the dimension of each vector-valued 
		 *	MatrixVariable as well as the number of grid points on which they are defined. 
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _dim			Dimension of each vector.
		 *	@param[in] _nPoints		Number of grid points on which the vector-valued MatrixVariable(s) are defined.
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
        returnValue init(	uint _dim,
							uint _nPoints,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							const DVector* const _scaling = 0,
							const DVector* const _lb = 0,
							const DVector* const _ub = 0,
							const BooleanType* const _autoInit = 0
							);

		/** Initializes the MatrixVariablesGrid taking the dimension of each vector-valued 
		 *	MatrixVariable as well as the number of grid points on which they are defined. Moreover,
		 *	it takes the time of the first and the last grid point; all intermediate
		 *	grid points are setup to form a equidistant grid of time points.
		 *	Further information can optionally be specified.
		 *
		 *	@param[in] _dim			Dimension of each vector.
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 *	@param[in] _nPoints		Number of grid points on which the vector-valued MatrixVariable(s) are defined.
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
        returnValue init(	uint _dim,
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

		/** Initializes the VariablesGrid on a given grid with given type.
		 *	At each grid point, the vector-valued MatrixVariable is constructed from the matrix passed.
		 *
		 *	@param[in] arg			DVector to be assign at each point of the grid.
		 *	@param[in] _grid		Grid on which the vector-valued MatrixVariable(s) are defined.
		 *	@param[in] _type		Type of the variable(s).
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue init(	const DVector& arg,
							const Grid& _grid = trivialGrid,
							VariableType _type = VT_UNKNOWN
							);


		/** Adds a new grid point with given vector and time to grid.
		 *
		 *	@param[in] newVector	DVector of grid point to be added.
		 *	@param[in] newTime		Time of grid point to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue addVector(	const DVector& newVector,
								double newTime = -INFTY
								);


		/** Assigns new vector to grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _value		New vector.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *			RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue setVector(	uint pointIdx,			/**< Index of the grid point. */
								const DVector& _values	/**< New values of the sub-vector. */
								);

		/** Assigns new vector to all grid points.
		 *
		 *	@param[in] _value		New vector.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *			RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue setAllVectors(	const DVector& _values
									);


		/** Returns vector at grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return DVector at grid point with given index (empty if index is out of bounds)
		 */
		DVector getVector(	uint pointIdx
							) const;

		/** Returns vector at first grid point.
		 *
		 *  \return DVector at first grid point
		 */
		DVector getFirstVector( ) const;

		/** Returns vector at first grid point.
		 *
		 *  \return DVector at first grid point
		 */
		DVector getLastVector( ) const;


		/** Appends grid point of given grid to object. A merge
		 *	method defines the way duplicate entries are handled.
		 *
		 *	@param[in] arg				Grid to append.
		 *	@param[in] _mergeMethod		Merge method, see documentation of MergeMethod for details.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue appendTimes(	const VariablesGrid& arg,
									MergeMethod _mergeMethod = MM_DUPLICATE
									);

		/** Appends grid point of given grid to object. A merge
		 *	method defines the way duplicate entries are handled.
		 *
		 *	@param[in] arg				Grid to append in matrix form.
		 *	@param[in] _mergeMethod		Merge method, see documentation of MergeMethod for details.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue appendTimes(	const DMatrix& arg,
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
		returnValue appendValues(	const VariablesGrid& arg
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
		returnValue merge(	const VariablesGrid& arg,
							MergeMethod _mergeMethod = MM_DUPLICATE,
							BooleanType keepOverlap = BT_TRUE
							);


		/** Returns the sub grid in time starting and ending at given 
		 *	indices.
		 *
		 *	@param[in] startIdx		Index of first grid point to be included in sub grid.
		 *	@param[in] endIdx		Index of last grid point to be included in sub grid.
		 *
		 *	\return Sub grid in time
		 */
        VariablesGrid getTimeSubGrid(	uint startIdx,
										uint endIdx
										) const;

		/** Returns the sub grid in time starting and ending at given 
		 *	times.
		 *
		 *	@param[in] startTime	Time of first grid point to be included in sub grid.
		 *	@param[in] endTime		Time of last grid point to be included in sub grid.
		 *
		 *	\return Sub grid in time
		 */
		VariablesGrid getTimeSubGrid(	double startTime,
										double endTime
										) const;

		/** Returns the sub grid of values. It comprises all grid points of the 
		 *	object, but comprises at each grid point only the compenents starting and 
		 *	ending at given indices.
		 *
		 *	@param[in] startIdx		Index of first compenent to be included in sub grid.
		 *	@param[in] endIdx		Index of last compenent to be included in sub grid.
		 *
		 *	\note This function implicitly assumes that vectors at all grid points 
		 *	      have same number of components (or at least more than 'endIdx').
		 *
		 *	\return Sub grid of values
		 */
        VariablesGrid getValuesSubGrid(	uint startIdx,
										uint endIdx
										) const;


		/** Shifts times at all grid points by a given offset.
		 *
		 *	@param[in] timeShift	Time offset for shifting.
		 *
		 *  \return Reference to object with shifted times
		 */
        VariablesGrid& shiftTimes(	double timeShift
									);

		/** Shifts all grid points backwards by one grid point, 
		 *	deleting the first one and doubling the value at 
		 *	last grid point.
		 *
		 *  \return Reference to object with shifted points
		 */
		VariablesGrid& shiftBackwards( DVector lastValue = emptyVector );


		/** Returns the component-wise sum over all vectors at all grid points.
		 *
		 *	@param[out] sum		Component-wise sum over all vectors at all grid points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getSum(	DVector& sum
							) const;

		/** Returns a component-wise approximation of the integral over all vectors at all grid points.
		 *
		 *	@param[in]  mode	Specifies how the vector-values are interpolated for approximating the integral, see documentation of InterpolationMode for details.
		 *	@param[out] value	Component-wise approximation of the integral over all vectors at all grid points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getIntegral(	InterpolationMode mode,
									DVector& value
									) const;

    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

        /** Initializes the grid vector by taking average between upper and lower
         *	bound. If one of these bounds is infinity it is initialized with the
         *	bound. If both bounds are infinity it is initialized with 0.
         *	This routine is only for internal use by the OptimizationAlgorithm, 
		 *	that is a friend of this class.
		 *
		 *  \return SUCCESSFUL_RETURN
         */
        returnValue initializeFromBounds( );
};

CLOSE_NAMESPACE_ACADO

#include <acado/variables_grid/variables_grid.ipp>

BEGIN_NAMESPACE_ACADO

static       VariablesGrid emptyVariablesGrid;
static const VariablesGrid emptyConstVariablesGrid;

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_VARIABLES_GRID_HPP

/*
 *	end of file
 */
