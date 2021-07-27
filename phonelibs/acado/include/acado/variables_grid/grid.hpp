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
 *    \file include/acado/variables_grid/grid.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_GRID_HPP
#define ACADO_TOOLKIT_GRID_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>


BEGIN_NAMESPACE_ACADO



/**
 *	\brief Allows to conveniently handle (one-dimensional) grids consisting of time points.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class Grid allows to conveniently handle (one-dimensional)
 *  grids consisting of time points, as they usually occur when discretizing
 *  optimal control problems.
 *
 *	\note Time points of the grid are assumed to be ordered in increasing order.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Grid
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor. */
		Grid( );

		/** Constructor that takes the number of grid points along with 
		 *	with an optional double array containing the initialization 
		 *	of the times.
		 *
		 *	@param[in] _nPoints		Number of grid points.
		 *	@param[in] times_		Initialization of time points.
		 */
		Grid(	uint    nPoints_,
				double* times_ = 0
				);

		/** Constructor that takes the number of grid points along with 
		 *	with a initialization of the times in form of a vector.
		 *
		 *	@param[in] times		Initialization of times.
		 */
		Grid(	const DVector& times_
				);

		/** Constructor that takes the number of grid points as well as
		 *	as the time of the first and the last grid point. All intermediate
		 *	grid points are setup to form a equidistant grid of time points.
		 *
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 *	@param[in] _nPoints		Number of grid points.
		 */
		Grid(	double _firstTime,
				double _lastTime,
				uint _nPoints = 2
				);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Grid(	const Grid& rhs
				);

        /** Destructor. 
		 */
        ~Grid( );

        /** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Grid& operator=(	const Grid& rhs
							);


		/** Initializes grid with given number of grid points and given
		 *	times.
		 *
		 *	@param[in] _nPoints		Number of grid points.
		 *	@param[in] _times 		Initialization of times.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue init(	uint _nPoints = 0,
							const double* const _times = 0
							);

		/** Initializes grid with given number of grid points and given
		 *	times in form of a vector.
		 *
		 *	@param[in] times		Initialization of times.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue init(	const DVector& times_
							);

		/** Initializes grid with given number of grid points and an 
		 *	equidistant grid of time points between given time of the first 
		 *	and last grid point.
		 *
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 *	@param[in] _nPoints		Number of grid points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue init(	double _firstTime,
							double _lastTime,
							uint _nPoints = 2
							);

		/** Initializes grid with given grid.
		 *
		 *	@param[in] rhs			Grid to be taken for initialization.
		 *
		 *	\note This routine is introduced only for convenience and
         *	      is equivalent to the assignment operator.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue init(	const Grid& rhs
							);


		/** Tests for equality, i.e. if number of grid points AND
		 *  time values at all grid points are equal.
		 *
		 *	@param[in] rhs	Object of comparison.
		 *
		 *  \return BT_TRUE  iff both objects are equal, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType operator==(	const Grid& arg
										) const;

		/** Tests for non-equality.
		 *
		 *	@param[in] rhs	Object of comparison.
		 *
		 *  \return BT_TRUE iff both objects are not equal
		 *			(in the above-mentioned sense), \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType operator!=(	const Grid& arg
										) const;

		/** Tests if left-hand side grid is a strict subset of the right-hand side
		 *  one, i.e. if each the rhs grid contains all time points of the lhs
		 *  (but is not equal).
		 *
		 *	@param[in] rhs	Object of comparison.
		 *
		 *  \return BT_TRUE iff left object is a strict subset of the right one, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType operator<(	const Grid& arg
										) const;

		/** Tests if left-hand side grid is a subset of the right-hand side
		 *  one, i.e. if each the rhs grid contains all time points of the lhs.
		 *
		 *	@param[in] rhs	Object of comparison.
		 *
		 *  \return BT_TRUE iff left object is a subset of the right one, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType operator<=(	const Grid& arg
										) const;

		/** Tests if right-hand side grid is a strict subset of the left-hand side
		 *  one, i.e. if each the lhs grid contains all time points of the rhs
		 *  (but is not equal).
		 *
		 *	@param[in] rhs	Object of comparison.
		 *
		 *  \return BT_TRUE iff right object is a strict subset of the left one, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType operator>(	const Grid& arg
										) const;

		/** Tests if right-hand side grid is a subset of the left-hand side
		 *  one, i.e. if each the lhs grid contains all time points of the rhs.
		 *
		 *	@param[in] rhs	Object of comparison.
		 *
		 *  \return BT_TRUE iff right object is a subset of the left one, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType operator>=(	const Grid& arg
										) const;


		/** Constructs the set union of two grids.
		 *
		 *	@param[in] arg	Right-hand side object.
		 *
		 *  \return Set union of two grids
		 */
		Grid& operator&(	const Grid& arg
							);


		/** Constructs the set union of two grids and replaces both grids
		 *  by this union grid. Note that both the original as well as the
		 *  argument grid is changed!
		 *
		 *	@param[in] arg	Right-hand side object.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue equalizeGrids(	Grid& arg
									);


		/** Assigns next unintialized time point.
		 *
		 *	@param[in] _time	New time point.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_GRIDPOINT_SETUP_FAILED, \n
		 *	        RET_GRIDPOINT_HAS_INVALID_TIME
		 */
		returnValue setTime(	double _time
								);

		/** Assigns new time to grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *	@param[in] _time		New time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setTime(	uint pointIdx,
								double _time
								);

		/** Adds a new grid point with given time to grid.
		 *
		 *	@param[in] _time	Time point to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue addTime(	double _time
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
		returnValue merge(	const Grid& arg,
							MergeMethod _mergeMethod = MM_DUPLICATE,
							BooleanType keepOverlap = BT_TRUE
							);


		/** Returns whether the grid is empty (i.e. no grid points) or not.
		 *
		 *  \return BT_TRUE  iff grid is empty, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isEmpty( ) const;

		/** Returns number of grid points.
		 *
		 *  \return Number of grid points
		 */
		inline uint getNumPoints( ) const;

		/** Returns number of grid intervals.
		 *
		 *  \return Number of grid intervals
		 */
		inline uint getNumIntervals( ) const;


		/** Returns time of first grid point.
		 *
		 *  \return Time of first grid point
		 */
		inline double getFirstTime( ) const;

		/** Returns time of last grid point.
		 *
		 *  \return Time of last grid point
		 */
		inline double getLastTime( ) const;

		/** Returns time of grid point with given index.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return Time of grid point with given index
		 */
		inline double getTime(	uint pointIdx
								) const;


		/** Shifts times at all grid points by a given offset.
		 *
		 *	@param[in] timeShift	Time offset for shifting.
		 *
		 *  \return Reference to object with shifted times
		 */
        Grid& shiftTimes(	double timeShift
							);


		/** Returns whether the grid has equally spaced grid points or not.
		 *
		 *  \return BT_TRUE  iff grid is equidistant, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isEquidistant( ) const;

		/** Returns total interval length of grid.
		 *
		 *  \return Total interval length of grid
		 */
		inline double getIntervalLength( ) const;

		/** Returns interval length between given grid point and next one.
		 *
		 *	@param[in] pointIdx		Index of grid point at beginning of interval.
		 *
		 *  \return Interval length between given grid point and next one
		 */
		inline double getIntervalLength(	uint pointIdx
											) const;

		/** Scales times at all grid points by a given positive factor.
		 *
		 *	@param[in] scaling	Positive scaling factor.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue scaleTimes(	double scaling
								);


		/** Refines grid by a given factor by adding equally spaced 
		 *	additional time points in between existing ones.
		 *
		 *	@param[in] factor	Refinement factor.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue refineGrid(	uint factor
								);

		/** Coarsens grid by a given factor by equally leaving out
		 *	time points from the existing ones.
		 *
		 *	@param[in] factor	Coarsening factor.
		 *
		 *  \return RET_NOT_YET_IMPLEMENTED, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue coarsenGrid(	uint factor
									);


		/** Returns whether the grid contains a given time point.
		 *
		 *	@param[in] _time	Time point to be checked for existence.
		 *
		 *  \return BT_TRUE  iff grid contains given time point, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType hasTime(	double _time
								) const;


		/** Returns index of an grid point at given time, starting at
		 *	startIdx.
		 *
		 *	@param[in] _time		Time to be found.
		 *	@param[in] startIdx		Start index for searching for time point.
		 *
		 *  \return >= 0: index of grid point with given time, \n
		 *	          -1: time point does not exist
		 */
		int findTime(	double _time,
						uint startIdx = 0
						) const;

		/** Returns index of first grid point at given time, starting at
		 *	startIdx.
		 *
		 *	@param[in] _time		Time to be found.
		 *	@param[in] startIdx		Start index for searching for time point.
		 *
		 *  \return >= 0: index of first grid point with given time, \n
		 *	          -1: time point does not exist
		 */
		int findFirstTime(	double _time,
							uint startIdx = 0
							) const;

		/** Returns index of last grid point at given time, starting at
		 *	startIdx.
		 *
		 *	@param[in] _time		Time to be found.
		 *	@param[in] startIdx		Start index for searching for time point.
		 *
		 *  \return >= 0: index of last grid point with given time, \n
		 *	          -1: time point does not exist
		 */
		int findLastTime(	double _time,
							uint startIdx = 0
							) const;


		/** Returns index of grid point with greatest time smaller or equal to given time.
		 *
		 *	@param[in] _time	Time greater or equal than that of the time point to be found.
		 *
		 *  \return >= 0: index of grid point with greatest time smaller or equal to given time, \n
		 *	          -1: time point does not exist
		 */
		uint getFloorIndex(	double time
							) const;

		/** Returns index of grid point with smallest time greater or equal to given time.
		 *
		 *	@param[in] _time	Time smaller or equal than that of the time point to be found.
		 *
		 *  \return >= 0: index of grid point with smallest time greater or equal to given time, \n
		 *	          -1: time point does not exist
		 */
		uint getCeilIndex (	double time
							) const;


		/** Returns largest index of grid (note the difference to getNumPoints()).
		 *
		 *  \return Largest index of grid.
		 */
		inline uint getLastIndex (	) const;


		/** Returns whether given index is the last one of the grid.
		 *
		 *	@param[in] pointIdx		Index of grid point.
		 *
		 *  \return BT_TRUE  iff given index is the last one, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isLast(	uint pointIdx
									) const;


		/** Returns whether given time lies within the total interval of the grid.
		 *
		 *	@param[in] _time	Time point to be checked.
		 *
		 *  \return BT_TRUE  iff time within total interval, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isInInterval(	double _time
											) const;

		/** Returns whether given time lies within the interval between 
		 *	given grid point and next one.
		 *
		 *	@param[in] pointIdx		Index of grid point at beginning of interval.
		 *	@param[in] _time		Time point to be checked.
		 *
		 *  \return BT_TRUE  iff time within interval, \n
		 *	        BT_FALSE otherwise
		 */
        inline BooleanType isInInterval(	uint pointIdx,
											double _time
											) const;

		/** Returns whether given time lies within the half-open interval between 
		 *	given grid point and next one (next one not included).
		 *
		 *	@param[in] pointIdx		Index of grid point at beginning of interval.
		 *	@param[in] _time		Time point to be checked.
		 *
		 *  \return BT_TRUE  iff time within interval, \n
		 *	        BT_FALSE otherwise
		 */
        inline BooleanType isInUpperHalfOpenInterval(	uint pointIdx,
														double _time
														) const;

		/** Returns whether given time lies within the half-open interval between 
		 *	given grid point (given grid point not included) and next one.
		 *
		 *	@param[in] pointIdx		Index of grid point at beginning of interval.
		 *	@param[in] _time		Time point to be checked.
		 *
		 *  \return BT_TRUE  iff time within interval, \n
		 *	        BT_FALSE otherwise
		 */
        inline BooleanType isInLowerHalfOpenInterval(	uint pointIdx,
														double _time
														) const;


		/** Returns a sub-grid of current grid starting a given start time
		 *	end ending at given end time.
		 *
		 *	@param[in]  tStart		Start time of sub-grid.
		 *	@param[in]  tEnd		End time of sub-grid.
		 *	@param[out] _subGrid	Desired sub-grid.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue getSubGrid(	double tStart,
								double tEnd,
								Grid& _subGrid
								) const;


		/** Prints times of all grid points to screen.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
        returnValue print( ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Sets-up all times in an equidistant manner, starting at given 
		 *	start time and ending at given end time.
		 *
		 *	@param[in] _firstTime	Time of first grid point.
		 *	@param[in] _lastTime	Time of last grid point.
		 */
		returnValue setupEquidistant(	double _firstTime,
										double _lastTime
										);

		/** Returns index of next unintialized grid point.
		 *
		 *  \return >= 0: index of next unintialized grid point, \n
		 *	          -1: time point does not exist
		 */
		int findNextIndex( ) const;


    //
    // DATA MEMBERS:
    //
    protected:

		uint nPoints;					/**< Number of grid points. */
		double* times;					/**< Time values at grid points. */
};


CLOSE_NAMESPACE_ACADO

#include <acado/variables_grid/grid.ipp>

#endif  // ACADO_TOOLKIT_GRID_HPP

/*
 *	end of file
 */
