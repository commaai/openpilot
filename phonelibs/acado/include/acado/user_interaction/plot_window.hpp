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
 *    \file include/acado/user_interaction/plot_window.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_PLOT_WINDOW_HPP
#define ACADO_TOOLKIT_PLOT_WINDOW_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/function/function.hpp>

#include <acado/user_interaction/plot_window_subplot.hpp>
#include <acado/user_interaction/log_record.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Allows to setup and plot user-specified plot windows for algorithmic outputs.
 *
 *	\ingroup UserDataStructures
 *	
 *  The class PlotWindow allows to setup and plot user-specified plot windows for 
 *	algorithmic outputs to be plotted during runtime.
 *
 *	A plot windows comprises arbitrarily many PlotWindowSubplots stored in a simple 
 *	singly-linked list. These subplots store the information which algorithmic 
 *	output is to be plotted as well as settings defining the style of the subplot. 
 *	Note, however, that the actual data to be plotted is stored centrally for all
 *	subplots within the PlotWindow class for memory reasons. The data required for 
 *	plotting is stored within a LogRecord.
 *
 *	Additionally, a plot window stores the PlotFrequency defining whether the window 
 *	is to be plotted at each iteration or only at the start or end, respectively.
 *
 *	Finally, it is interesting to know that PlotWindows can be setup by the user and
 *	flushed to UserInterface classes. Internally, PlotWindows are stored as basic 
 *	singly-linked within a PlotCollection.
 *
 *	\note The class PlotWindow is designed as non-abstract base class for interfacing
 *	      different plotting routines, e.g. the interface to Gnuplot.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class PlotWindow
{
	friend class PlotCollection;
	friend class Plotting;

	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:
		/** Default constructor. 
		 */
		PlotWindow( );

		/** Constructor which takes the plot frequency. 
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
		 */
		PlotWindow(	PlotFrequency _frequency
					);


		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PlotWindow(	const PlotWindow& rhs
					);

		/** Destructor. 
		 */
		virtual ~PlotWindow( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PlotWindow& operator=(	const PlotWindow& rhs
								);


		/** Clone operator returning a base class pointer to a deep copy
		 *	of the respective class instance.
		 *
		 *	\return Base class pointer to a deep copy of respective class instance
		 */
		virtual PlotWindow* clone( ) const;


		/** Initialized the plot window.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        virtual returnValue init( );


        /** Plots window into a new figure.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		virtual returnValue plot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);


        /** Plots window into existing figure, if possible.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		virtual returnValue replot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);


		/** Sets title of the subplot with given index.
		 *
		 *	@param[in]  idx		Index of subplot.
		 *	@param[in]  title_	New title.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
        virtual returnValue setTitle(	uint idx,
										const char* const title_
										);

		/** Sets label of x-axis of the subplot with given index.
		 *
		 *	@param[in]  idx			Index of subplot.
		 *	@param[in]  xLabel_		New label of x-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
        virtual returnValue setLabelX(	uint idx,
										const char* const xLabel_
										);

		/** Sets label of y-axis of the subplot with given index.
		 *
		 *	@param[in]  idx			Index of subplot.
		 *	@param[in]  yLabel_		New label of y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
        virtual returnValue setLabelY(	uint idx,
										const char* const yLabel_
										);

		/** Sets plot mode of the subplot with given index, defining how the data points are to be plotted.
		 *
		 *	@param[in]  idx			Index of subplot.
		 *	@param[in]  plotMode	New plot mode, see the PlotMode documentation for details.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
        virtual returnValue setPlotMode(	uint idx,
											PlotMode plotMode
											);

		/** Sets ranges of the axes of the subplot with given index.
		 *
		 *	@param[in] idx		Index of subplot.
		 *	@param[in] xRange1	Lower limit of the x-axis.
		 *	@param[in] xRange2	Upper limit of the x-axis.
		 *	@param[in] yRange1	Lower limit of the y-axis.
		 *	@param[in] yRange2	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue setRanges(	uint idx,
										double xRange1,
										double xRange2,
										double yRange1,
										double yRange2
										);


		/** Adds an additional horizontal lines to be plotted within the subplot with given index.
		 *
		 *	@param[in]  idx			Index of subplot.
		 *	@param[in] _lineValue	Y-value of the additional horizontal line.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		virtual returnValue addLine(	uint idx,
										double _lineValue
										);

		/** Adds an additional discrete data points to be plotted within the subplot with given index.
		 *
		 *	@param[in]  idx			Index of subplot.
		 *	@param[in] _newData		Additional discrete data points.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		virtual returnValue addData(	uint idx,
										const VariablesGrid& _newData
										);


		/** Returns a subplot of the singly-linked list with given index.
		 *
		 *	@param[in] idx	Index of desired subplot.
		 *
		 *  \return Subplot with given index. 
		 */
		inline PlotWindowSubplot& operator()(	uint idx
												);

		/** Returns a subplot of the singly-linked list with given index (const version).
		 *
		 *	@param[in] idx	Index of desired subplot.
		 *
		 *  \return Subplot with given index. 
		 */
		inline PlotWindowSubplot operator()(	uint idx
												) const;


		/** Adds a subplot to the singly-linked list.
		 *
		 *	@param[in] _subplot		Subplot to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addSubplot member function and is introduced for syntax reasons only.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_WINDOW_CORRUPTED 
		 */
		returnValue operator<<(	PlotWindowSubplot& _subplot
								);

		/** Adds an subplot of given internal name to the singly-linked list.
		 *
		 *	@param[in] _name	Internal name of subplot to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addSubplot member function and is introduced for syntax reasons only.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		returnValue operator<<(	PlotName _name
								);

		/** Adds an subplot of given internal name to the singly-linked list.
		 *
		 *	@param[in] _name	Internal name of subplot to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addSubplot member function and is introduced for syntax reasons only.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		returnValue operator<<(	const Expression& _name
								);


		/** Adds a subplot to the singly-linked list.
		 *
		 *	@param[in] _subplot		Subplot to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_WINDOW_CORRUPTED 
		 */
		returnValue addSubplot(	PlotWindowSubplot& _subplot
								);


		/** Adds a subplot plotting the given symbolic expression to the singly-linked list.
		 *	In addition, title and other properties of the subplot can be specified.
		 *
		 *	@param[in] _expression			Symbolic expression to be plotted on the y-axis.
		 *	@param[in] _title				Title of the subplot.
		 *	@param[in] _xLabel				Label of x-axis of the subplot.
		 *	@param[in] _yLabel				Label of y-axis of the subplot.
		 *	@param[in] _plotMode			Plot mode, see the PlotMode documentation for details.
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_COLLECTION_CORRUPTED, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addSubplot(	const Expression& _expression,
								const char* const _title = "",
								const char* const _xLabel = "",
								const char* const _yLabel = "",
								PlotMode _plotMode = PM_UNKNOWN,
								double _xRangeLowerLimit = INFTY,
								double _xRangeUpperLimit = INFTY,
								double _yRangeLowerLimit = INFTY,
								double _yRangeUpperLimit = INFTY
								);

		/** Adds a subplot plotting the given symbolic expressions to the singly-linked list.
		 *	In addition, title and other properties of the subplot can be specified.
		 *
		 *	@param[in] _expressionX			Symbolic expression to be plotted on the x-axis.
		 *	@param[in] _expressionY			Symbolic expression to be plotted on the y-axis.
		 *	@param[in] _title				Title of the subplot.
		 *	@param[in] _xLabel				Label of x-axis of the subplot.
		 *	@param[in] _yLabel				Label of y-axis of the subplot.
		 *	@param[in] _plotMode			Plot mode, see the PlotMode documentation for details.
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_COLLECTION_CORRUPTED, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addSubplot(	const Expression& _expressionX,
								const Expression& _expressionY,
								const char* const _title = "",
								const char* const _xLabel = "",
								const char* const _yLabel = "",
								PlotMode _plotMode = PM_UNKNOWN,
								double _xRangeLowerLimit = INFTY,
								double _xRangeUpperLimit = INFTY,
								double _yRangeLowerLimit = INFTY,
								double _yRangeUpperLimit = INFTY
								);

		/** Adds a subplot plotting the given pre-defined information to the singly-linked list.
		 *	In addition, title and other properties of the subplot can be specified.
		 *
		 *	@param[in] _name				Internal name of pre-defined information to be plotted.
		 *	@param[in] _title				Title of the subplot.
		 *	@param[in] _xLabel				Label of x-axis of the subplot.
		 *	@param[in] _yLabel				Label of y-axis of the subplot.
		 *	@param[in] _plotMode			Plot mode, see the PlotMode documentation for details.
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_COLLECTION_CORRUPTED, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addSubplot(	PlotName _name,
								const char* const _title = "",
								const char* const _xLabel = "",
								const char* const _yLabel = "",
								PlotMode _plotMode = PM_UNKNOWN,
								double _xRangeLowerLimit = INFTY,
								double _xRangeUpperLimit = INFTY,
								double _yRangeLowerLimit = INFTY,
								double _yRangeUpperLimit = INFTY
								);

		/** Adds a subplot plotting the given discrete data to the singly-linked list.
		 *	In addition, title and other properties of the subplot can be specified.
		 *
		 *	@param[in] _variable			Discrete data to be plotted on the y-axis.
		 *	@param[in] _title				Title of the subplot.
		 *	@param[in] _xLabel				Label of x-axis of the subplot.
		 *	@param[in] _yLabel				Label of y-axis of the subplot.
		 *	@param[in] _plotMode			Plot mode, see the PlotMode documentation for details.
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_COLLECTION_CORRUPTED, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addSubplot(	const VariablesGrid& _variable,
								const char* const _title = "",
								const char* const _xLabel = "",
								const char* const _yLabel = "",
								PlotMode _plotMode = PM_UNKNOWN,
								double _xRangeLowerLimit = INFTY,
								double _xRangeUpperLimit = INFTY,
								double _yRangeLowerLimit = INFTY,
								double _yRangeUpperLimit = INFTY
								);

		/** Adds a 3D subplot plotting the given discrete data to the singly-linked list.
		 *	In addition, title and other properties of the subplot can be specified.
		 *
		 *	@param[in] _variable			Discrete data to be plotted on the y-axis.
		 *	@param[in] _title				Title of the subplot.
		 *	@param[in] _xLabel				Label of x-axis of the subplot.
		 *	@param[in] _yLabel				Label of y-axis of the subplot.
		 *	@param[in] _plotMode			Plot mode, see the PlotMode documentation for details.
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_COLLECTION_CORRUPTED, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addSubplot3D(	const VariablesGrid& _variable,
									const char* const _title = "",
									const char* const _xLabel = "",
									const char* const _yLabel = "",
									PlotMode _plotMode = PM_UNKNOWN,
									double _xRangeLowerLimit = INFTY,
									double _xRangeUpperLimit = INFTY,
									double _yRangeLowerLimit = INFTY,
									double _yRangeUpperLimit = INFTY
									);

		/** Adds a subplot plotting the given curve to the singly-linked list.
		 *	In addition, title and other properties of the subplot can be specified.
		 *
		 *	@param[in] _curve				Curve to be plotted on the y-axis.
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _title				Title of the subplot.
		 *	@param[in] _xLabel				Label of x-axis of the subplot.
		 *	@param[in] _yLabel				Label of y-axis of the subplot.
		 *	@param[in] _plotMode			Plot mode, see the PlotMode documentation for details.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_COLLECTION_CORRUPTED, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addSubplot(	const Curve& _curve,
								double _xRangeLowerLimit = 0.0,
								double _xRangeUpperLimit = 1.0,
								const char* const _title = "",
								const char* const _xLabel = "",
								const char* const _yLabel = "",
								PlotMode _plotMode = PM_UNKNOWN,
								double _yRangeLowerLimit = INFTY,
								double _yRangeUpperLimit = INFTY
								);


		/** Clears all subplots from the singly-linked list.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue clearAllSubplots( );


		/** Returns current plot frequency determining at which time instants 
		 *	the window is to be plotted
		 *
		 *  \return Current plot frequency
		 */
		inline PlotFrequency getPlotFrequency( ) const;


		/** Returns number of subplots contained in the window.
		 *
		 *  \return Number of subplots
		 */
		inline uint getNumSubplots( ) const;

		/** Returns whether the window is empty (i.e. contains no subplots) or not.
		 *
		 *  \return BT_TRUE  iff window is empty, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isEmpty( ) const;


		/** Returns deep-copy of internal plotDataRecord.
		 *
		 *	@param[out]  _record	Internal plotDataRecord.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getPlotDataRecord(	LogRecord& _record
												) const;

		/** Assigns new internal plotDataRecord.
		 *
		 *	@param[in]  _record		New internal plotDataRecord.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue setPlotDataRecord(	LogRecord& _record
												);



		/** Assigns numerical values of given expression/variable
		 *	to internal plotDataRecord.
		 *
		 *	@param[in]  _name		Name of expression/variable to be plotted.
		 *	@param[in]  value		New numerical values in form of a discrete data grid.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue setPlotData(	const Expression& _name,
										VariablesGrid& value
										);

		/** Assigns numerical values of given pre-defined plotting 
		 *	information to internal plotDataRecord.
		 *
		 *	@param[in]  _name		Name of pre-defined information to be plotted.
		 *	@param[in]  value		New numerical values in form of a discrete data grid.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue setPlotData(	LogName _name,
										VariablesGrid& value
										);


		/** Returns whether window is an alias of another one.
		 *
		 *  \return BT_TRUE  iff window is an alias, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isAlias( ) const;

		/** Returns alias index of window.
		 *
		 *  \return >= 0: alias index of window, \n
		 *	        -1:   window is not an alias window
		 */
		inline int getAliasIdx( ) const;


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Assigns pointer to next PlotWindow within a PlotCollection.
		 *
		 *	@param[in]  _next	New pointer to next window.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue setNext(	PlotWindow* const _next
									);

		/** Returns pointer to next PlotWindow within a PlotCollection.
		 *
		 *  \return Pointer to next window (or NULL iff window is terminal element). 
		 */
		inline PlotWindow* getNext( ) const;


		/** Adds all log record items to internal plotDataRecord that
		 *	that are required to plot variables of given type.
		 *
		 *	@param[in]  _type	Type of variable to be plotted.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		returnValue addPlotDataItem(	VariableType _type
										);

		/** Adds all log record items to internal plotDataRecord that
		 *	that are required to plot given expression.
		 *
		 *	@param[in]  _expression	Expression to be plotted.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		returnValue addPlotDataItem(	const Expression* const _expression
										);

		/** Adds all log record items to internal plotDataRecord that
		 *	that are required to plot given pre-defined information.
		 *
		 *	@param[in]  _name	Pre-defined information to be plotted.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		returnValue addPlotDataItem(	PlotName _name
										);


		/** Returns internal name of pre-defined logging information that
		 *	corresponds to given pre-defined plotting information.
		 *
		 *	@param[in]  _name	Pre-defined information to be plotted.
		 *
		 *  \return Internal name of pre-defined logging information
		 */
		LogName convertPlotToLogName(	PlotName _name
										) const;

		/** Returns internal name of pre-defined plotting information that
		 *	corresponds to given pre-defined logging information.
		 *
		 *	@param[in]  _name	Name of pre-defined logging information.
		 *
		 *  \return Internal name of pre-defined plotting information
		 */
		PlotName convertLogToPlotName(	LogName _name
										) const;


		/** Sets-up log frequency that corresponds to given plot frequency
		 *	determining at which time instants the window is to be plotted.
		 *
		 *	@param[in]  _frequency	Plot frequency.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setupLogFrequency(	PlotFrequency _frequency = PLOT_AT_EACH_ITERATION
										);


		/** Assigns alias index of window.
		 *
		 *	@param[in] _aliasIdx	New alias index (-1 = no alias window)
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setAliasIdx(	int _aliasIdx
										);


		/** Determines type, discrete plot data grid and the higher-level 
		 *	discretization grid for a given variable to be plotted. The 
		 *	last two arguments are obtained from the internal plotDataRecord.
		 *
		 *	@param[in]  variable				Variable to be plotted.
		 *	@param[out] _type					Type of variable to be plotted.
		 *	@param[out] _dataGrid				Discrete plot data grid of variable to be plotted.
		 *	@param[out] _discretizationGrid		Higher-level discretization grid of variable to be plotted.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getVariableDataGrids(	const Expression* const variable,
											VariableType& _type,
											VariablesGrid& _dataGrid,
											Grid& _discretizationGrid
											);

		/** Determines type, discrete plot data grid and the higher-level 
		 *	discretization grid for a given expression to be plotted. The 
		 *	last two arguments are obtained from the internal plotDataRecord.
		 *
		 *	@param[in]  expression				Expression to be plotted.
		 *	@param[out] _type					Type of variable to be plotted.
		 *	@param[out] _dataGrid				Discrete plot data grid of variable to be plotted.
		 *	@param[out] _discretizationGrid		Higher-level discretization grid of variable to be plotted.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getExpressionDataGrids(	const Expression* const expression,
											VariableType& _type,
											VariablesGrid& _dataGrid,
											Grid& _discretizationGrid
											);

		/** Determines type and discrete plot data grid for a given data grid to 
		 *	be plotted. 
		 *
		 *	@param[in]  variable				Variable to be plotted.
		 *	@param[out] _type					Type of variable to be plotted.
		 *	@param[out] _dataGrid				Discrete plot data grid of variable to be plotted.
		 *	@param[out] _discretizationGrid		Higher-level discretization grid of variable to be plotted.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getDataGrids(	const VariablesGrid* const variablesGrid,
									VariableType& _type,
									VariablesGrid& _dataGrid,
									Grid& _discretizationGrid
									);


		/** Determines suitable range of the y-axis for plotting a given 
		 *	discrete data grid in an automated way.
		 *
		 *	@param[in]  dataGridY				Discrete data grid to be plotted.
		 *	@param[in]  plotFormat				Plot format of data grid to be plotted.
		 *	@param[out] lowerLimit				Suggested lower limit of the y-axis.
		 *	@param[out] upperLimit				Suggested upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getAutoScaleYLimits(	const VariablesGrid& dataGridY,
											PlotFormat plotFormat,
											double& lowerLimit,
											double& upperLimit
											) const;


		inline returnValue enableNominalControls( );
		
		inline returnValue disableNominalControls( );


		inline returnValue enableNominalParameters( );
		
		inline returnValue disableNominalParameters( );

		
		inline returnValue enableNominalOutputs( );
		
		inline returnValue disableNominalOutputs( );



    //
    // DATA MEMBERS:
    //
	protected:
		PlotWindow* next;						/**< Pointer to next windows within a PlotCollection. */
		int aliasIdx;							/**< Alias index of window (-1 = no alias window). */

		PlotFrequency frequency;				/**< Frequency determining at which time instants the window is to be plotted. */

		PlotWindowSubplot* first;				/**< Pointer to first subplot of the singly-linked list. */
		PlotWindowSubplot* last;				/**< Pointer to last subplot of the singly-linked list. */

		uint number;							/**< Total number of subplots within the singly-linked list of the window. */

		LogRecord plotDataRecord;				/**< LogRecord to store all data necessary for plotting the window. */

		BooleanType shallPlotNominalControls;	
		BooleanType shallPlotNominalParameters;	
		BooleanType shallPlotNominalOutputs;	
};


CLOSE_NAMESPACE_ACADO


#include <acado/user_interaction/plot_window.ipp>


#endif	// ACADO_TOOLKIT_PLOT_WINDOW_HPP


/*
 *	end of file
 */
