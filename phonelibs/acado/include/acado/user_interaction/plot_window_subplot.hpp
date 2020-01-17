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
 *    \file include/acado/user_interaction/plot_window_subplot.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_PLOT_WINDOW_SUBPLOT_HPP
#define ACADO_TOOLKIT_PLOT_WINDOW_SUBPLOT_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>

BEGIN_NAMESPACE_ACADO

class Curve;

/**
 *	\brief Allows to manage sub-windows of user-specified plot windows for algorithmic outputs (for internal use).
 *
 *	\ingroup AuxiliaryFunctionality
 *	
 *  The class PlotWindowSubplot allows to manage sub-windows of user-specified 
 *	plot windows for algorithmic outputs to be plotted during runtime. 
 *	It is intended for internal use only and is used by the class PlotWindow.
 *
 *	The class stores the internal name or the symbolic expression to be plotted within
 *	one sub-plot of a figure. Moreover, output settings like title, axis labels or 
 *	ranges of the sub-plot are stored.
 *
 *	Note that PlotWindowSubplot is always stored in a basic singly-linked list within 
 *	a PlotWindow. Thus, also a pointer to the next subplot is stored. Also the actual
 *	numerical data to be plotted is stored centrally for each PlotWindow.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class PlotWindowSubplot
{
	friend class GnuplotWindow;

	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:
		/** Default constructor. */
		PlotWindowSubplot( );

		/** Constructor which takes the symbolic expression to be plotted
		 *	along with settings defining the output format of the subplot. 
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
		 */
		PlotWindowSubplot(	const Expression& _expression,
							const char* const _title = "",
							const char* const _xLabel = "",
							const char* const _yLabel = "",
							PlotMode _plotMode = PM_UNKNOWN,
							double _xRangeLowerLimit = INFTY,
							double _xRangeUpperLimit = INFTY,
							double _yRangeLowerLimit = INFTY,
							double _yRangeUpperLimit = INFTY
							);

		/** Constructor which takes symbolic expressions to be plotted
		 *	along with settings defining the output format of the subplot. 
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
		 */
		PlotWindowSubplot(	const Expression& _expressionX,
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

		/** Constructor which takes the internal name of pre-defined information to be plotted
		 *	along with settings defining the output format of the subplot. 
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
		 */
		PlotWindowSubplot(	PlotName _name,
							const char* const _title = "",
							const char* const _xLabel = "",
							const char* const _yLabel = "",
							PlotMode _plotMode = PM_UNKNOWN,
							double _xRangeLowerLimit = INFTY,
							double _xRangeUpperLimit = INFTY,
							double _yRangeLowerLimit = INFTY,
							double _yRangeUpperLimit = INFTY
							);

		/** Constructor which takes discrete data to be plotted
		 *	along with settings defining the output format of the subplot. 
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
		 */
        PlotWindowSubplot( const VariablesGrid& _plotVariable,
                           const char* const _title = "",
                           const char* const _xLabel = "",
                           const char* const _yLabel = "",
                           PlotMode _plotMode = PM_UNKNOWN,
                           double _xRangeLowerLimit = INFTY,
                           double _xRangeUpperLimit = INFTY,
                           double _yRangeLowerLimit = INFTY,
                           double _yRangeUpperLimit = INFTY,
                           BooleanType  _plot3D = BT_FALSE
						   );

        /** Constructor which takes ... */
        PlotWindowSubplot( const Curve& _curve,
                           double _xRangeLowerLimit = 0.0,
                           double _xRangeUpperLimit = 1.0,
                           const char* const _title = "",
                           const char* const _xLabel = "",
                           const char* const _yLabel = "",
                           PlotMode _plotMode = PM_UNKNOWN,
                           double _yRangeLowerLimit = INFTY,
                           double _yRangeUpperLimit = INFTY
						   );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PlotWindowSubplot(	const PlotWindowSubplot& rhs
							);

		/** Destructor. */
		~PlotWindowSubplot( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PlotWindowSubplot& operator=(	const PlotWindowSubplot& rhs
										);


		/** Sets title of the subplot.
		 *
		 *	@param[in]  _title	New title.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setTitle(	const std::string& _title
										);

		/** Sets label of x-axis of the subplot.
		 *
		 *	@param[in]  _xLabel		New label of x-axis.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setXLabel(	const std::string& _xLabel
										);

		/** Sets label of y-axis of the subplot.
		 *
		 *	@param[in]  _yLabel		New label of y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setYLabel(	const std::string& _yLabel
										);

		/** Sets plot mode of the subplot, defining how the data points are to be plotted.
		 *
		 *	@param[in]  _plotMode	New plot mode, see the PlotMode documentation for details.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setPlotMode(	PlotMode _plotMode
										);

		/** Sets plot format of the axes of the subplot.
		 *
		 *	@param[in]  _plotFormat		New plot format, see the PlotFormat documentation for details.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setPlotFormat(	PlotFormat _plotFormat
											);

		/** Sets ranges of the axes of the subplot.
		 *
		 *	@param[in] _xRangeLowerLimit	Lower limit of the x-axis.
		 *	@param[in] _xRangeUpperLimit	Upper limit of the x-axis.
		 *	@param[in] _yRangeLowerLimit	Lower limit of the y-axis.
		 *	@param[in] _yRangeUpperLimit	Upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		inline returnValue setRanges(	double _xRangeLowerLimit,
										double _xRangeUpperLimit,
										double _yRangeLowerLimit,
										double _yRangeUpperLimit
										);

		/** Adds an additional horizontal lines to be plotted within the subplot.
		 *
		 *	@param[in] _lineValue	Y-value of the additional horizontal line.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue addLine(	double _lineValue
								);

		/** Adds an additional discrete data points to be plotted within the subplot.
		 *
		 *	@param[in] _newData		Additional discrete data points.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue addData(	const VariablesGrid& _newData
								);


		/** Returns current title of the subplot.
		 *
		 *	@param[out]  _title		Current title.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue getTitle(	std::string& _title
										);

		/** Returns current label of x-axis of the subplot.
		 *
		 *	@param[out]  _xLabel		Current label of x-axis.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue getXLabel(	std::string& _xLabel
										);

		/** Returns current label of y-axis of the subplot.
		 *
		 *	@param[out]  _yLabel		Current label of y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue getYLabel(	std::string& _yLabel
										);

		/** Returns current plot mode of the subplot.
		 *
		 *  \return Current plot mode
		 */
		inline PlotMode getPlotMode( ) const;

		/** Returns current plot format of the subplot.
		 *
		 *  \return Current plot format
		 */
		inline PlotFormat getPlotFormat( ) const;

		/** Returns current ranges of the axes of the subplot.
		 *
		 *	@param[out] _xRangeLowerLimit	Current lower limit of the x-axis.
		 *	@param[out] _xRangeUpperLimit	Current upper limit of the x-axis.
		 *	@param[out] _yRangeLowerLimit	Current lower limit of the y-axis.
		 *	@param[out] _yRangeUpperLimit	Current upper limit of the y-axis.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue getRanges(	double& _xRangeLowerLimit,
										double& _xRangeUpperLimit,
										double& _yRangeLowerLimit,
										double& _yRangeUpperLimit
										) const;

		/** Returns number of additional horizontal lines of the subplot.
		 *
		 *  \return Number of additional horizontal lines
		 */
		inline uint getNumLines( ) const;

		/** Returns number of additional discrete data grids of the subplot.
		 *
		 *  \return Number of additional discrete data grids
		 */
		inline uint getNumData( ) const;


		/** Returns internal type of the subplot.
		 *
		 *  \return Internal type of the subplot
		 */
		inline SubPlotType getSubPlotType( ) const;

		/** Returns type of the variable on the x-axis of the subplot.
		 *
		 *  \return Type of the variable on the x-axis
		 */
		inline VariableType getXVariableType( ) const;

		/** Returns type of the variable on the y-axis of the subplot.
		 *
		 *  \return Type of the variable on the x-axis
		 */
		inline VariableType getYVariableType( ) const;

		/** Returns pointer to the expression on the x-axis of the subplot.
		 *
		 *  \return Pointer to the expression on the x-axis of the subplot.
		 */
		inline Expression* getXPlotExpression( ) const;

		/** Returns pointer to the expression on the y-axis of the subplot.
		 *
		 *  \return Pointer to the expression on the y-axis of the subplot.
		 */
		inline Expression* getYPlotExpression( ) const;

		/** Returns name of pre-defined information to be plotted in the subplot.
		 *
		 *  \return Name of pre-defined information to be plotted
		 */
		inline PlotName getPlotEnum( ) const;


		/** Assigns pointer to next SubPlotWindow within a PlotWindow.
		 *
		 *	@param[in]  _next	New pointer to next item.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue setNext(	PlotWindowSubplot* const _next
									);

		/** Returns pointer to next SubPlotWindow within a PlotWindow.
		 *
		 *  \return Pointer to next subplot (or NULL iff subplot is terminal element). 
		 */
		inline PlotWindowSubplot* getNext( ) const;


	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:


	//
	// DATA MEMBERS:
	//
	protected:

		Expression*    plotVariableX;				/**< Continuous variable on x-axis to be plotted. */
		Expression*    plotVariableY;				/**< Continuous variable on y-axis to be plotted. */
		VariablesGrid* plotVariablesGrid;			/**< Discrete data to be plotted (time on x-axis, data on y-axis). */
		Expression*    plotExpressionX;				/**< Continuous expression on x-axis to be plotted. */
		Expression*    plotExpressionY;				/**< Continuous expression on x-axis to be plotted. */
		PlotName       plotEnum;					/**< Pre-defined information to be plotted (on y-axis). */

		std::string title;							/**< Title of the subplot. */
		std::string xLabel;							/**< Label of x-axis of the subplot. */
		std::string yLabel;							/**< Label of y-axis of the subplot. */

		PlotMode plotMode;							/**< Plot mode defining how the data points are to be plotted. */
		PlotFormat plotFormat;						/**< Plot format of the axes of the subplot. */

		double xRangeLowerLimit;					/**< Lower limit of the x-axis. */
		double xRangeUpperLimit;					/**< Upper limit of the x-axis. */
		double yRangeLowerLimit;					/**< Lower limit of the y-axis. */
		double yRangeUpperLimit;					/**< Upper limit of the y-axis. */

        BooleanType plot3D;							/**< Flag indicating whether data is to be plotted in 3D. */

		uint nLines;								/**< Number of additional horizontal lines to be plotted. */
		double* lineValues;							/**< Values of additional horizontal lines to be plotted. */

		uint nData;									/**< Number of additional discrete data points to be plotted. */
		VariablesGrid** data;						/**< Values of additional discrete data points to be plotted. */

		PlotWindowSubplot* next;					/**< Pointer to next subplot within a PlotWindow. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/user_interaction/plot_window_subplot.ipp>


#endif	// ACADO_TOOLKIT_PLOT_WINDOW_SUBPLOT_HPP


/*
 *	end of file
 */
