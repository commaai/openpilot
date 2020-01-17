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
 *    \file include/acado/ocp/modelData.hpp
 *    \author Rien Quirynen
 */


#ifndef ACADO_TOOLKIT_MODELDATA_HPP
#define ACADO_TOOLKIT_MODELDATA_HPP

#include <acado/function/function.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Data class for defining models and everything that is related, to be passed to the integrator.
 *
 *	\ingroup BasicDataStructures
 *
 *	TODO: Rien
 *
 *  \author Rien Quirynen
 */

class ModelData {


//
// PUBLIC MEMBER FUNCTIONS:
//
public:


    /**
     * Default constructor.
     */
	ModelData( );


	/** Assigns the model dimensions to be used by the integrator.
	 *
	 *	@param[in] _NX1		Number of differential states in linear input subsystem.
	 *	@param[in] _NX2		Number of differential states in nonlinear subsystem.
	 *	@param[in] _NX3		Number of differential states in linear output subsystem.
	 *	@param[in] _NDX		Number of differential states derivatives.
	 *	@param[in] _NDX3	Number of differential states derivatives in the linear output subsystem.
	 *	@param[in] _NXA		Number of algebraic states.
	 *	@param[in] _NXA3	Number of algebraic states in the linear output subsystem.
	 *	@param[in] _NU		Number of control inputs
	 *	@param[in] _NOD		Number of "online data" values
	 *	@param[in] _NP		Number of parameters
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue setDimensions( uint _NX1, uint _NX2, uint _NX3, uint _NDX, uint _NDX3, uint _NXA, uint _NXA3, uint _NU, uint _NOD, uint _NP );


	/** Adds an output function.
	 *
	 *  \param outputEquation_ 	  	an output function to be added
     *  \param measurements	  		The measurement grid per interval
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	uint addOutput( const OutputFcn& outputEquation_, const Grid& measurements );


	/** Adds an output function.
	 *
	 *  \param output 	  			The output function to be added.
	 *  \param diffs_output 	  	The derivatives of the output function to be added.
	 *  \param dim					The dimension of the output function.
     *  \param measurements	  		The measurement grid per interval
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	uint addOutput( const std::string& output, const std::string& diffs_output, const uint dim, const Grid& measurements );


	/** Adds an output function.
	 *
	 *  \param output 	  			The output function to be added.
	 *  \param diffs_output 	  	The derivatives of the output function to be added.
	 *  \param dim					The dimension of the output function.
     *  \param measurements	  		The measurement grid per interval
	 *  \param colInd				DVector stores the column indices of the elements for Compressed Row Storage (CRS).
	 *  \param rowPtr				DVector stores the locations that start a row for Compressed Row Storage (CRS).
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	uint addOutput( 	const std::string& output, const std::string& diffs_output, const uint dim,
						const Grid& measurements, const std::string& colInd, const std::string& rowPtr	);


	/** Returns true if there are extra outputs, specified for the integrator.
	 *
	 *  \return True if there are extra outputs, specified for the integrator.
	 */
	BooleanType hasOutputs		() const;


	/** Returns the dimension of a specific output function.
	 *
	 *  \param index	The index of the output function.
	 *
	 *  \return The dimension of a specific output function.
	 */
	uint getDimOutput( uint index ) const;


	 /** Returns the number of integration steps along the horizon.
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	 returnValue getNumSteps( DVector& _numSteps ) const;


	 /** Sets the number of integration steps along the horizon.
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	 returnValue setNumSteps( const DVector& _numSteps );


     /** Returns the output functions.
      *
      *  \return SUCCESSFUL_RETURN
      */
     returnValue getOutputExpressions( std::vector<Expression>& outputExpressions_ ) const;


     /** Returns the output grids.
      *
      *  \return SUCCESSFUL_RETURN
      */
     returnValue getOutputGrids( std::vector<Grid>& outputGrids_ ) const;


     /** Returns the dependency matrix for each output function, which is defined externally.
      *
      * \return The dependency matrix for each output function, defined externally.
      */
     std::vector<DMatrix> getOutputDependencies( ) const;


     /** Assigns Differential Equation to be used by the integrator.
      *
      *	@param[in] f		Differential equation.
      *
      *	\return SUCCESSFUL_RETURN
      */

     returnValue setModel( const DifferentialEquation& _f );


     /** Assigns a polynomial NARX model to be used by the integrator.
 	 *
 	 *	@param[in] delay		The delay for the states in the NARX model.
 	 *	@param[in] parms		The parameters defining the polynomial NARX model.
 	 *
 	 *	\return SUCCESSFUL_RETURN
 	 */
 	returnValue setNARXmodel( const uint _delay, const DMatrix& _parms );


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue setLinearInput( const DMatrix& M1_, const DMatrix& A1_, const DMatrix& B1_ );


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue setLinearOutput( const DMatrix& M3_, const DMatrix& A3_, const OutputFcn& rhs_ );


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue setNonlinearFeedback( const DMatrix& C_, const OutputFcn& feedb_ );


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue setLinearOutput( 	const DMatrix& M3_, const DMatrix& A3_,
    		 	 	 	 			const std::string& _rhs3,
    		 	 	 	 			const std::string& _diffs3 );


     /** Assigns the model to be used by the integrator.
      *
      *	@param[in] _rhs_ODE				Name of the function, evaluating the ODE right-hand side.
      *	@param[in] _diffs_rhs_ODE		Name of the function, evaluating the derivatives of the ODE right-hand side.
      *
      *	\return SUCCESSFUL_RETURN
      */

     returnValue setModel( 	const std::string& fileName,
    		 	 	 	 	const std::string& _rhs_ODE,
    		 	 	 	 	const std::string& _diffs_rhs_ODE );


     /** Returns the grid to be used by the integrator.
      *
      *	\return The grid to be used by the integrator.
      */
     returnValue getIntegrationGrid( Grid& integrationGrid_ ) const;


     /** Sets integration grid.
      *
      *	@param[in] _ocpGrid		Evaluation grid for optimal control.
      *	@param[in] numSteps		The number of integration steps along the horizon.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue setIntegrationGrid(	const Grid& _ocpGrid,
    		 	 	 	 				const uint _numSteps	);


     /** Clears any previously set integration grid.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue clearIntegrationGrid( );


     /** Returns the differential equations in the model.
      *
      *  \return SUCCESSFUL_RETURN
      */
     returnValue getModel( DifferentialEquation& _f ) const;


     /** Returns the polynomial NARX model.
      *
      *  \return SUCCESSFUL_RETURN
      */
     returnValue getNARXmodel( uint& _delay, DMatrix& _parms ) const;


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue getLinearInput( DMatrix& M1_, DMatrix& A1_, DMatrix& B1_ ) const;


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue getLinearOutput( DMatrix& M3_, DMatrix& A3_, OutputFcn& rhs_ ) const;


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue getNonlinearFeedback( DMatrix& C_, OutputFcn& feedb_ ) const;


     /** .
      *
      *	@param[in] 		.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue getLinearOutput( DMatrix& M3_, DMatrix& A3_ ) const;


     BooleanType hasEquidistantControlGrid		() const;
     BooleanType hasOutputFunctions		() const;
     BooleanType hasDifferentialEquation() const;
     BooleanType modelDimensionsSet() const;
     BooleanType exportRhs() const;
     BooleanType hasCompressedStorage() const;


     /** Returns number of differential states.
      *
      *  \return Number of differential states
      */
     uint getNX( ) const;
     uint getNX1( ) const;
     uint getNX2( ) const;
     uint getNX3( ) const;


     /** Returns number of differential state derivatives.
      *
      *  \return Number of differential state derivatives
      */
     uint getNDX( ) const;
     uint getNDX3( ) const;


     /** Returns number of algebraic states.
      *
      *  \return Number of algebraic states
      */
     uint getNXA( ) const;
     uint getNXA3( ) const;

     /** Returns number of control inputs.
      *
      *  \return Number of control inputs
      */
     uint getNU( ) const;
     returnValue setNU( const uint NU_ );

     /** Returns number of parameters.
      *
      *  \return Number of parameters
      */
     uint getNP( ) const;
     returnValue setNP( const uint NP_ );

     /** Returns number of parameters.
      *
      *  \return Number of parameters
      */
     uint getNOD( ) const;
     returnValue setNOD( const uint NOD_ );

     /** Returns number of shooting intervals.
      *
      *  \return Number of shooting intervals
      */
     uint getN( ) const;

     /** Sets the number of shooting intervals.
      *
      *  @param[in] N_		The number of shooting intervals.
      *
      *	\return SUCCESSFUL_RETURN
      */
     returnValue setN( const uint N_ );


     /** Returns the dimensions of the different output functions.
      *
      *  \return dimensions of the different output functions.
      */
     DVector getDimOutputs( ) const;


     /** Returns the number of different output functions.
      *
      *  \return the number of different output functions.
      */
     uint getNumOutputs( ) const;


     /** Returns the dimensions of the different output functions.
      *
      *  \return dimensions of the different output functions.
      */
     returnValue getDimOutputs( std::vector<uint>& dims ) const;


     /** Returns the number of measurements for the different output functions.
      *
      *  \return number of measurements for the different output functions.
      */
     DVector getNumMeas( ) const;


     const std::string getFileNameModel() const;
     const std::string getNameRhs() const;
     const std::string getNameDiffsRhs() const;
     const std::string getNameOutput() const;
     const std::string getNameDiffsOutput() const;
     returnValue getNameOutputs( std::vector<std::string>& names ) const;
     returnValue getNameDiffsOutputs( std::vector<std::string>& names ) const;


     //
    // PROTECTED FUNCTIONS:
    //
    protected:


    //
    // DATA MEMBERS:
    //
    protected:

     uint NX1;										/**< Number of differential states (defined by input system). */
     uint NX2;										/**< Number of differential states (defined by implicit system). */
     uint NX3;										/**< Number of differential states (defined by output system). */
     uint NDX;										/**< Number of differential states derivatives. */
     uint NDX3;										/**< Number of differential states derivatives in output system. */
     uint NXA;										/**< Number of algebraic states. */
     uint NXA3;										/**< Number of algebraic states in output system. */
     uint NU;										/**< Number of control inputs. */
     uint NP;										/**< Number of parameters. */
     uint NOD;										/**< Number of online data values. */
     uint N;										/**< Number of shooting intervals. */

     BooleanType export_rhs;						/**< True if the right-hand side and their derivatives should be exported too. */
     BooleanType model_dimensions_set;				/**< True if the model dimensions have been set. */
     std::string externModel;							/**< The name of the file containing the needed functions, if provided. */
     std::string rhs_name;								/**< The name of the function evaluating the ODE right-hand side, if provided. */
     std::string diffs_name;								/**< The name of the function evaluating the derivatives of the ODE right-hand side, if provided. */
     std::string rhs3_name;								/**< The name of the nonlinear function in the linear output system, if provided. */
     std::string diffs3_name;							/**< The name of the function evaluating the derivatives for the linear output system, if provided. */
     DifferentialEquation differentialEquation;  	/**< The differential equations in the model. */

     Grid integrationGrid;							/**< Integration grid. */
     DVector numSteps;								/**< The number of integration steps per shooting interval. */

     std::vector<Expression> outputExpressions;		/**< A vector with the output functions.     				*/
     std::vector<Grid> outputGrids;					/**< A separate grid for each output function.  			*/
     std::vector<uint> dim_outputs;					/**< Dimensions of the different output functions. */
     std::vector<uint> num_meas;					/**< Number of measurements for the different output functions. */
     std::vector<std::string> outputNames;				/**< A separate function name for each output. */
     std::vector<std::string> diffs_outputNames;			/**< A separate function name for evaluating the derivatives of each output. */
     std::vector<DVector> colInd_outputs;			/**< A separate DVector of column indices for each output if in CRS format. */
     std::vector<DVector> rowPtr_outputs;			/**< A separate DVector of row pointers for each output if in CRS format. */

     // ------------------------------------
     // ------------------------------------
     // 		NEW VARIABLES IN VERSION 2.0:
     // ------------------------------------
     // ------------------------------------
     DMatrix M1;
     DMatrix A1;
     DMatrix B1;

     DMatrix M3;
     DMatrix A3;
     OutputFcn rhs3;

     DMatrix C;
     OutputFcn feedb;

     // NARX model:
     uint delay;
     DMatrix parms;

};


CLOSE_NAMESPACE_ACADO



#endif  // ACADO_TOOLKIT_MODELDATA_HPP

/*
 *   end of file
 */
