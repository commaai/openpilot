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
 *    \file include/acado/code_generation/sim_export.hpp
 *    \author Rien Quirynen
 *    \date 2012
 */


#ifndef ACADO_TOOLKIT_SIM_EXPORT_HPP
#define ACADO_TOOLKIT_SIM_EXPORT_HPP

#include <acado/code_generation/export_module.hpp>
#include <acado/ocp/model_container.hpp>

BEGIN_NAMESPACE_ACADO

class IntegratorExport;

/** 
 *	\brief User-interface to automatically generate simulation algorithms for fast optimal control.
 *
 *	\ingroup UserInterfaces
 *
 *  The class SIMexport is a user-interface to automatically generate tailored
 *  simulation algorithms for fast optimal control. It takes an optimal control 
 *  problem (OCP) formulation and generates code based on given user options, 
 *  e.g specifying the integrator and the number of integration steps.
 * 	In addition to the export of such a simulation algorithm, the performance
 * 	of this integrator will be evaluated on accuracy of the results and the time
 * 	complexity. 
 *
 *	\author Rien Quirynen
 */
class SIMexport : public ExportModule, public ModelContainer
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor. 
		 *
		 *	@param[in] simIntervals		The number of simulation intervals.
		 *	@param[in] totalTime		The total simulation time.
		 */
		SIMexport( 	const uint simIntervals = 1,
					const double totalTime = 1.0 );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		SIMexport(	const SIMexport& arg
					);

		/** Destructor. 
		 */
		virtual ~SIMexport( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        SIMexport& operator=(	const SIMexport& arg
								);


		/** Exports all files of the auto-generated code into the given directory.
		 *
		 *	@param[in] dirName			Name of directory to be used to export files.
		 *	@param[in] _realString		std::string to be used to declare real variables.
		 *	@param[in] _intString		std::string to be used to declare integer variables.
		 *	@param[in] _precision		Number of digits to be used for exporting real values.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        virtual returnValue exportCode(	const std::string& dirName,
										const std::string& _realString = "real_t",
										const std::string& _intString = "int",
										int _precision = 16
										);


		/** Exports main header file for using the exported algorithm.
		 *
		 *	@param[in] _dirName			Name of directory to be used to export file.
		 *	@param[in] _fileName		Name of file to be exported.
		 *	@param[in] _realString		std::string to be used to declare real variables.
		 *	@param[in] _intString		std::string to be used to declare integer variables.
		 *	@param[in] _precision		Number of digits to be used for exporting real values.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue exportAcadoHeader(	const std::string& _dirName,
										const std::string& _fileName,
										const std::string& _realString = "real_t",
										const std::string& _intString = "int",
										int _precision = 16
										) const;


		/** Exports all files of the auto-generated code into the given directory and runs the test
		 * 	to evaluate the performance of the exported integrator.
		 *
		 *	@param[in] dirName			Name of directory to be used to export files.
		 * 	@param[in] initStates		Name of the file containing the initial values of all the states.
		 * 	@param[in] controls			Name of the file containing the control values over the OCP grid.
		 * 	@param[in] results			Name of the file in which the integration results will be written.
		 * 	@param[in] ref				Name of the file in which the reference will be written,
		 * 								to which the results of the integrator will be compared.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        virtual returnValue exportAndRun(	const std::string& dirName,
//											const std::string& initStates = std::string( "initStates.txt" ),
											const std::string& initStates,
//											const std::string& controls = std::string( "controls.txt" ),
											const std::string& controls,
											const std::string& results = std::string( "results.txt" ),
											const std::string& ref = std::string( "ref.txt" )
										);
			
			
		/** This function should be used if the user wants to provide the file containing the
		 * 	reference solution, to which the results of the integrator are compared.
		 *
		 *	@param[in] reference		Name of the file containing the reference.
		 * 	@param[in] outputReference	The names of the files containing the reference for the output results if any.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */							
		virtual returnValue setReference(	const std::string& reference, const std::vector<std::string>& outputReference = *(new std::vector<std::string>())
										);
			
			
		/** This function sets the number of integration steps performed for the timing results.
		 *
		 *	@param[in] _timingSteps		The new number of integration steps performed for the timing results.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */							
		virtual returnValue setTimingSteps( uint _timingSteps
										);
										
			
		/** This function sets a boolean if the exported simulation code should print all the details
		 * 	about the results or not.
		 *
		 *	@param[in] details		true if the exported simulation code should print all the details, otherwise false
		 *
		 *	\return SUCCESSFUL_RETURN
		 */								
		virtual returnValue printDetails( bool details );



    protected:

		/** Copies all class members from given object.
		 *
		 *	@param[in] arg		Right-hand side object.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue copy(	const SIMexport& arg
							);

		/** Frees internal dynamic memory to yield an empty function.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue clear( );


		/** Sets-up code export and initializes underlying export modules.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_OPTION, \n
		 *	        RET_INVALID_OBJECTIVE_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_ODE_FOR_CODE_EXPORT, \n
		 *	        RET_NO_DISCRETE_ODE_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_STATES_AND_CONTROLS_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_EQUIDISTANT_GRID_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_BOUNDS_FOR_CODE_EXPORT, \n
		 *	        RET_UNABLE_TO_EXPORT_CODE
		 */
		returnValue setup( );


		/** Checks whether OCP formulation is compatible with code export capabilities.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_OBJECTIVE_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_ODE_FOR_CODE_EXPORT, \n
		 *	        RET_NO_DISCRETE_ODE_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_STATES_AND_CONTROLS_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_EQUIDISTANT_GRID_FOR_CODE_EXPORT, \n
		 *	        RET_ONLY_BOUNDS_FOR_CODE_EXPORT
		 */
		returnValue checkConsistency( ) const;


		/** Collects all data declarations of the auto-generated sub-modules to given 
		 *	list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_UNABLE_TO_EXPORT_CODE
		 */
		returnValue collectDataDeclarations(	ExportStatementBlock& declarations,
												ExportStruct dataStruct = ACADO_ANY
												) const;

		/** Collects all function (forward) declarations of the auto-generated sub-modules 
		 *	to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_UNABLE_TO_EXPORT_CODE
		 */
		returnValue collectFunctionDeclarations(	ExportStatementBlock& declarations
													) const;


		/** Exports test file with template main function for using the 
		 *  exported simulation algorithm.
		 *
		 *	@param[in] _dirName			Name of directory to be used to export file.
		 *	@param[in] _fileName		Name of file to be exported.
		 *	@param[in] _resultsFile		Name of the file in which the integration results will be written.
		 * 	@param[in] _outputFiles		Names of the files in which the output results will be written.
		 * 	@param[in] TIMING			A boolean that is true when timing results are desired.
		 * 	@param[in] jumpReference	The reference factor if the reference output results are computed.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue exportTest(	const std::string& _dirName,
								const std::string& _fileName,
								const std::string& _resultsFile,
								const std::vector<std::string>& _outputFiles,
								const bool& TIMING = false,
								const uint jumpReference = 1
										) const;

		/** Exports the file evaluating the performance of the exported integrator,
		 * 	based on its results from the test and the corresponding reference results.
		 *
		 *	@param[in] _dirName			Name of directory to be used to export file.
		 *	@param[in] _fileName		Name of file to be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue exportEvaluation(	const std::string& _dirName,
										const std::string& _fileName
										) const;

		/** Exports GNU Makefile for compiling the exported MPC algorithm.
		 *
		 *	@param[in] _dirName			Name of directory to be used to export file.
		 *	@param[in] _fileName		Name of file to be exported.
		 *	@param[in] _realString		std::string to be used to declare real variables.
		 *	@param[in] _intString		std::string to be used to declare integer variables.
		 *	@param[in] _precision		Number of digits to be used for exporting real values.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue exportMakefile(	const std::string& _dirName,
									const std::string& _fileName,
									const std::string& _realString = "real_t",
									const std::string& _intString = "int",
									int _precision = 16
									) const;

		/** Compiles the exported source files and runs the corresponding test.
		 *
		 *	@param[in] _dirName			Name of directory in which the files are exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        returnValue executeTest( const std::string& _dirName );

        /** This function sets the number of calls performed for the timing results.
         *
         *	@param[in] _timingCalls		The new number of calls performed for the timing results.
         *
         *	\return SUCCESSFUL_RETURN
         */
        virtual returnValue setTimingCalls( uint _timingCalls
        									);

    protected:

        uint timingCalls;						/**< The number of calls to the exported function for the timing results. */

        double T;								/**< The total simulation time. */
		IntegratorExport*  integrator;			/**< Module for exporting a tailored integrator. */
		
		bool referenceProvided;			/**< True if the user provided a file with the reference solution. */
		bool PRINT_DETAILS;				/**< True if the user wants all the details about the results being printed. */
		
		static const uint factorRef = 10;		/**< The used factor in the number of integration steps to get the reference. */
		uint timingSteps;						/**< The number of integration steps performed for the timing results. */
		
		std::string _initStates;						/**< Name of the file containing the initial values of all the states. */
		std::string _controls;						/**< Name of the file containing the control values over the OCP grid. */
		std::string _results;						/**< Name of the file in which the integration results will be written. */
		std::string _ref;							/**< Name of the file in which the reference will be written, 
													 to which the results of the integrator will be compared. */
		std::vector<std::string> _refOutputFiles;	/**< Names of the files in which the outputs will be written for the reference. */
		std::vector<std::string> _outputFiles;		/**< Names of the files in which the outputs will be written for the integrator. */
		
};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_SIM_EXPORT_HPP

// end of file.
