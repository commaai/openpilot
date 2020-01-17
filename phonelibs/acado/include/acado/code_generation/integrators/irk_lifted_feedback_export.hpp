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
 *    \file include/acado/code_generation/integrators/irk_lifted_feedback_export.hpp
 *    \author Rien Quirynen
 *    \date 2016
 */


#ifndef ACADO_TOOLKIT_LIFTED_IRK_FEEDBACK_EXPORT_HPP
#define ACADO_TOOLKIT_LIFTED_IRK_FEEDBACK_EXPORT_HPP

#include <acado/code_generation/integrators/irk_forward_export.hpp>


BEGIN_NAMESPACE_ACADO

/** 
 *  \brief Allows to export a tailored lifted implicit Runge-Kutta integrator with forward sensitivity generation for extra fast model predictive control.
 *
 *  \ingroup NumericalAlgorithms
 *
 *  The class FeedbackLiftedIRKExport allows to export a tailored lifted implicit Runge-Kutta integrator
 *  with forward sensitivity generation for extra fast model predictive control.
 *
 *  \author Rien Quirynen
 */
class FeedbackLiftedIRKExport : public ForwardIRKExport
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //

    public:

        /** Default constructor.
         *
         *  @param[in] _userInteraction     Pointer to corresponding user interface.
         *  @param[in] _commonHeaderName    Name of common header file to be included.
         */
        FeedbackLiftedIRKExport(    UserInteraction* _userInteraction = 0,
                            const std::string& _commonHeaderName = ""
                            );

        /** Copy constructor (deep copy).
         *
         *  @param[in] arg      Right-hand side object.
         */
        FeedbackLiftedIRKExport(    const FeedbackLiftedIRKExport& arg
                            );

        /** Destructor. 
         */
        virtual ~FeedbackLiftedIRKExport( );


        /** Assignment operator (deep copy).
         *
         *  @param[in] arg      Right-hand side object.
         */
        FeedbackLiftedIRKExport& operator=( const FeedbackLiftedIRKExport& arg
                                        );


        /** Initializes export of a tailored integrator.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue setup( );


        /** .
         *
         *  @param[in]      .
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue setNonlinearFeedback( const DMatrix& C, const Expression& feedb );
        

        /** Adds all data declarations of the auto-generated integrator to given list of declarations.
         *
         *  @param[in] declarations     List of declarations.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue getDataDeclarations(    ExportStatementBlock& declarations,
                                                    ExportStruct dataStruct = ACADO_ANY
                                                    ) const;


        /** Adds all function (forward) declarations of the auto-generated integrator to given list of declarations.
         *
         *  @param[in] declarations     List of declarations.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue getFunctionDeclarations(    ExportStatementBlock& declarations
                                                        ) const;



        /** Exports source code of the auto-generated integrator into the given directory.
         *
         *  @param[in] code             Code block containing the auto-generated integrator.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue getCode(    ExportStatementBlock& code
                                        );


    protected:


        /** Precompute as much as possible for the linear input system and export the resulting definitions.
         *
         *  @param[in] code         The block to which the code will be exported.
         *
         *  \return SUCCESSFUL_RETURN
         */
        returnValue setInputSystem( );
        returnValue prepareInputSystem( ExportStatementBlock& code );


        /** Exports the code needed to solve the system of collocation equations for the linear input system.
         *
         *  @param[in] block            The block to which the code will be exported.
         *  @param[in] A1               A constant matrix defining the equations of the linear input system.
         *  @param[in] B1               A constant matrix defining the equations of the linear input system.
         *  @param[in] Ah               The variable containing the internal coefficients of the RK method, multiplied with the step size.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue solveInputSystem(   ExportStatementBlock* block,
                                        const ExportIndex& index1,
                                        const ExportIndex& index2,
                                        const ExportIndex& index3,
                                        const ExportIndex& k_index,
                                        const ExportVariable& Ah );


        /** Exports the evaluation of the states at all stages.
         *
         *  @param[in] block            The block to which the code will be exported.
         *  @param[in] Ah               The matrix A of the IRK method, multiplied by the step size h.
         *  @param[in] index            The loop index, defining the stage.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual returnValue evaluateAllStatesImplicitSystem(    ExportStatementBlock* block,
                                            const ExportIndex& k_index,
                                            const ExportVariable& Ah,
                                            const ExportVariable& C,
                                            const ExportIndex& stage,
                                            const ExportIndex& i,
                                            const ExportIndex& tmp_index );


        /** Returns the largest global export variable.
         *
         *  \return SUCCESSFUL_RETURN
         */
        virtual ExportVariable getAuxVariable() const;


    protected:

        ExportVariable  rk_seed;                /**< Variable containing the forward seed. */
        ExportVariable  rk_stageValues;         /**< Variable containing the evaluated stage values. */

        ExportVariable  rk_Xprev;               /**< Variable containing the full previous state trajectory. */
        ExportVariable  rk_Uprev;               /**< Variable containing the previous control trajectory. */

        ExportVariable  rk_delta;               /**< Variable containing the update on the optimization variables. */

        ExportVariable  rk_xxx_lin;
        ExportVariable  rk_Khat_traj;
        ExportVariable  rk_Xhat_traj;

        // Static feedback function:
        ExportAcadoFunction feedb;
        DMatrix C11;
        uint NF;

        DMatrix mat1, sensMat;

        ExportVariable  rk_kTemp;
        ExportVariable  rk_dk1_tmp;
        ExportVariable  rk_dk2_tmp;

        ExportAcadoFunction sens_input;
        ExportAcadoFunction sens_fdb;
        ExportVariable  rk_sensF;

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_LIFTED_IRK_FEEDBACK_EXPORT_HPP

// end of file.
