// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpTripletHelper.hpp 2380 2013-09-06 22:57:49Z ghackebeil $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPTRIPLETHELPER_HPP__
#define __IPTRIPLETHELPER_HPP__

#include "IpTypes.hpp"
#include "IpException.hpp"

namespace Ipopt
{

  DECLARE_STD_EXCEPTION(UNKNOWN_MATRIX_TYPE);
  DECLARE_STD_EXCEPTION(UNKNOWN_VECTOR_TYPE);

  /** forward declarations */
  class Matrix;
  class GenTMatrix;
  class SymTMatrix;
  class DiagMatrix;
  class IdentityMatrix;
  class ExpansionMatrix;
  class ScaledMatrix;
  class SymScaledMatrix;
  class SumMatrix;
  class SumSymMatrix;
  class ZeroMatrix;
  class ZeroSymMatrix;
  class CompoundMatrix;
  class CompoundSymMatrix;
  class TransposeMatrix;
  class ExpandedMultiVectorMatrix;
  class Vector;

  class TripletHelper
  {
  public:
    /**@name A set of recursive routines that help with the Triplet format. */
    //@{
    /** find the total number of triplet entries of a Matrix */
    static Index GetNumberEntries(const Matrix& matrix);

    /** fill the irows, jcols structure for the triplet format from the matrix */
    static void FillRowCol(Index n_entries, const Matrix& matrix, Index* iRow, Index* jCol, Index row_offset=0, Index col_offset=0);

    /** fill the values for the triplet format from the matrix */
    static void FillValues(Index n_entries, const Matrix& matrix, Number* values);

    /** fill the values from the vector into a dense double* structure */
    static void FillValuesFromVector(Index dim, const Vector& vector, Number* values);

    /** put the values from the double* back into the vector */
    static void PutValuesInVector(Index dim, const double* values, Vector& vector);
    //@}

  private:
    /** find the total number of triplet entries for the SumMatrix */
    static Index GetNumberEntries_(const SumMatrix& matrix);

    /** find the total number of triplet entries for the SumSymMatrix */
    static Index GetNumberEntries_(const SumSymMatrix& matrix);

    /** find the total number of triplet entries for the CompoundMatrix */
    static Index GetNumberEntries_(const CompoundMatrix& matrix);

    /** find the total number of triplet entries for the CompoundSymMatrix */
    static Index GetNumberEntries_(const CompoundSymMatrix& matrix);

    /** find the total number of triplet entries for the TransposeMatrix */
    static Index GetNumberEntries_(const TransposeMatrix& matrix);

    /** find the total number of triplet entries for the TransposeMatrix */
    static Index GetNumberEntries_(const ExpandedMultiVectorMatrix& matrix);

    static void FillRowCol_(Index n_entries, const GenTMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const GenTMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const SymTMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const SymTMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const DiagMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const DiagMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const IdentityMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const IdentityMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const ExpansionMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const ExpansionMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const SumMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const SumMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const SumSymMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const SumSymMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const CompoundMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const CompoundMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const CompoundSymMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const CompoundSymMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const ScaledMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const ScaledMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const SymScaledMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const SymScaledMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const TransposeMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const TransposeMatrix& matrix, Number* values);

    static void FillRowCol_(Index n_entries, const ExpandedMultiVectorMatrix& matrix, Index row_offset, Index col_offset, Index* iRow, Index* jCol);

    static void FillValues_(Index n_entries, const ExpandedMultiVectorMatrix& matrix, Number* values);

  };
} // namespace Ipopt

#endif
