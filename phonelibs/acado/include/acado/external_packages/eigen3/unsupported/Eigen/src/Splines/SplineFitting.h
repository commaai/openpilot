// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 20010-2011 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPLINE_FITTING_H
#define EIGEN_SPLINE_FITTING_H

#include <numeric>

#include "SplineFwd.h"

#include <Eigen/QR>

namespace Eigen
{
  /**
   * \brief Computes knot averages.
   * \ingroup Splines_Module
   *
   * The knots are computed as
   * \f{align*}
   *  u_0 & = \hdots = u_p = 0 \\
   *  u_{m-p} & = \hdots = u_{m} = 1 \\
   *  u_{j+p} & = \frac{1}{p}\sum_{i=j}^{j+p-1}\bar{u}_i \quad\quad j=1,\hdots,n-p
   * \f}
   * where \f$p\f$ is the degree and \f$m+1\f$ the number knots
   * of the desired interpolating spline.
   *
   * \param[in] parameters The input parameters. During interpolation one for each data point.
   * \param[in] degree The spline degree which is used during the interpolation.
   * \param[out] knots The output knot vector.
   *
   * \sa Les Piegl and Wayne Tiller, The NURBS book (2nd ed.), 1997, 9.2.1 Global Curve Interpolation to Point Data
   **/
  template <typename KnotVectorType>
  void KnotAveraging(const KnotVectorType& parameters, DenseIndex degree, KnotVectorType& knots)
  {
    knots.resize(parameters.size()+degree+1);      

    for (DenseIndex j=1; j<parameters.size()-degree; ++j)
      knots(j+degree) = parameters.segment(j,degree).mean();

    knots.segment(0,degree+1) = KnotVectorType::Zero(degree+1);
    knots.segment(knots.size()-degree-1,degree+1) = KnotVectorType::Ones(degree+1);
  }

  /**
   * \brief Computes chord length parameters which are required for spline interpolation.
   * \ingroup Splines_Module
   *
   * \param[in] pts The data points to which a spline should be fit.
   * \param[out] chord_lengths The resulting chord lenggth vector.
   *
   * \sa Les Piegl and Wayne Tiller, The NURBS book (2nd ed.), 1997, 9.2.1 Global Curve Interpolation to Point Data
   **/   
  template <typename PointArrayType, typename KnotVectorType>
  void ChordLengths(const PointArrayType& pts, KnotVectorType& chord_lengths)
  {
    typedef typename KnotVectorType::Scalar Scalar;

    const DenseIndex n = pts.cols();

    // 1. compute the column-wise norms
    chord_lengths.resize(pts.cols());
    chord_lengths[0] = 0;
    chord_lengths.rightCols(n-1) = (pts.array().leftCols(n-1) - pts.array().rightCols(n-1)).matrix().colwise().norm();

    // 2. compute the partial sums
    std::partial_sum(chord_lengths.data(), chord_lengths.data()+n, chord_lengths.data());

    // 3. normalize the data
    chord_lengths /= chord_lengths(n-1);
    chord_lengths(n-1) = Scalar(1);
  }

  /**
   * \brief Spline fitting methods.
   * \ingroup Splines_Module
   **/     
  template <typename SplineType>
  struct SplineFitting
  {
    typedef typename SplineType::KnotVectorType KnotVectorType;

    /**
     * \brief Fits an interpolating Spline to the given data points.
     *
     * \param pts The points for which an interpolating spline will be computed.
     * \param degree The degree of the interpolating spline.
     *
     * \returns A spline interpolating the initially provided points.
     **/
    template <typename PointArrayType>
    static SplineType Interpolate(const PointArrayType& pts, DenseIndex degree);

    /**
     * \brief Fits an interpolating Spline to the given data points.
     *
     * \param pts The points for which an interpolating spline will be computed.
     * \param degree The degree of the interpolating spline.
     * \param knot_parameters The knot parameters for the interpolation.
     *
     * \returns A spline interpolating the initially provided points.
     **/
    template <typename PointArrayType>
    static SplineType Interpolate(const PointArrayType& pts, DenseIndex degree, const KnotVectorType& knot_parameters);
  };

  template <typename SplineType>
  template <typename PointArrayType>
  SplineType SplineFitting<SplineType>::Interpolate(const PointArrayType& pts, DenseIndex degree, const KnotVectorType& knot_parameters)
  {
    typedef typename SplineType::KnotVectorType::Scalar Scalar;      
    typedef typename SplineType::ControlPointVectorType ControlPointVectorType;      

    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

    KnotVectorType knots;
    KnotAveraging(knot_parameters, degree, knots);

    DenseIndex n = pts.cols();
    MatrixType A = MatrixType::Zero(n,n);
    for (DenseIndex i=1; i<n-1; ++i)
    {
      const DenseIndex span = SplineType::Span(knot_parameters[i], degree, knots);

      // The segment call should somehow be told the spline order at compile time.
      A.row(i).segment(span-degree, degree+1) = SplineType::BasisFunctions(knot_parameters[i], degree, knots);
    }
    A(0,0) = 1.0;
    A(n-1,n-1) = 1.0;

    HouseholderQR<MatrixType> qr(A);

    // Here, we are creating a temporary due to an Eigen issue.
    ControlPointVectorType ctrls = qr.solve(MatrixType(pts.transpose())).transpose();

    return SplineType(knots, ctrls);
  }

  template <typename SplineType>
  template <typename PointArrayType>
  SplineType SplineFitting<SplineType>::Interpolate(const PointArrayType& pts, DenseIndex degree)
  {
    KnotVectorType chord_lengths; // knot parameters
    ChordLengths(pts, chord_lengths);
    return Interpolate(pts, degree, chord_lengths);
  }
}

#endif // EIGEN_SPLINE_FITTING_H
