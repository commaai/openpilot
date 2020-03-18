// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Tal Hadad <tal_hd@hotmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_EULERSYSTEM_H
#define EIGEN_EULERSYSTEM_H

namespace Eigen
{
  // Forward declerations
  template <typename _Scalar, class _System>
  class EulerAngles;
  
  namespace internal
  {
    // TODO: Check if already exists on the rest API
    template <int Num, bool IsPositive = (Num > 0)>
    struct Abs
    {
      enum { value = Num };
    };
  
    template <int Num>
    struct Abs<Num, false>
    {
      enum { value = -Num };
    };

    template <int Axis>
    struct IsValidAxis
    {
      enum { value = Axis != 0 && Abs<Axis>::value <= 3 };
    };
  }
  
  #define EIGEN_EULER_ANGLES_CLASS_STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(COND)?1:-1]
  
  /** \brief Representation of a fixed signed rotation axis for EulerSystem.
    *
    * \ingroup EulerAngles_Module
    *
    * Values here represent:
    *  - The axis of the rotation: X, Y or Z.
    *  - The sign (i.e. direction of the rotation along the axis): positive(+) or negative(-)
    *
    * Therefore, this could express all the axes {+X,+Y,+Z,-X,-Y,-Z}
    *
    * For positive axis, use +EULER_{axis}, and for negative axis use -EULER_{axis}.
    */
  enum EulerAxis
  {
    EULER_X = 1, /*!< the X axis */
    EULER_Y = 2, /*!< the Y axis */
    EULER_Z = 3  /*!< the Z axis */
  };
  
  /** \class EulerSystem
    *
    * \ingroup EulerAngles_Module
    *
    * \brief Represents a fixed Euler rotation system.
    *
    * This meta-class goal is to represent the Euler system in compilation time, for EulerAngles.
    *
    * You can use this class to get two things:
    *  - Build an Euler system, and then pass it as a template parameter to EulerAngles.
    *  - Query some compile time data about an Euler system. (e.g. Whether it's tait bryan)
    *
    * Euler rotation is a set of three rotation on fixed axes. (see \ref EulerAngles)
    * This meta-class store constantly those signed axes. (see \ref EulerAxis)
    *
    * ### Types of Euler systems ###
    *
    * All and only valid 3 dimension Euler rotation over standard
    *  signed axes{+X,+Y,+Z,-X,-Y,-Z} are supported:
    *  - all axes X, Y, Z in each valid order (see below what order is valid)
    *  - rotation over the axis is supported both over the positive and negative directions.
    *  - both tait bryan and proper/classic Euler angles (i.e. the opposite).
    *
    * Since EulerSystem support both positive and negative directions,
    *  you may call this rotation distinction in other names:
    *  - _right handed_ or _left handed_
    *  - _counterclockwise_ or _clockwise_
    *
    * Notice all axed combination are valid, and would trigger a static assertion.
    * Same unsigned axes can't be neighbors, e.g. {X,X,Y} is invalid.
    * This yield two and only two classes:
    *  - _tait bryan_ - all unsigned axes are distinct, e.g. {X,Y,Z}
    *  - _proper/classic Euler angles_ - The first and the third unsigned axes is equal,
    *     and the second is different, e.g. {X,Y,X}
    *
    * ### Intrinsic vs extrinsic Euler systems ###
    *
    * Only intrinsic Euler systems are supported for simplicity.
    *  If you want to use extrinsic Euler systems,
    *   just use the equal intrinsic opposite order for axes and angles.
    *  I.e axes (A,B,C) becomes (C,B,A), and angles (a,b,c) becomes (c,b,a).
    *
    * ### Convenient user typedefs ###
    *
    * Convenient typedefs for EulerSystem exist (only for positive axes Euler systems),
    *  in a form of EulerSystem{A}{B}{C}, e.g. \ref EulerSystemXYZ.
    *
    * ### Additional reading ###
    *
    * More information about Euler angles: https://en.wikipedia.org/wiki/Euler_angles
    *
    * \tparam _AlphaAxis the first fixed EulerAxis
    *
    * \tparam _AlphaAxis the second fixed EulerAxis
    *
    * \tparam _AlphaAxis the third fixed EulerAxis
    */
  template <int _AlphaAxis, int _BetaAxis, int _GammaAxis>
  class EulerSystem
  {
    public:
    // It's defined this way and not as enum, because I think
    //  that enum is not guerantee to support negative numbers
    
    /** The first rotation axis */
    static const int AlphaAxis = _AlphaAxis;
    
    /** The second rotation axis */
    static const int BetaAxis = _BetaAxis;
    
    /** The third rotation axis */
    static const int GammaAxis = _GammaAxis;

    enum
    {
      AlphaAxisAbs = internal::Abs<AlphaAxis>::value, /*!< the first rotation axis unsigned */
      BetaAxisAbs = internal::Abs<BetaAxis>::value, /*!< the second rotation axis unsigned */
      GammaAxisAbs = internal::Abs<GammaAxis>::value, /*!< the third rotation axis unsigned */
      
      IsAlphaOpposite = (AlphaAxis < 0) ? 1 : 0, /*!< weather alpha axis is negative */
      IsBetaOpposite = (BetaAxis < 0) ? 1 : 0, /*!< weather beta axis is negative */
      IsGammaOpposite = (GammaAxis < 0) ? 1 : 0, /*!< weather gamma axis is negative */
      
      IsOdd = ((AlphaAxisAbs)%3 == (BetaAxisAbs - 1)%3) ? 0 : 1, /*!< weather the Euler system is odd */
      IsEven = IsOdd ? 0 : 1, /*!< weather the Euler system is even */

      IsTaitBryan = ((unsigned)AlphaAxisAbs != (unsigned)GammaAxisAbs) ? 1 : 0 /*!< weather the Euler system is tait bryan */
    };
    
    private:
    
    EIGEN_EULER_ANGLES_CLASS_STATIC_ASSERT(internal::IsValidAxis<AlphaAxis>::value,
      ALPHA_AXIS_IS_INVALID);
      
    EIGEN_EULER_ANGLES_CLASS_STATIC_ASSERT(internal::IsValidAxis<BetaAxis>::value,
      BETA_AXIS_IS_INVALID);
      
    EIGEN_EULER_ANGLES_CLASS_STATIC_ASSERT(internal::IsValidAxis<GammaAxis>::value,
      GAMMA_AXIS_IS_INVALID);
      
    EIGEN_EULER_ANGLES_CLASS_STATIC_ASSERT((unsigned)AlphaAxisAbs != (unsigned)BetaAxisAbs,
      ALPHA_AXIS_CANT_BE_EQUAL_TO_BETA_AXIS);
      
    EIGEN_EULER_ANGLES_CLASS_STATIC_ASSERT((unsigned)BetaAxisAbs != (unsigned)GammaAxisAbs,
      BETA_AXIS_CANT_BE_EQUAL_TO_GAMMA_AXIS);

    enum
    {
      // I, J, K are the pivot indexes permutation for the rotation matrix, that match this Euler system. 
      // They are used in this class converters.
      // They are always different from each other, and their possible values are: 0, 1, or 2.
      I = AlphaAxisAbs - 1,
      J = (AlphaAxisAbs - 1 + 1 + IsOdd)%3,
      K = (AlphaAxisAbs - 1 + 2 - IsOdd)%3
    };
    
    // TODO: Get @mat parameter in form that avoids double evaluation.
    template <typename Derived>
    static void CalcEulerAngles_imp(Matrix<typename MatrixBase<Derived>::Scalar, 3, 1>& res, const MatrixBase<Derived>& mat, internal::true_type /*isTaitBryan*/)
    {
      using std::atan2;
      using std::sin;
      using std::cos;
      
      typedef typename Derived::Scalar Scalar;
      typedef Matrix<Scalar,2,1> Vector2;
      
      res[0] = atan2(mat(J,K), mat(K,K));
      Scalar c2 = Vector2(mat(I,I), mat(I,J)).norm();
      if((IsOdd && res[0]<Scalar(0)) || ((!IsOdd) && res[0]>Scalar(0))) {
        if(res[0] > Scalar(0)) {
          res[0] -= Scalar(EIGEN_PI);
        }
        else {
          res[0] += Scalar(EIGEN_PI);
        }
        res[1] = atan2(-mat(I,K), -c2);
      }
      else
        res[1] = atan2(-mat(I,K), c2);
      Scalar s1 = sin(res[0]);
      Scalar c1 = cos(res[0]);
      res[2] = atan2(s1*mat(K,I)-c1*mat(J,I), c1*mat(J,J) - s1 * mat(K,J));
    }

    template <typename Derived>
    static void CalcEulerAngles_imp(Matrix<typename MatrixBase<Derived>::Scalar,3,1>& res, const MatrixBase<Derived>& mat, internal::false_type /*isTaitBryan*/)
    {
      using std::atan2;
      using std::sin;
      using std::cos;

      typedef typename Derived::Scalar Scalar;
      typedef Matrix<Scalar,2,1> Vector2;
      
      res[0] = atan2(mat(J,I), mat(K,I));
      if((IsOdd && res[0]<Scalar(0)) || ((!IsOdd) && res[0]>Scalar(0)))
      {
        if(res[0] > Scalar(0)) {
          res[0] -= Scalar(EIGEN_PI);
        }
        else {
          res[0] += Scalar(EIGEN_PI);
        }
        Scalar s2 = Vector2(mat(J,I), mat(K,I)).norm();
        res[1] = -atan2(s2, mat(I,I));
      }
      else
      {
        Scalar s2 = Vector2(mat(J,I), mat(K,I)).norm();
        res[1] = atan2(s2, mat(I,I));
      }

      // With a=(0,1,0), we have i=0; j=1; k=2, and after computing the first two angles,
      // we can compute their respective rotation, and apply its inverse to M. Since the result must
      // be a rotation around x, we have:
      //
      //  c2  s1.s2 c1.s2                   1  0   0 
      //  0   c1    -s1       *    M    =   0  c3  s3
      //  -s2 s1.c2 c1.c2                   0 -s3  c3
      //
      //  Thus:  m11.c1 - m21.s1 = c3  &   m12.c1 - m22.s1 = s3

      Scalar s1 = sin(res[0]);
      Scalar c1 = cos(res[0]);
      res[2] = atan2(c1*mat(J,K)-s1*mat(K,K), c1*mat(J,J) - s1 * mat(K,J));
    }
    
    template<typename Scalar>
    static void CalcEulerAngles(
      EulerAngles<Scalar, EulerSystem>& res,
      const typename EulerAngles<Scalar, EulerSystem>::Matrix3& mat)
    {
      CalcEulerAngles(res, mat, false, false, false);
    }
    
    template<
      bool PositiveRangeAlpha,
      bool PositiveRangeBeta,
      bool PositiveRangeGamma,
      typename Scalar>
    static void CalcEulerAngles(
      EulerAngles<Scalar, EulerSystem>& res,
      const typename EulerAngles<Scalar, EulerSystem>::Matrix3& mat)
    {
      CalcEulerAngles(res, mat, PositiveRangeAlpha, PositiveRangeBeta, PositiveRangeGamma);
    }
    
    template<typename Scalar>
    static void CalcEulerAngles(
      EulerAngles<Scalar, EulerSystem>& res,
      const typename EulerAngles<Scalar, EulerSystem>::Matrix3& mat,
      bool PositiveRangeAlpha,
      bool PositiveRangeBeta,
      bool PositiveRangeGamma)
    {
      CalcEulerAngles_imp(
        res.angles(), mat,
        typename internal::conditional<IsTaitBryan, internal::true_type, internal::false_type>::type());

      if (IsAlphaOpposite == IsOdd)
        res.alpha() = -res.alpha();
        
      if (IsBetaOpposite == IsOdd)
        res.beta() = -res.beta();
        
      if (IsGammaOpposite == IsOdd)
        res.gamma() = -res.gamma();
      
      // Saturate results to the requested range
      if (PositiveRangeAlpha && (res.alpha() < 0))
        res.alpha() += Scalar(2 * EIGEN_PI);
      
      if (PositiveRangeBeta && (res.beta() < 0))
        res.beta() += Scalar(2 * EIGEN_PI);
      
      if (PositiveRangeGamma && (res.gamma() < 0))
        res.gamma() += Scalar(2 * EIGEN_PI);
    }
    
    template <typename _Scalar, class _System>
    friend class Eigen::EulerAngles;
  };

#define EIGEN_EULER_SYSTEM_TYPEDEF(A, B, C) \
  /** \ingroup EulerAngles_Module */ \
  typedef EulerSystem<EULER_##A, EULER_##B, EULER_##C> EulerSystem##A##B##C;
  
  EIGEN_EULER_SYSTEM_TYPEDEF(X,Y,Z)
  EIGEN_EULER_SYSTEM_TYPEDEF(X,Y,X)
  EIGEN_EULER_SYSTEM_TYPEDEF(X,Z,Y)
  EIGEN_EULER_SYSTEM_TYPEDEF(X,Z,X)
  
  EIGEN_EULER_SYSTEM_TYPEDEF(Y,Z,X)
  EIGEN_EULER_SYSTEM_TYPEDEF(Y,Z,Y)
  EIGEN_EULER_SYSTEM_TYPEDEF(Y,X,Z)
  EIGEN_EULER_SYSTEM_TYPEDEF(Y,X,Y)
  
  EIGEN_EULER_SYSTEM_TYPEDEF(Z,X,Y)
  EIGEN_EULER_SYSTEM_TYPEDEF(Z,X,Z)
  EIGEN_EULER_SYSTEM_TYPEDEF(Z,Y,X)
  EIGEN_EULER_SYSTEM_TYPEDEF(Z,Y,Z)
}

#endif // EIGEN_EULERSYSTEM_H
