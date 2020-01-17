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
 *    \file include/acado/validated_integrator/ellipsoidal_integrator.ipp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 *    \date   2013
 */


BEGIN_NAMESPACE_ACADO


// IMPLEMENTATION OF PUBLIC MEMBER FUNCTIONS:
// ======================================================================================

template <typename T> returnValue EllipsoidalIntegrator::integrate( double t0, double tf,
																	Tmatrix<T> *x, Tmatrix<T> *p,
																	Tmatrix<T> *w ){
	 
	double t = t0;
	double h;
	
	int Nmax;
	get( MAX_NUM_INTEGRATOR_STEPS, Nmax );
	
	int PrintLevel;
	get( INTEGRATOR_PRINTLEVEL, PrintLevel );
	
	totalTime.start();
	
	int count = 1;
	while( count <= Nmax ){

		if( t >= tf-10.*EPS ) break; 
		h = step( t, tf, x, p, w );
		if( h <= -0.5 ) break;
		t += h;
		count++;
		if( count == Nmax ) setInfinity();
		
		if( PrintLevel == HIGH ){
			std::cout << "\n\nSTEP" << count << ":\n---------------------------------------------------\n";
			std::cout << std::scientific << "\ntime = " << t << "\t h = " << h << "\n";
			std::cout << "\nSTATE ENCLOSURE: \n\n";
			Tmatrix<Interval> result = getStateBound(*x);
			for( int i=0; i<nx; i++ )
				std::cout << "x[" << i <<"]:  " << result(i)
						  << "     R[" << i <<"]:  " << (boundQ())(i) << "\n";
		}
	}
	
	if( PrintLevel == MEDIUM ){
		std::cout << "\ntime = " << t << std::endl;
		std::cout << "\nSTATE ENCLOSURE: \n\n";
		Tmatrix<Interval> result = getStateBound(*x);
		for( int i=0; i<nx; i++ )
			std::cout << "x[" << i <<"]:  " << result(i) << "\n";
	}
	
	totalTime.stop();
	
	int profile;
	get( PRINT_INTEGRATOR_PROFILE, profile );
	
	if ( (BooleanType)profile == BT_TRUE ){
	  
		std::cout << "\nCOMPUTATION TIME: \n\n";
		
		std::cout << "Total  :  " << std::setprecision(3) << std::fixed << totalTime.getTime() << " sec        \n";
		std::cout << "Phase 0:  " << std::setprecision(3) << std::fixed << Phase0Time.getTime() << " sec"
		          << "   ( "     << std::setprecision(1) << std::fixed << 100.0*(Phase0Time.getTime()/totalTime.getTime())
				  << " % )\n" ;
		std::cout << "Phase 1:  " << std::setprecision(3) << std::fixed << Phase1Time.getTime() << " sec"
		          << "   ( "     << std::setprecision(1) << std::fixed << 100.0*(Phase1Time.getTime()/totalTime.getTime())
				  << " % )\n\n" ;
				  
		if( PrintLevel == MEDIUM ) std::cout << "Number of Steps:  " << count << "\n\n";
	}
	
	return SUCCESSFUL_RETURN;
}


template <typename T> double EllipsoidalIntegrator::step( const double &t, const double &tf,
														  Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w ){

	if( nx        != 0 ) if( x == 0 ){ ACADOERROR(RET_DIFFERENTIAL_STATE_DIMENSION_MISMATCH); setInfinity(); return -1.0; }
	if( g.getNP() != 0 ) if( p == 0 ){ ACADOERROR(RET_PARAMETER_DIMENSION_MISMATCH); setInfinity(); return -1.0; }
	if( g.getNW() != 0 ) if( w == 0 ){ ACADOERROR(RET_DISTURBANCE_DIMENSION_MISMATCH); setInfinity(); return -1.0; }

   Tmatrix<T>      coeff;
   Tmatrix<double> C    ;

//      printf("e-i: setup step \n");

   Phase0Time.start();
   phase0( t, x, p, w, coeff, C );
   Phase0Time.stop();

//      printf("e-i: phase 0 succeeded. \n");
   
   double h;
   
   Phase1Time.start();
   h = phase1( t, tf, x, p, w, coeff, C );
   Phase1Time.stop();

   if( h <= -0.5 ){ setInfinity(); return h; }
//      printf("e-i: phase 1 succeeded: h = %.6e \n", h );
   
   phase2( t, h, x, p, w, coeff, C );

//     printf("e-i: phase 2 succeeded.\n" );
   
   return h;
}


template <typename T> Tmatrix<Interval> EllipsoidalIntegrator::getStateBound( const Tmatrix<T> &x ) const{
	
	return bound(x)+boundQ();
}


// IMPLEMENTATION OF PRIVATE MEMBER FUNCTIONS:
// ======================================================================================

template <typename T> void EllipsoidalIntegrator::phase0( double t,
														  Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w,
														  Tmatrix<T> &coeff, Tmatrix<double> &C ){
	
	
	// Compute an interval bound of the ellipsoidal remainder term:
	// --------------------------------------------------------------
	
	Tmatrix<Interval> R = boundQ();
	
// 	printf("\n PHASE 0 \n\n");
	
//  	std::cout << "R:  " << R << "\n";
	
	
	// Evaluate g:
	// --------------------------------------------------------------
	
// 	std::cout << "x:  " << *x << "\n";
// 	std::cout << "p:  " << *p << "\n";
	
	coeff = evaluate( g, t, x, p, w );
	
//  	std::cout << "coeff:  " << coeff << "\n";
	
	
	// Evaluate the Jacobian of g:
	// --------------------------------------------------------------
	
	Tmatrix<Interval> *xI = 0;
	Tmatrix<Interval> *pI = 0;
	Tmatrix<Interval> *wI = 0;
	
	if( x != 0 ) xI = new Tmatrix<Interval>(bound(*x));
	if( p != 0 ) pI = new Tmatrix<Interval>(bound(*p));
	if( w != 0 ) wI = new Tmatrix<Interval>(bound(*w));
	
	
	Tmatrix<T> J = evaluate( dg, t, xI, pI, wI );
	
// 	std::cout << "\n\n J:  " << J << "\n";
	
	C = hat(J);
	
	J -= C;
	
// 	std::cout << "\n\n C:  " << C << "\n";
// 	
// 	std::cout << "\n\n J-C:  " << J << "\n";
	
	for( int i=0; i<N+1; i++ )
		for( int j=0; j<nx; j++ )
			for( int k=0; k<nx; k++ )
				coeff(i*nx+j) += J((i*nx+j)*nx+k)*R(k);
	
//  	std::cout << "\n\n coeff:  " << coeff << "\n";
	
	
	// Evaluate a bound on the second order term of g:
	// --------------------------------------------------------------
	
	Tmatrix<Interval> stack(2*nx);
	for( int i=0; i<nx; i++ ){
		stack(   i) =  xI->operator()(i) + R(i);
		stack(nx+i) =  R(i);
	}

//  	std::cout << "\n\n stack:  " << stack << "\n";
	
	coeff += evaluate( ddg, t, &stack, pI, wI );
	
//  	std::cout << "\n\n coeff:  " << coeff << "\n";
	
	if( xI != 0 ) delete xI;
	if( pI != 0 ) delete pI;
	if( wI != 0 ) delete wI;
	
}


template <typename T> double EllipsoidalIntegrator::phase1(	double t, double tf,
															Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w,
															Tmatrix<T> &coeff,
															Tmatrix<double> &C ){
  
  Tmatrix<Interval> X = bound(*x) + boundQ();

  Tmatrix<Interval> *P = 0;
  Tmatrix<Interval> *W = 0;
  if( p != 0 ) P = new Tmatrix<Interval>( bound(*p) );
  if( w != 0 ) W = new Tmatrix<Interval>( bound(*w) );
  
// 	std::cout << "\n\n X:   " << X << "\n";
// 	std::cout << "\n\n gr:  " << evaluate( gr, t, &X, P, W ) << "\n";
// 	std::cout << "\n\n ||gr||:  " << norm(evaluate( gr, t, &X, P, W ),X) << "\n";
  
	double h0 = ::pow( 1.0/(norm(evaluate( gr, t, &X, P, W ),X)+EPS) , 1./((double) N) );
	
// 	printf("exp = %.16e \n", 1./((double) N) );
// 	printf("ratio = %.16e \n", 1.0/(norm(evaluate( gr, t, &X, P, W ),X)+EPS) );
// 	printf("h0 = %.16e \n", h0 );

// 	if( acadoIsNan() ==  );
	
	
	double hmin,hmax,rho;
	get( MIN_INTEGRATOR_STEPSIZE, hmin );
	get( MAX_INTEGRATOR_STEPSIZE, hmax );
	get( STEPSIZE_TUNING        , rho  );

	h0 *= rho;
	if( h0 < hmin ) h0 = hmin;
	if( h0 > hmax ) h0 = hmax;
	if( t+h0 > tf ) h0 = tf-t;

	double TOL;
	get( INTEGRATOR_TOLERANCE, TOL  );

	Tmatrix<Interval> E(nx);
	for( int i=0; i<nx; i++ ) E(i) = TOL*Interval(-1.,1.);
	
//  	std::cout << "x " << *x << "\n";
	
	*x = phi(coeff,h0);
	
//  	std::cout << "coeff " << coeff << "\n";
// 	std::cout << "x " << *x << "\n";
	
	Tmatrix<Interval> tmp  = bound(*x) + evalC(C,h0)*boundQ();
	Tmatrix<Interval> Xhat = tmp + h0*E;
	Tmatrix<Interval> X0   = tmp + ::pow(h0,N+1)*evaluate( gr, Interval(t,t+h0), &Xhat, P, W );
	
// 	printf("h0 = %.16e, rho = %.16e \n", h0, rho );
// 	std::cout << "h0*E  " << h0*E << "\n";
// 	std::cout << "Xtilde"  << ::pow(h0,N+1)*evaluate( gr, Interval(t,t+h0), &Xhat, P, W ) << "\n";
	
	while( isIncluded( X0, Xhat ) == BT_FALSE ){

      if( h0 < hmin ){ h0 = -1.0; break; }
	  h0 *= 0.5;
// 	  printf("\n\n h0 = %.16e \n--------------\n\n", h0);
	  *x = phi(coeff,h0);
	  tmp  = bound(*x) + evalC(C,h0)*boundQ();
	  Xhat = tmp + h0*E;
	  X0   = tmp + ::pow(h0,N+1)*evaluate( gr, Interval(t,t+h0), &Xhat, P, W );
  }

  if( P != 0 ) delete P;
  if( W != 0 ) delete W;

  return h0;
}



template <typename T> void EllipsoidalIntegrator::phase2(	double t, double h,
															Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w,
															Tmatrix<T> &coeff,
															Tmatrix<double> &C ){

	center(*x);
	
	Tmatrix<Interval> R = getRemainder(*x);
	
	double TOL;
	get( INTEGRATOR_TOLERANCE, TOL  );
	
	Interval discretizationError(-h*TOL,h*TOL);
	
	for( int i=0; i<nx; i++ ) R(i) += discretizationError;
	
	updateQ( evalC2(C,h), R );
	
	*x = getPolynomial(*x);
}


template <typename T> Tmatrix<T> EllipsoidalIntegrator::evaluate(	Function &f, double t,
																	Tmatrix<T> *x, Tmatrix<T> *p,
																	Tmatrix<T> *w ) const{
 
    return evaluate( f, Interval(t), x, p, w );
}


template <typename T> Tmatrix<T> EllipsoidalIntegrator::evaluate(	Function &f, Interval t,
																	Tmatrix<T> *x, Tmatrix<T> *p,
																	Tmatrix<T> *w ) const{

	
	const int na = 0;
	const int nu = 0;
	int np = 0;
	int nw = 0;
	
	if( p!= 0 ) np = p->getDim();
	if( w!= 0 ) nw = w->getDim();
	
	TevaluationPoint<T> z(f,2*nx,na,np,nu,nw);
	
	Tmatrix<T> time(1);
	time(0) = t;
	
	z.setT(time);
	if( x!= 0 ){ z.setX(*x); }
	if( p!= 0 ){ z.setP(*p); }
	if( w!= 0 ){ z.setW(*w); }
	
	return f.evaluate(z);
}


template <typename T> Tmatrix<double> EllipsoidalIntegrator::hat( const Tmatrix<T> &x ) const{

	Tmatrix<double> result(x.getNumRows(),x.getNumCols());
	
	for( int i=0; i < (int) result.getDim(); i++ ) result(i) = (x(i)).constant() + mid((x(i)).remainder());
	
	return result;
}



template <typename T> Tmatrix<T> EllipsoidalIntegrator::phi( const Tmatrix<T> &coeff, const double &h ) const{
    
	Tmatrix<T> r(nx);
	
	for( int i=0; i<nx; i++ ){
	  
		T r_i;
		r_i = coeff(i);
		for( int j=0; j<N; j++ ) r_i += ::pow(h,j+1)*coeff(i+(j+1)*nx);
		r(i) = r_i;
	}
	return r;
}


template <typename T> Tmatrix<Interval> EllipsoidalIntegrator::bound( const Tmatrix<T> &x ) const{

	Tmatrix<Interval> result(x.getDim());

	for( int i=0; i < (int) x.getDim(); i++ ) result(i) = x(i).bound();

	return result;
}



template <typename T> Tmatrix<Interval> EllipsoidalIntegrator::getRemainder( const Tmatrix<T> &x ) const{

	Tmatrix<Interval> R(x.getDim());
	for( uint i=0; i<x.getDim(); i++ ) R(i) = x(i).remainder();
	return R;
}


template <typename T> Tmatrix<T> EllipsoidalIntegrator::getPolynomial( const Tmatrix<T> &x ) const{

	Tmatrix<T> R(x.getDim());
	for( uint i=0; i<x.getDim(); i++ ) R(i) = x(i).polynomial();
	return R;
}


template <typename T> void EllipsoidalIntegrator::center( Tmatrix<T> &x ) const{
 
	for( uint i=0; i<x.getDim(); i++ ) x(i).center();
}




CLOSE_NAMESPACE_ACADO

// end of file.
