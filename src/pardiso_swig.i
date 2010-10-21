%module core
%{
#define SWIG_FILE_WITH_INIT


extern int pardiso_(void *, int *, int *, int *, int *, int *,
                void *, int *, int*, int *, int *, int *,
                int *, void *, void*, int *);

%}
%include "numpy.i"
%init %{
import_array();
%}

%pythoncode %{
import numpy
%}

%apply long INPLACE_ARRAY1[ANY] { long pt[64] };
%apply (int* INPUT) { int *maxfct, int *mnum, int *mtype, int *phase, int *n, int *nrhs, int *msglvl };
%apply (double* IN_ARRAY1, int DIM1) { (double *a, int na) };
%apply (int* IN_ARRAY1, int DIM1) { (int *ia, int nia), (int *ja, int nja) };
%apply (int* INPLACE_ARRAY1, int DIM1) { (int *perm, int nperm) };
%apply (int INPLACE_ARRAY1[ANY]) { int iparm[64] };
%apply (double* INPLACE_ARRAY1, int DIM1) { (double *b, int nb) };
%apply (double* ARGOUT_ARRAY1, int DIM1) { (double *x, int nx) };
%apply int* OUTPUT { int *error };

%numpy_typemaps(npy_cdouble, NPY_CDOUBLE, int)


%inline %{

void pardiso_real(long pt[64], int *maxfct, int *mnum, int *mtype, int *phase, int *n, double *a, int na, int *ia, int nia, int *ja, int nja, int *perm, int nperm, int *nrhs, int iparm[64], int *msglvl, double *b, int nb, double *x, int nx, int *error){
	pardiso_(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x,error);
	}
%}

%apply (npy_cdouble* IN_ARRAY1, int DIM1) { (npy_cdouble *a, int na) };
%apply (npy_cdouble* INPLACE_ARRAY1, int DIM1) { (npy_cdouble *b, int nb) };
%apply (npy_cdouble* ARGOUT_ARRAY1, int DIM1) { (npy_cdouble *x, int nx) };

%inline %{

void pardiso_complex(long pt[64], int *maxfct, int *mnum, int *mtype, int *phase, int *n, npy_cdouble *a, int na, int *ia, int nia, int *ja, int nja, int *perm, int nperm, int *nrhs, int iparm[64], int *msglvl, npy_cdouble *b, int nb, npy_cdouble *x, int nx, int *error){
	pardiso_(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x,error);
	}
%}

%pythoncode %{

	def pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b):
		"""Low-level interface to the real double precision Pardiso solver
		
		   (x,error)=pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
		
		   For a detailed description of the Parameters see the MKL Pardio documentation.
		   The following data types are assumed
		   
		   pt        - numpy.long
		   maxfct    - int
		   mnum      - int
		   mtype     - int
		   phase     - int
		   n         - int
		   a         - numpy.double
		   ia        - numpy.int32
		   ja        - numpy.int32
		   perm      - numpy.int32
		   nrhs,     - int
		   iparm     - numpy.int32
		   msglvl    - int
		   b         - numpy.double
		   
		   Note that Fortran convention is assumed for the arrays (i.e. column-major order and
		   indices start from 1)
		   
		"""
		return _core.pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,n)
		   
	def pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b):
		"""Low-level interface to the complex double precision Pardiso solver
		
		   (x,error)=pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
		
		   For a detailed description of the Parameters see the MKL Pardio documentation.
		   The following data types are assumed
		   
		   pt        - numpy.long
		   maxfct    - int
		   mnum      - int
		   mtype     - int
		   phase     - int
		   n         - int
		   a         - numpy.complex128
		   ia        - numpy.int32
		   ja        - numpy.int32
		   perm      - numpy.int32
		   nrhs,     - int
		   iparm     - numpy.int32
		   msglvl    - int
		   b         - numpy.complex128
		   
		   Note that Fortran convention is assumed for the arrays (i.e. column-major order and
		   indices start from 1)
		   
		"""
		return _core.pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,n)
		
%}   

%pythoncode %{

if __name__ == "__main__":
	# Matrix data
	n=5
	ia=numpy.array([1,4,6,9,12,14],dtype=numpy.int32)
	ja=numpy.array([1,2,4,1,2,3,4,5,1,3,4,2,5],dtype=numpy.int32)
	a=numpy.array([1.0,-1.0,-3.0,-2.0,5.0,4.0,6.0,4.0,-4.0,2.0,7.0,8.0,-5.0],dtype=numpy.double)
	mtype=11
	nrhs=1
	b=numpy.array([1,1,1,1,1],dtype=numpy.double)
	x=numpy.zeros(5,dtype=numpy.double)
	iparm=numpy.zeros(64,dtype=numpy.int32)
	iparm[0]=1
	iparm[1]=2
	iparm[2]=2
	iparm[7]=2
	iparm[9]=13
	iparm[10]=1
	iparm[17]=-1
	iparm[18]=-1
	maxfct=1
	mnum=1
	msglvl=1
	error=0
	perm=numpy.zeros(n,dtype=numpy.int32)
	
	pt=numpy.zeros(64,dtype=numpy.long)
	phase=11
	
	
	(x,error)=pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
	
	phase=22
	(x,error)=pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
	
	phase=33;
	iparm[7]=2
	(x,error)=pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
	print "Solution: "
	print x
	phase=-1
	(x,error)=pardiso_real(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
		
		
	# Complex test
		
	n=8
	ia=numpy.array([ 1, 5, 8, 10, 12, 13, 16, 18, 21 ],dtype=numpy.int32)
	ja=numpy.array([1,3,6,7, 2, 3, 5,3,8,4,7,2,3,6,8,2,7,3,7, 8],dtype=numpy.int32)
	a=numpy.array([7+1j,1+1j,2+1j,7+1j,-4+0j,8+1j,2+1j,1+1j,5+1j,7+0j,9+1j,-4+1j,7+1j,3+1j,8+0j,1+1j,11+1j,-3+1j,2+1j,5+0j],dtype='complex128')
                   
                 
	mtype=13
	nrhs=1
	pt=numpy.zeros(64,dtype=numpy.long)
	iparm=numpy.zeros(64,dtype=numpy.int32)
	iparm[0] = 1
	iparm[1] = 2 
	iparm[2] = 2
	iparm[3] = 0 
	iparm[4] = 0 
	iparm[5] = 0 
	iparm[6] = 0 
	iparm[7] = 2 
	iparm[8] = 0 
	iparm[9] = 13  
	iparm[10] = 1 
	iparm[11] = 0 
	iparm[12] = 1 
	iparm[13] = 0 
	iparm[14] = 0 
	iparm[15] = 0 
	iparm[16] = 0
	iparm[17] = -1 
	iparm[18] = -1
	iparm[19] = 0 
	maxfct=1
	mnum=1
	msglvl=1
	error=0
	perm=numpy.zeros(n,dtype=numpy.int32)
	phase=11
	b=(1+1j)*numpy.ones(n,dtype='complex128')
		
	(x,error)=pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
	
	phase=22
	(x,error)=pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
	
	phase=33;
	iparm[7]=1
	(x,error)=pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
	print "Solution: "
	print x
	phase=-1
	(x,error)=pardiso_complex(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
	print "Error %i" % error
             
	
	
%}
	