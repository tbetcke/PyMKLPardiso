'''
Created on Oct 21, 2010

@author: tbetcke
'''

from scipy.sparse import csr_matrix
import numpy
import core

def solve(A,b,iparm=None,msglvl=1):
    """Solve a general unsymmetric linear system using Pardiso
    
    x=solve(A,b,iparm=None)
    INPUT:
    A      - Scipy Sparse Matrix. If the matrix is not in csr format it is converted to csr
    b      - Right-Hand Side vector (multiple right-hand sides are possible)
    iparm  - Optional parameter vector for Pardiso (see MKL Pardiso documentation for details)
             If not specified default Pardiso values are used
    msglvl - Determine the verbosity. If msglvl=1 print statisticial information. If msglvl=0
             no information is given.
             
    """
    
    # Check the type of the matrix and convert if necessary
    if not isinstance(A,csr_matrix): A=csr_matrix(A)
    
    ia=A.indptr
    ja=A.indices
    a=A.data
    
    # Adjust data for Fortran format
    
    ja = ja+1
    ia = ia+1
    
    

    # Assign variables and fix data types

    complex=False
    n=A.shape[0]
    if b.ndim>1: 
        nrhs=b.shape[1]
    else:
        nrhs=1

    if not ia.dtype==numpy.int32: ia=numpy.array(ia,numpy.int32)
    if not ja.dtype==numpy.int32: ja=numpy.array(ja,numpy.int32)
    if numpy.iscomplexobj(a) or numpy.iscomplexobj(b):
        if not a.dtype==numpy.complex128: a=numpy.array(a,numpy.complex128)
        if not b.dtype==numpy.complex128: b=numpy.array(b,numpy.complex128)
        complex=True
    else:
        if not a.dtype==numpy.double: a=numpy.array(a,numpy.double)
        if not b.dtype==numpy.double: b=numpy.array(a,numpy.double)
    
    if complex:
        dt=numpy.complex128
        pardiso=core.pardiso_complex
        mtype=13
    else:
        dt=numpy.double
        pardiso=core.pardiso_real
        mtype=11
    
    b=b.flatten('F')
    
    if iparm==None: iparm=numpy.zeros(64,dtype=numpy.int32)
    if not iparm.dtype==numpy.int32: iparm=numpy.array(iparm,numpy.int32)
    
    pt=numpy.zeros(64,dtype=numpy.long)
    perm=numpy.zeros(n,dtype=numpy.int32)
    x=numpy.zeros(n,dtype=dt)
    
    maxfct=1
    mnum=1
    
    # Solve
    phase=13
    x,error=pardiso(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
    
    # Clean up
    phase=-1
    pardiso(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b)
    
    return (x,error)
    
        
if __name__ == "__main__":

    from scipy import sparse
    from numpy import linalg
    from numpy.random import rand
        
    
    n=1000
    m=100
    A = sparse.lil_matrix((n, n))
    A[0, :m] = rand(m)
    A[1, m:2*m] = A[0, :m]
    A.setdiag(rand(n))
    A = A.tocsr()
    b = rand(n)

    x,error = solve(A, b)
    if error==0: 
        print "No error during computation"
    else:
        print "An error occurred"
        
    residual=linalg.norm(A*x-b)
    
    print "Residual: %e" % residual
    
    