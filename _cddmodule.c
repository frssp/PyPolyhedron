/*
 * _cddmodule.c is Python/CAPI interface to C-Library cddlib.
 * Author: Pearu Peterson <pearu@ioc.ee>
 * WWW: http://cens.ioc.ee/projects/
 * Requires NumPy: http://scipy.org/
 *
 * Use polyhedron.py for user friendly interface.
 * Anyway, this module exports two functions:
 *   hrep(A,b=[0],eq=[])
 *   vrep(v,r=[],fr=[])
 * returning both a 10-tuple
 *   (A,b,eq,Gen,is_v,fr,Inc,Adj,InInc,InAdj)
 * where
 *   A,b define polyhedron by a matrix inequality A x <= b;
 *   eq  is a list of integers for which A_i x == b_i, i in eq;
 *   Gen is array of generators (vertices,rays) defining polyhedron;
 *   is_v is a list of integers such that `Gen[i] is vertex <=> is_v[i]==1';
 *   fr  is a list of integers for which i-th Gen is free, i in fr;
 * Both eq and fr may contain also negative indices that are interpreted
 * in the same way as for Python lists.
 *   Inc   defines generator-halfspaces relations
 *   Adj   defines generator-generators relations
 *   InInc defines halfspace-generators relations
 *   InAdj defines halfspace-halfspaces relations
 * Halfspace[len(A)] corresponds to Infinity
 *
 * Note: different from cddlib, all indices start from 0, not from 1.
 *
 * For more information about polyhedrons, see
 *   http://www.ifor.math.ethz.ch/ifor/staff/fukuda/polyfaq/polyfaq.html
 *
 * Copyright 2000, 2007 Pearu Peterson all rights reserved,
 * Pearu Peterson <pearu@ioc.ee>          
 * Permission to use, modify, and distribute this software is given under the
 * terms of the LGPL.  See http://www.fsf.org
 *
 * NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
 *
 * README.cddlib:
...
1. The C-library  cddlib is a C implementation of the Double Description 
Method of Motzkin et al. for generating all vertices (i.e. extreme points)
and extreme rays of a general convex polyhedron in R^d given by a system 
of linear inequalities:

   P = { x=(x1, ..., xd)^T :  b - A  x  >= 0 }

where  A  is a given m x d real matrix, b is a given m-vector 
and 0 is the m-vector of all zeros.
...
 * The author of cddlib is Komei Fukuda <fukuda@ifor.math.ethz.ch>.
 * See http://www.ifor.math.ethz.ch/staff/fukuda/fukuda.html
 *
 */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "setoper.h"
#include "cdd.h"

#define Inequality dd_Inequality
#define Real dd_Real
#define NoError dd_NoError
#define Generator dd_Generator

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#define PRINTPYOBJERR(obj)\
  fprintf(stderr,"afoo.error is related to ");\
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
/*     Here starts Travis Oliphant's contribution (copied from f2py) */
#define INCREMENT(ret_ind, nd, max_ind) \
{ \
  int k; \
  k = (nd) - 1; \
  if (k<0) (ret_ind)[0] = (max_ind)[0]; else \
  if (++(ret_ind)[k] >= (max_ind)[k]) { \
    while (k >= 0 && ((ret_ind)[k] >= (max_ind)[k]-1)) \
      (ret_ind)[k--] = 0; \
    if (k >= 0) (ret_ind)[k]++; \
    else (ret_ind)[0] = (max_ind)[0]; \
  }  \
}
#define CALCINDEX(indx, nd_index, strides, ndim) \
{ \
  int i; \
  indx = 0; \
  for (i=0; i < (ndim); i++)  \
    indx += nd_index[i]*strides[i]; \
} 
#ifndef NUMERIC
extern
int copy_ND_array(const PyArrayObject *arr, PyArrayObject *out)
{
    return PyArray_CopyInto(out, (PyArrayObject *)arr);
}
#else
static int copy_ND_array(PyArrayObject *in, PyArrayObject *out)
{

  /* This routine copies an N-D array in to an N-D array out where both
     can be discontiguous.  An appropriate (raw) cast is made on the data.
  */

  /* It works by using an N-1 length vector to hold the N-1 first indices 
     into the array.  This counter is looped through copying (and casting) 
     the entire last dimension at a time.
  */

  int *nd_index, indx1;
  int indx2, last_dim;
  int instep, outstep;

  if (0 == in->nd) {
    in->descr->cast[out->descr->type_num]((void *)in->data,1,(void *)out->data,1,1);
    return 0;
  }
  nd_index = (int *)calloc(in->nd-1,sizeof(int));
  last_dim = in->nd - 1;
  instep = in->strides[last_dim] / in->descr->elsize;
  outstep = out->strides[last_dim] / out->descr->elsize;
  if (NULL == nd_index ) {
     fprintf(stderr,"Could not allocate memory for index array.\n");
     return -1;
  }
  CFUNCSMESS("copy_ND_array: doing a complete copy\n");
  while(nd_index[0] != in->dimensions[0]) {
    CALCINDEX(indx1,nd_index,in->strides,in->nd-1);
    CALCINDEX(indx2,nd_index,out->strides,out->nd-1);
    /* Copy (with an appropriate cast) the last dimension of the array */
    (in->descr->cast[out->descr->type_num])((void *)(in->data+indx1),instep,\
					    (void *)(out->data+indx2),\
					    outstep,in->dimensions[last_dim]); 
    INCREMENT(nd_index,in->nd-1,in->dimensions);
  }
  free(nd_index);
  return 0;
} 
/* EOF T.O.'s contib */
#endif

static PyArrayObject *arr_from_pyobj(int type,npy_intp *dims,int rank,PyObject *obj) {
  PyArrayObject *self = NULL;
  PyArrayObject *self_cp = NULL;
  int i;
  if (obj == Py_None) {
    CFUNCSMESS("arr_from_pyobj: obj = None. Doing FromDims\n");
    /*self = (PyArrayObject *)PyArray_FromDims(rank,dims,type);*/
    self = (PyArrayObject *)PyArray_SimpleNew(rank, dims, type);
  } else {
    CFUNCSMESS("arr_from_pyobj: Trying ContigiousFromObject\n");
    self = (PyArrayObject *)PyArray_ContiguousFromObject(obj,type,0,0);
    if (self == NULL)
      CFUNCSMESS("arr_from_pyobj:  ContigiousFromObject unsuccesful\n");
  }
  if ((self == NULL) && PyArray_Check(obj)) { /* if could not cast safely in above */
    int loc_rank = ((PyArrayObject *)obj)->nd;
    npy_intp *loc_dims = ((PyArrayObject *)obj)->dimensions;
    CFUNCSMESS("arr_from_pyobj: isarray(obj). Doing FromDims\n");
    /*self = (PyArrayObject *)PyArray_FromDims(loc_rank,loc_dims,type);*/
    self = (PyArrayObject *)PyArray_SimpleNew(loc_rank, loc_dims, type);
  }
  if (self == NULL) {
    int i;
    fprintf(stderr,"arr_from_pyobj: PyArray_FromDims failed (rank=%d,type=%d,dims=(%"NPY_INTP_FMT,\
	    rank,type,dims[0]);
    for(i=1;i<rank;i++) fprintf(stderr,", %" NPY_INTP_FMT,dims[i]);
    fprintf(stderr,"))\n");
    goto capi_fail;
  }
    self_cp = self;
  if (!(rank==self->nd)) {
    int u_dim = -1, dims_s = 1, self_s = (self->nd)?PyArray_Size((PyObject *)self):1;
    CFUNCSMESS("arr_from_pyobj: Mismatch of ranks. Trying to match.\n");
    CFUNCSMESS("arr_from_pyobj:");
#ifdef DEBUGCFUNCS
    fprintf(stderr,"rank=%d,self->nd=%d,dims=(",rank,self->nd);
    for(i=0;i<rank;i++) fprintf(stderr," %"NPY_INTP_FMT,dims[i]);
    fprintf(stderr,")\n");
#endif
    for(i=0;i<rank;i++)
      if (dims[i]<1)
        if (u_dim<0) u_dim = i;
        else dims[i] = 1;
      else dims_s *= dims[i];
    if (u_dim >= 0) {
      dims[u_dim] = self_s/dims_s;
      dims_s *= dims[u_dim];
    }
    CFUNCSMESS("arr_from_pyobj:");
#ifdef DEBUGCFUNCS
    fprintf(stderr,"rank=%d,self->nd=%d,self_s=%d,dims_s=%d,dims=(",rank,self->nd,self_s,dims_s);
    for(i=0;i<rank;i++) fprintf(stderr," %"NPY_INTP_FMT,dims[i]);
    fprintf(stderr,")\n");
#endif
    if (self_s != dims_s) {
      fprintf(stderr,"afoo:arr_from_pyobj: expected rank-%d array but got rank-%d array with different size.\n",rank,self->nd);
    goto capi_fail;
    }
    /*    self = (PyArrayObject *)PyArray_FromDimsAndDataAndDescr(rank,dims,self_cp->descr,\
	  self_cp->data);*/
    self = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, self_cp->descr,
						 rank, dims,
						 NULL, self_cp->data,
						 NPY_CARRAY, NULL);

    if (self == NULL)
      goto capi_fail;
    Py_INCREF(self_cp);
    self->base = (PyObject *)self_cp;
  }
  for (i=0;i<rank;i++)
    if (dims[i]>self->dimensions[i]) {
      fprintf(stderr,"afoo:arr_from_pyobj: %d-th dimension must be at least %"NPY_INTP_FMT" but got %"NPY_INTP_FMT".\n",\
	      i+1,dims[i],self->dimensions[i]);
      goto capi_fail;
    }
  if (((PyObject *)self_cp != obj) && PyArray_Check(obj)) {
    if (copy_ND_array((PyArrayObject *)obj,self_cp)) {
      fprintf(stderr,"afoo:arr_from_pyobj: failed to copy object to rank-%d array with shape (",\
	      self_cp->nd);
      for(i=0;i<self_cp->nd;i++) fprintf(stderr,"%"NPY_INTP_FMT",",self_cp->dimensions[i]);
      fprintf(stderr,")\n");
      PRINTPYOBJERR((PyObject *)self_cp);
      goto capi_fail;
    }
  }
  if (self != NULL)
    return self;
  CFUNCSMESS("arr_from_pyobj: self==NULL. Confused?!.\n");
capi_fail:
  PRINTPYOBJERR(obj);
  Py_XDECREF(self);
  return NULL;
}

static int cdd_SetFamily2PyList(dd_SetFamilyPtr *f, PyObject **l_py)
     /*  dd_SetFamilyPtr *f;
	 PyObject **l_py;*/
{
  int i,s1;
  long j;
  PyObject *row_py;
  CFUNCSMESS("cdd_SetFamily2PyList()");
  s1=(*f)->famsize;
  *l_py=PyList_New(s1);
  for (i=0;i<s1;i++) {
    if (!(row_py=PyList_New(0))) return 0;
    for (j=1;j<=(*f)->set[i][0];j++)
      if (set_member(j,(*f)->set[i]))
        PyList_Append(row_py,PyLong_FromLong(j-1));
    PyList_SetItem(*l_py,i,row_py);
  }
  return 1;
}

static PyObject *cdd_error = NULL;

static PyObject *cdd_Poly2PyTuple(dd_PolyhedraPtr poly) {
  dd_SetFamilyPtr Inc_dd = NULL;
  dd_SetFamilyPtr Adj_dd = NULL;
  dd_SetFamilyPtr InInc_dd = NULL;
  dd_SetFamilyPtr InAdj_dd = NULL;
  PyArrayObject *py_Gen = NULL;
  /*PyArrayObject *py_GenRay = NULL;*/
  PyArrayObject *py_A = NULL;
  PyArrayObject *py_b = NULL;
  PyObject *Inc_py = NULL;
  PyObject *Adj_py = NULL;
  PyObject *InInc_py = NULL;
  PyObject *InAdj_py = NULL;
  PyObject *Eq_py = NULL;
  PyObject *Fr_py = NULL;
  PyObject *IsVertex_py = NULL;

  PyObject *buildvalue = NULL;
  CFUNCSMESS("cdd_Poly2PyTuple()");
  
  { /* Get inequalities (A,b,eq: b-Ax>=0) from poly */
    int m,d,i,j;
    double *A = NULL;
    npy_intp A_Dims[] = {-1, -1};
    double *b = NULL;
    npy_intp b_Dims[] = {-1};
    dd_MatrixPtr Ineq_dd = NULL;

    Ineq_dd = dd_CopyInequalities(poly);
    d = Ineq_dd->colsize - 1;
    m = Ineq_dd->rowsize;
    A_Dims[1] = d;
    A_Dims[0] = m;
    b_Dims[0] = m;
    py_A = arr_from_pyobj(PyArray_DOUBLE,A_Dims,2,Py_None);
    py_b = arr_from_pyobj(PyArray_DOUBLE,b_Dims,1,Py_None);
    A = (double *)(py_A->data);
    b = (double *)(py_b->data);
    for (i=0; i < m; i++)
      for (j=0; j < d+1; j++)
	if (j) A[i*d+j-1] = -dd_get_d(Ineq_dd->matrix[i][j]);
	else b[i] = dd_get_d(Ineq_dd->matrix[i][0]);
    Eq_py = PyList_New(0);
    if (set_card(Ineq_dd->linset)>0)
      for (i=0; i<m; i++)
	if (set_member(i+1, Ineq_dd->linset))
	  PyList_Append(Eq_py,PyLong_FromLong(i));
    if (Ineq_dd!=NULL) dd_FreeMatrix(Ineq_dd);
  }
  { /* Get generators (vertices and rays,fr) from poly */
    int i,j,d,m;
    /*dd_rowrange total;*/
    dd_MatrixPtr Gen_dd = NULL;
    double *Gen = NULL;
    npy_intp Gen_Dims[] = {-1, -1};
    /*double *GenRay = NULL;*/
    /*npy_intp GenRay_Dims[] = {-1, -1};*/

    Gen_dd = dd_CopyGenerators(poly);
    d = Gen_dd->colsize - 1;
    m = Gen_dd->rowsize;
    Gen_Dims[0] = m;
    Gen_Dims[1] = d;
    py_Gen = arr_from_pyobj(PyArray_DOUBLE,Gen_Dims,2,Py_None);
    Gen = (double *)(py_Gen->data);
    IsVertex_py = PyList_New(0);
    for (i=0; i < m; i++) {
      PyList_Append(IsVertex_py,PyLong_FromLong((int)dd_get_d(Gen_dd->matrix[i][0])));
      for (j=1; j<d+1; j++)
	Gen[i*d+j-1] = dd_get_d(Gen_dd->matrix[i][j]);
    }
	
    Fr_py = PyList_New(0);
    if (set_card(Gen_dd->linset)>0)
      for (i=0; i<m; i++)
	if (set_member(i+1, Gen_dd->linset))
	  PyList_Append(Fr_py,PyLong_FromLong(i));
    if (Gen_dd!=NULL) dd_FreeMatrix(Gen_dd);
  }
  if (poly->representation==Inequality) {
    Inc_dd = dd_CopyIncidence(poly);
    Adj_dd = dd_CopyAdjacency(poly);
    InInc_dd = dd_CopyInputIncidence(poly);
    InAdj_dd = dd_CopyInputAdjacency(poly);
  } else {
    InInc_dd = dd_CopyIncidence(poly);
    InAdj_dd = dd_CopyAdjacency(poly);
    Inc_dd = dd_CopyInputIncidence(poly);
    Adj_dd = dd_CopyInputAdjacency(poly);
  }

  if (!cdd_SetFamily2PyList(&Inc_dd,&Inc_py)) {
    PyErr_SetString(cdd_error,"failed to copy setfamily `Inc' to list" );
    goto fail;
  }
  if (!cdd_SetFamily2PyList(&Adj_dd,&Adj_py)) {
    PyErr_SetString(cdd_error,"failed to copy setfamily `Adj' to list" );
    goto fail;
  }
  if (!cdd_SetFamily2PyList(&InInc_dd,&InInc_py)) {
    PyErr_SetString(cdd_error,"failed to copy setfamily `InInc' to list" );
    goto fail;
  }
  if (!cdd_SetFamily2PyList(&InAdj_dd,&InAdj_py)) {
    PyErr_SetString(cdd_error,"failed to copy setfamily `InAdj' to list" );
    goto fail;
  }
  buildvalue = Py_BuildValue("NNNNNNNNNN",py_A,py_b,Eq_py,py_Gen,IsVertex_py,Fr_py,
			     Inc_py,Adj_py,InInc_py,InAdj_py);
 fail:
  if (Inc_dd!=NULL) dd_FreeSetFamily(Inc_dd);
  if (Adj_dd!=NULL) dd_FreeSetFamily(Adj_dd);
  if (InInc_dd!=NULL) dd_FreeSetFamily(InInc_dd);
  if (InAdj_dd!=NULL) dd_FreeSetFamily(InAdj_dd);
  return buildvalue;
}

static char doc_hrep[] = "(A,b,eq,Gen,is_v,fr,Inc,Adj,InInc,InAdj) = hrep(A,b=[0],eq=[])";

static PyObject *cdd_hrep(PyObject *self, PyObject *args, PyObject *kws) {
  int m,d,i,j;
  mytype value;
  dd_PolyhedraPtr poly = NULL;
  dd_ErrorType err;
  
  dd_MatrixPtr Ineq_dd = NULL;
  PyObject *A_py = Py_None;
  PyArrayObject *py_A = NULL;
  double *A = NULL;
  npy_intp A_Dims[] = {-1, -1};

  PyObject *b_py = Py_None;
  PyArrayObject *py_b = NULL;
  double *b = NULL;
  npy_intp b_Dims[] = {-1};
  
  PyObject *eq_py = Py_None;
  PyArrayObject *py_eq = NULL;
  int *eq = NULL;
  npy_intp eq_Dims[] = {-1};

  PyObject *buildvalue = NULL;
  static char *kwlist[] = {"A","b","eq",NULL};

  if (!PyArg_ParseTupleAndKeywords(args,kws,
				   "O|OO:_cdd.hrep",kwlist,&A_py,&b_py,&eq_py))
    goto fail;
  py_A = arr_from_pyobj(PyArray_DOUBLE,A_Dims,2,A_py);
  if (py_A == NULL) {
    PyErr_SetString(cdd_error,"failed in converting `A' to C array" );
    goto fail;
  }
  A = (double *)(py_A->data);

  if (b_py == Py_None)
    b_Dims[0] = py_A->dimensions[0];
  py_b = arr_from_pyobj(PyArray_DOUBLE,b_Dims,1,b_py);
  if (py_b == NULL) {
    PyErr_SetString(cdd_error,"failed in converting `b' to C array" );
    goto fail;
  }
  b = (double *)(py_b->data);
  
  m = py_A->dimensions[0];
  d = py_A->dimensions[1];
  if (m!=py_b->dimensions[0]) {
    PyErr_SetString(cdd_error,"mismatch of `A' and `b' lengths");
    goto fail;
  }
  if (b_py == Py_None)
    for (i=0;i<m;i++) b[i] = 0.0;

  Ineq_dd=dd_CreateMatrix(m, d+1);
  Ineq_dd->representation=Inequality;
  Ineq_dd->numbtype=Real;
  dd_init(value);
  for (i=0; i<m; i++)
    for (j=0; j<d+1; j++) {
      if (j) dd_set_d(value, -A[i*d+j-1]);
      else dd_set_d(value, b[i]);
      dd_set(Ineq_dd->matrix[i][j],value);
    }
  if (eq_py!=Py_None) {
    py_eq = arr_from_pyobj(PyArray_INT,eq_Dims,1,eq_py);
    if (py_eq == NULL) {
      PyErr_SetString(cdd_error,"failed in converting `eq' to C array" );
      goto fail;
    }
    eq = (int *)(py_eq->data);
    for (i=0; i<py_eq->dimensions[0]; i++) {
      if (eq[i]<0) eq[i] += m;
      if ((eq[i]<0)||(eq[i]>=m))
	printf("there is inconsistencies in equality setting (0<=(eq[%i]=%i)<m=%i)\n",i,eq[i],m);
      set_addelem(Ineq_dd->linset,eq[i]+1);
    }
  }

  poly=dd_DDMatrix2Poly(Ineq_dd, &err);
  if (err!=NoError) {
    PyErr_SetString(cdd_error,"failed to convert matrix to poly" );
    dd_WriteErrorMessages(stdout,err);
    goto fail;
  }
  buildvalue = cdd_Poly2PyTuple(poly);

 fail:
  if (poly!=NULL) dd_FreePolyhedra(poly);
  if (Ineq_dd!=NULL) dd_FreeMatrix(Ineq_dd);
  if (py_A!=NULL) {Py_XDECREF(py_A->base); }
  Py_XDECREF(py_A);
  if (py_b!=NULL) {Py_XDECREF(py_b->base); }
  Py_XDECREF(py_b);
  if (py_eq!=NULL) {Py_XDECREF(py_eq); /* ??????? */ }
  return buildvalue;
}

static char doc_vrep[] = "(A,b,eq,Gen,is_v,fr,Inc,Adj,InInc,InAdj) = vrep(v,r=[],fr=[])";

static PyObject *cdd_vrep(PyObject *self, PyObject *args, PyObject *kws) {
  int m,d,i,j,n,s;
  mytype value;
  dd_PolyhedraPtr poly = NULL;
  dd_ErrorType err;
  
  dd_MatrixPtr Gen_dd = NULL;
  PyObject *Gen_py = Py_None;
  PyArrayObject *py_Gen = NULL;
  double *Gen = NULL;
  npy_intp Gen_Dims[] = {-1, -1};

  PyObject *GenRay_py = Py_None;
  PyArrayObject *py_GenRay = NULL;
  double *GenRay = NULL;
  npy_intp GenRay_Dims[] = {-1, -1};
  
  PyObject *fr_py = Py_None;
  PyArrayObject *py_fr = NULL;
  int *fr = NULL;
  npy_intp fr_Dims[] = {-1};

  PyObject *buildvalue = NULL;
  static char *kwlist[] = {"v","r","fr",NULL};

  if (!PyArg_ParseTupleAndKeywords(args,kws,
				   "O|OO:_cdd.vrep",kwlist,&Gen_py,&GenRay_py,&fr_py))
    goto fail;

  py_Gen = arr_from_pyobj(PyArray_DOUBLE,Gen_Dims,2,Gen_py);
  if (py_Gen == NULL) {
    PyErr_SetString(cdd_error,"failed in converting `v' to C array" );
    goto fail;
  }
  Gen = (double *)(py_Gen->data);

  if (GenRay_py==Py_None) GenRay_Dims[0] = 0;
  GenRay_Dims[1] = py_Gen->dimensions[1];
  
  py_GenRay = arr_from_pyobj(PyArray_DOUBLE,GenRay_Dims,2,GenRay_py);
  if (py_GenRay == NULL) {
    PyErr_SetString(cdd_error,"failed in converting `r' to C array" );
    goto fail;
  }
  GenRay = (double *)(py_GenRay->data);

  n = py_Gen->dimensions[0];
  s = py_GenRay->dimensions[0];
  m = n + s;
  d = py_Gen->dimensions[1];

  Gen_dd=dd_CreateMatrix(m, d+1);
  Gen_dd->representation=Generator;
  Gen_dd->numbtype=Real;
  dd_init(value);
  for (i=0; i<n; i++)
    for (j=0; j<d+1; j++) {
      if (j) dd_set_d(value, Gen[i*d+j-1]);
      else dd_set_d(value, 1);
      dd_set(Gen_dd->matrix[i][j],value);
    }
  for (i=0; i<s; i++)
    for (j=0; j<d+1; j++) {
      if (j) dd_set_d(value, GenRay[i*d+j-1]);
      else dd_set_d(value, 0);
      dd_set(Gen_dd->matrix[n+i][j],value);
    }

  if (fr_py!=Py_None) {
    py_fr = arr_from_pyobj(PyArray_INT,fr_Dims,1,fr_py);
    if (py_fr == NULL) {
      PyErr_SetString(cdd_error,"failed in converting `fr' to C array" );
      goto fail;
    }
    fr = (int *)(py_fr->data);
    for (i=0; i<py_fr->dimensions[0]; i++) {
      if (fr[i]<0) fr[i] += m;
      if ((fr[i]<0)||(fr[i]>=m))
	printf("there is inconsistencies in free setting (0<=(fr[%i]=%i)<m=%i)\n",i,fr[i],m);
      set_addelem(Gen_dd->linset,fr[i]+1);
    }
  }

  poly=dd_DDMatrix2Poly(Gen_dd, &err);
  if (err!=NoError) {
    PyErr_SetString(cdd_error,"failed to convert matrix to poly" );
    dd_WriteErrorMessages(stdout,err);
    goto fail;
  }
  buildvalue = cdd_Poly2PyTuple(poly);

 fail:
  if (poly!=NULL) dd_FreePolyhedra(poly);
  if (Gen_dd!=NULL) dd_FreeMatrix(Gen_dd);
  if (py_Gen!=NULL) {Py_XDECREF(py_Gen->base);}
  Py_XDECREF(py_Gen);
  if (py_GenRay!=NULL) {Py_XDECREF(py_GenRay->base);}
  Py_XDECREF(py_GenRay);
  if (py_fr!=NULL) {Py_XDECREF(py_fr); /* ??????? */}
  return buildvalue;
}

static PyMethodDef cdd_module_methods[] = {
  {"hrep", (PyCFunction)cdd_hrep,METH_VARARGS|METH_KEYWORDS,doc_hrep},
  {"vrep", (PyCFunction)cdd_vrep,METH_VARARGS|METH_KEYWORDS,doc_vrep},
  {NULL, NULL} /*sentinel*/
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef CDDCallModuleDef = {
  PyModuleDef_HEAD_INIT,
  "_cdd",
  NULL,
  -1,
  cdd_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit__cdd(void)
{
  PyObject *m, *d, *s;
  dd_set_global_constants();
  m = PyModule_Create(&CDDCallModuleDef);
  if (m == NULL)
    return NULL;
  import_array();
  d = PyModule_GetDict(m);
  s = PyUnicode_FromString("0.2");
  PyDict_SetItemString(d, "__version__", s);
  s = PyUnicode_FromString("See polyhedron.py");
  PyDict_SetItemString(d, "__doc__", s);
  cdd_error = PyUnicode_FromString("_cdd.error");
  PyDict_SetItemString(d, "error", cdd_error);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module _cdd");
  return m;
}

#else
PyMODINIT_FUNC init_cdd(void)
{
  PyObject *m, *d, *s;
  dd_set_global_constants();
  m=Py_InitModule("_cdd",cdd_module_methods);
  import_array();
  d = PyModule_GetDict(m);
  s = PyString_FromString("0.2");
  PyDict_SetItemString(d, "__version__", s);
  s = PyString_FromString("See polyhedron.py");
  PyDict_SetItemString(d, "__doc__", s);
  cdd_error = PyString_FromString("_cdd.error");
  PyDict_SetItemString(d, "error", cdd_error);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module _cdd");
}
#endif


