%if cuda:
	#define INTERNAL_FUNC inline __device__
	#define GLOBAL_MEM
	#define EXPORTED_FUNC __global__
	#define SYNC __syncthreads()
	#define SHARED_MEM __shared__

	#define CUDA
%else:
	#define INTERNAL_FUNC
	#define GLOBAL_MEM __global
	#define EXPORTED_FUNC __kernel
	#define SYNC barrier(CLK_LOCAL_MEM_FENCE)
	#define SHARED_MEM __local
%endif

<%
	scalar = 'double' if double else 'float'
	complex = 'double2' if double else 'float2'
%>

#define SCALAR ${scalar}
#define COMPLEX ${complex}

%if cuda:
	## multiplication + addition
	#define mad(x, y, z) ((x) * (y) + (z))

	## integer multiplication
	//#define mul24(x, y) __mul24(x, y)
	#define mul24(x, y) ((x)*(y))
%endif

INTERNAL_FUNC ${complex} complex_ctr(${scalar} x, ${scalar} y)
{
%if cuda:
	return make_${complex}(x, y);
%else:
	return (${complex})(x, y);
%endif
}

## These operators are supported by OpenCL
%if cuda:
	INTERNAL_FUNC ${complex} operator+(${complex} a, ${complex} b) { return complex_ctr(a.x + b.x, a.y + b.y); }
	INTERNAL_FUNC ${complex} operator-(${complex} a, ${complex} b) { return complex_ctr(a.x - b.x, a.y - b.y); }
%endif

INTERNAL_FUNC ${complex} complex_mul_scalar(${complex} a, ${scalar} b)
{
	return complex_ctr(a.x * b, a.y * b);
}

INTERNAL_FUNC ${complex} complex_mul(${complex} a, ${complex} b)
{
	return complex_ctr(mad(-a.y, b.y, a.x * b.x), mad(a.y, b.x, a.x * b.y));
}

INTERNAL_FUNC ${complex} complex_inv(${complex} a)
{
	${scalar} module = a.x * a.x + a.y * a.y;
	return complex_ctr(a.x / module, - a.y / module);
}

INTERNAL_FUNC ${complex} complex_div(${complex} a, ${complex} b)
{
	return complex_mul(a, complex_inv(b));
}

INTERNAL_FUNC ${scalar} squared_abs(${complex} a)
{
	return a.x * a.x + a.y * a.y;
}

INTERNAL_FUNC ${complex} conj(${complex} a)
{
	return complex_ctr(a.x, -a.y);
}

INTERNAL_FUNC ${complex} cexp(${complex} a)
{
	${scalar} module = exp(a.x);
	${scalar} angle = a.y;
	${scalar} cos_a, sin_a;

%if cuda:
	${"sincos" + ("f" if scalar == "float" else "")}(angle, &sin_a, &cos_a);
%else:
	cos_a = native_cos(angle);
	sin_a = native_sin(angle);
%endif
	return complex_mul_scalar(complex_ctr(cos_a, sin_a), module);
}

%if not cuda:
	#define sin native_sin
	#define cos native_cos
	#define exp native_exp
%endif

%if cuda:
	#define THREAD_ID_X threadIdx.x
	#define BLOCK_ID_X blockIdx.x
	#define GLOBAL_ID_X (threadIdx.x + blockDim.x * blockIdx.x)
	#define GLOBAL_SIZE_X (blockDim.x * gridDim.x)

	#define THREAD_ID_Y threadIdx.y
	#define BLOCK_ID_Y blockIdx.y
	#define GLOBAL_ID_Y (threadIdx.y + blockDim.y * blockIdx.y)
	#define GLOBAL_SIZE_Y (blockDim.y * gridDim.y)

	#define GLOBAL_ID_FLAT threadIdx.x + blockDim.x * blockIdx.x + gridDim.x * blockDim.x * blockIdx.y
	#define BLOCK_ID_FLAT blockIdx.x + gridDim.x * blockIdx.y
%else:
	#define THREAD_ID_X get_local_id(0)
	#define BLOCK_ID_X get_group_id(0)
	#define GLOBAL_ID_X get_global_id(0)
	#define GLOBAL_SIZE_X get_global_size(0)

	#define THREAD_ID_Y get_local_id(1)
	#define BLOCK_ID_Y get_group_id(1)
	#define GLOBAL_ID_Y get_global_id(1)
	#define GLOBAL_SIZE_Y get_global_size(1)

	#define GLOBAL_ID_FLAT get_global_id(0)
	#define BLOCK_ID_FLAT get_num_groups(0)
%endif

${prelude}

%if cuda:
	extern "C" {
%endif

${kernels}

%if cuda:
	} // extern "C"
%endif
