#include "PPintrin.h"
#include <iostream>

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
	__pp_vec_float val_vec , tmp_val_vec;
	__pp_vec_int exp_vec , tmp_exp_vec;
	__pp_mask mask_range , mask_exp_zero , mask_exp_not_zero;
	__pp_vec_int zero_vec = _pp_vset_int(0);	//set all vector equal 0
	__pp_vec_int one_vec = _pp_vset_int(1);		//set all vector equal 1
  	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{
		if((i + VECTOR_WIDTH) > N)
		{
			mask_range = _pp_init_ones(N % VECTOR_WIDTH);	//1110 0000
		}
		else
		{
			mask_range = _pp_init_ones();					//1111 1111 (default size = VECTOR_WIDTH)
		}
		_pp_vload_float(val_vec, values + i, mask_range);						//float x = values[i];
		_pp_vload_int(exp_vec, exponents + i, mask_range);						//int y = exponents[i];
		_pp_veq_int(mask_exp_zero, exp_vec, zero_vec, mask_range);				//if (y == 0)
		_pp_vset_float(tmp_val_vec, 1.f, mask_exp_zero);							//output[i] = 1.f;
		mask_exp_not_zero = _pp_mask_not(mask_exp_zero);
		mask_exp_not_zero = _pp_mask_and(mask_range, mask_exp_not_zero);		//else
		_pp_vmove_float(tmp_val_vec, val_vec, mask_exp_not_zero);					//float result = x;
		_pp_vsub_int(exp_vec, exp_vec, one_vec, mask_exp_not_zero);					//y - 1;
		_pp_vmove_int(tmp_exp_vec, exp_vec, mask_exp_not_zero);						//int count = y - 1;
		_pp_veq_int(mask_exp_zero, tmp_exp_vec, zero_vec, mask_exp_not_zero);		/* check whether have zero element */
		mask_exp_zero = _pp_mask_not(mask_exp_zero);
		mask_exp_not_zero = _pp_mask_and(mask_exp_not_zero, mask_exp_zero);			/* update mask_exp_not_zero */
		while(_pp_cntbits(mask_exp_not_zero))										//while (count > 0)
		{
			_pp_vmult_float(tmp_val_vec, tmp_val_vec, val_vec, mask_exp_not_zero);		//result *= x;
			_pp_vsub_int(tmp_exp_vec, tmp_exp_vec, one_vec, mask_exp_not_zero);			//count--;
			_pp_veq_int(mask_exp_zero, tmp_exp_vec, zero_vec, mask_exp_not_zero);		/* check whether have zero element */
			mask_exp_zero = _pp_mask_not(mask_exp_zero);
			mask_exp_not_zero = _pp_mask_and(mask_exp_not_zero, mask_exp_zero);			/* update mask_exp_not_zero */
		}

		__pp_mask mask_max_val;
		__pp_vec_float max_val_vec = _pp_vset_float(9.999999f);
		_pp_vgt_float(mask_max_val, tmp_val_vec, max_val_vec, mask_range);			//if (result > 9.999999f)
		_pp_vset_float(tmp_val_vec, 9.999999f, mask_max_val);							//result = 9.999999f;
		_pp_vstore_float(output + i, tmp_val_vec, mask_range);						//output[i] = result;
	}
	return;
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
	__pp_mask mask_range;

	if(N == VECTOR_WIDTH)		//last vector (all tmp_sum are in this vector)
	{
		__pp_vec_float val_vec;
		mask_range = _pp_init_ones();
		_pp_vload_float(val_vec, values + 0, mask_range);
		while(1)				//accumulate last vector , put sum in the first half of vector
		{
			_pp_hadd_float(val_vec, val_vec);
			_pp_interleave_float(val_vec, val_vec);
			N /= 2;
			if(N == 1)
			{
				break;
			}
		}
		_pp_vstore_float(values + 0, val_vec, mask_range);
		return *(values + 0);	//sum is put in the first element
	}

	__pp_vec_float val_vec1 , val_vec2;

	int total_vec = (N/VECTOR_WIDTH);
	int need_to_sum = total_vec/2;
	int still_need_to_add = (total_vec % 2 == 0) ?(total_vec / 2) :(total_vec / 2 + 1);
	still_need_to_add *= VECTOR_WIDTH;	//first half of all element

	for(int i = 0;i < need_to_sum;i++)
	{
		int offset1 = i * VECTOR_WIDTH;
		int offset2 = still_need_to_add;
		mask_range = _pp_init_ones();
		_pp_vload_float(val_vec1, values + offset1, mask_range);
		_pp_vload_float(val_vec2, values + offset1 + offset2, mask_range);
		_pp_vadd_float(val_vec1, val_vec1, val_vec2, mask_range);
		_pp_vstore_float(values + offset1, val_vec1, mask_range);
	}

	return arraySumVector(values, still_need_to_add);	//keep accumulate all element until leave one vector
}
