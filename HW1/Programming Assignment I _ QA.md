# Programming Assignment I : QA

# Student ID : 311552046

## Q1-1

```
Run ./myexp -s 10000 and sweep the vector width from 2, 4, 8, to 16. Record
the resulting vector utilization. You can do this by changing the #define 
VECTOR_WIDTH value in def.h. 
Q1-1: Does the vector utilization increase, decrease or stay the same as 
VECTOR_WIDTH changes? Why?
```
- vector width = 2 , vector utilization : 86.0%

![](https://hackmd.io/_uploads/SJyYOb4Wp.png)

- vector width = 4 , vector utilization : 80.8%

![](https://hackmd.io/_uploads/rJVJubV-T.png)

- vector width = 8 , vector utilization : 78.2%

![](https://hackmd.io/_uploads/B1u3OZ4ZT.png)

- vector width = 16 , vector utilization : 77.0%

![](https://hackmd.io/_uploads/SJzJYbNZT.png)


==The vector utilization decrease when VECTOR_WIDTH increase==

It is because my code will keep trying to do vector instructions until whole mask's bit become zero, thus if one value's exponent larger than others, those mask's bit which is zero(exponent become zero) won't change the value at that time (waste), so the more larger vector width, the more chance to waste a lot of vector utilization.

---

## Q2-1

```
Q2-1: Fix the code to make sure it uses aligned moves for the best performance.

Hint: we want to see vmovaps rather than vmovups.
```

==Both vmovaps & vmovups are used to moving data between memory/AVX register==

The major different between two instruction is that vmovaps's `a` means aligned & vmovups's `u` means unaligned

Since avx-2 is 256 bit(32 bytes) instruction set extension, so the code need to be fixed to `__builtin_assume_aligned(a, 32)`

When type 
`$ diff assembly/test1.vec.restr.align.s       assembly/test1.vec.restr.align.avx2.s` again, can see that vmovaps instruction appear

![](https://hackmd.io/_uploads/BJz6uM4bp.png)

---

## Q2-2

```
Q2-2: What speedup does the vectorized code achieve over the unvectorized 
code? What additional speedup does using -mavx2 give (AVX2=1 in the 
Makefile)? You may wish to run this experiment several times and take 
median elapsed times; you can report answers to the nearest 100% (e.g., 
2×, 3×, etc). What can you infer about the bit width of the default vector 
registers on the PP machines? What about the bit width of the AVX2 vector 
registers.

Hint: Aside from speedup and the vectorization report, the most relevant 
information is that the data type for each array is float.
```

==modify test1.cpp run 10 times and get its average time==

- unvectorized code : 8.27957 sec

`$ make clean && make && ./test_auto_vectorize -t 1`

![](https://hackmd.io/_uploads/B1cRhbHW6.png)


- vectorized code : 2.63185 sec

`$ make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 1`

![](https://hackmd.io/_uploads/B1imaZSb6.png)

==vectorized code achieve about 314% speedup(over unvectorized code)==

- avx2 support code : 1.40079 sec

`$ make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 1`

![](https://hackmd.io/_uploads/SkhBa-H-6.png)

==avx2 support code achieve about 187% speedup(over vectorized code)==

==Read `test1.vec.restr.align.s` can found vector instruction is align (movaps), and since in test1.cpp all data element is align to 16 byte, so can infer that width of the default vector register on PP machines is 128 bit==

![](https://hackmd.io/_uploads/S1bqMfBb6.png)

==Base on Q2-1's answer, the bit width of the AVX2 vector registers is 256 bit==

---

## Q2-3

```
Now, you actually see the vectorized assembly with the movaps and maxps 
instructions.
Q2-3: Provide a theory for why the compiler is generating dramatically 
different assembly.
```

- Code before patch (assign c[j] before if-condition)

![](https://hackmd.io/_uploads/rJBHuv8WT.png)

- Compiler won't generate movaps or maxps instruction

![](https://hackmd.io/_uploads/ry0OPPUbp.png)


- Code after patch (assign op all execute inside the if-else-condition)

![](https://hackmd.io/_uploads/r1XzuDUZa.png)


- Compiler will generate movaps and maxps instruction

![](https://hackmd.io/_uploads/S1z1YP8WT.png)

==As we can see that in after-patch assembly code, `maxps` will put the larger one into xmm register, and in the end put xmm register's value back to memory==

==Since the before-patch code will follow the order to deal with `c[j] = a[j];` first(use movl to put `a[j]` into `%edx` register), if it use `maxps` to handle condition, when the condition is false, it will put `a[j]` into xmm register again, compiler know it is redundant, so it won't use `maxps` to handle condition, by the way use the branch to jump into condition body.==








