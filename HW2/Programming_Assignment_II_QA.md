# Programming Assignment II : QA

- Student ID : 311552046

## Q1

```
Q1: In your write-up, produce a graph of speedup compared to the reference
sequential implementation as a function of the number of threads used FOR 
VIEW 1. Is speedup linear in the number of threads used? In your writeup 
hypothesize why this is (or is not) the case?
```

- My distribution method

```c=1
void workerThreadStart(WorkerArgs *const args)
{
    int size = args->height / args->numThreads;
    int start_row = args->threadId * size
    mandelbrotSerial(..., start_row, size,...);
}
```

- VIEW 1

![](https://hackmd.io/_uploads/B1UAIQ9fT.png)

![](https://hackmd.io/_uploads/BJMQvX9GT.png)

- VIEW 2

![](https://hackmd.io/_uploads/SkNYvQ5GT.png)

![](https://hackmd.io/_uploads/BJ22w7cz6.png)

==In VIEW 1, speedup not linear in the number of threads used==

==In VIEW 2, speedup linear in the number of threads used==

==The reason may cause this result is that VIEW 1 and VIEW 2's light region different(light point means more compute).==
  
![](https://hackmd.io/_uploads/HySNtm5Ma.png)

==Take VIEW 1 as example, we can see that thread1's workload is in image's central part, which is larger than thread0, thread2, I think it is the reason that performance loss, since in workers.join(), threads need to wait the lowest thread finish its work.==

==And VIEW 2's light region is disperse, so each thread's workload will similar.==

![](https://hackmd.io/_uploads/HJRxMQqfT.png)

---

## Q2

```
 Q2: How do your measurements explain the speedup graph you previously
 created?
```

- Result

![](https://hackmd.io/_uploads/BkFf275z6.png)

==The result can tells that in different threads number, the total execute time is almost equal to the lowest thread's execute time. It can prove our conclusion in Q1 is correct.==

---

## Q3

```
Q3: In your write-up, describe your approach to parallelization and 
report the final 4-thread speedup obtained
```

- My distribution method

```c=1
void workerThreadStart(WorkerArgs *const args)
{
    int size = 1;
    for(int i = args->threadId;i < args->height;i += args->numThreads)
    {
        mandelbrotSerial(..., i, size,...);
    }
}
```

==The idea here is to try as much as possible to let every thread's workload balance, so we tile the image's height, and each tile distribution to all threads we have.==

:::info
Note here we had set tile size be 2, 3, 4, 8's mulitple, but the problem will occure when thread number be 5, 6, 7... , because the tile size can't exact divide thread number, so at the end, we set tile size equal thread number we have(make sure tile size is thread number's multiple).
:::

==Take VIEW 1 as example, different color represent different thread, region in each color means the thread's workload, when the tile size small enough, each thread's workload will more balance.==

- VIEW 1

![](https://hackmd.io/_uploads/H1uzZr5f6.png)

==The result in both VIEW can get about 3.7 speedup==

- Result

![](https://hackmd.io/_uploads/HybweE9fa.png)

---

## Q4

```
Q4: Now run your improved code with eight threads. Is performance 
noticeably greater than when running with four threads? Why or why not?
(Notice that the workstation server provides 4 cores 4 threads.)
```

- Result

![](https://hackmd.io/_uploads/SJxdbVqM6.png)

==We can see that speedup can't get 7.X, I think the reason is that workstation only have four cores four threads, so when threads number is 8, each core will have more than one threads, and context switch will occured in each core, the context switch overhead may lose performance.== 










