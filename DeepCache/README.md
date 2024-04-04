The DeepCache with the proposed TGATE is developed based on [DeepCache: Accelerating Diffusion Models for Free](https://github.com/horseee/DeepCache) 

NOTE:

* `gate_idx` refers to the index of cache update in DeepCache.
For example, if `cache_interval=3` and `gate_idx=3`, the gate step will be `3*(3+1) = 12`. 

* The code runs well in the environment of [DeepCache](https://github.com/horseee/DeepCache).
