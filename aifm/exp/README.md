In this folder, we provide code and scripts for reproducing figures in our submission-version paper. For most figures, there are three lines: `AIFM`, `Fastswap`, and `All in memory`.
`AIFM` stands for our system. `Fastswap` is [a recent system](https://github.com/clusterfarmem/fastswap) published in EuroSys' 20; we use it as a baseline to show the best possible result achieved so far by OS swap-based systems. `All in memory` means running the original program in Linux with enough local memory to accommodate the entire working set; it corresponds to the theoretical performance upper bound.

This artifact is able to (almost) automatically generate `AIFM` and `All in memory` lines, but not the `Fastswap` line. However, as long as you have set up a Fastswap environment, it would be straightforward to reproduce the `Fastswap` line as well; you simply run the original Linux programs of `All in memory` in Fastswap.

The results you get may not be exactly the same as the ones shown in the paper due to changes in system implementation and differences in configurations and machines. However, all results here support the arguments and conclusions we made in the paper.

We have READMEs for each figure folder, please refer to them for further details. We rank the difficulties of reproducing results (ascendingly) as follows and encourage you to go with that order.
```
fig10a --> fig10b --> fig12 --> fig13 --> fig11 --> fig9a --> fig9b --> fig8b --> fig8a --> fig7
```

Cloudlab has a 16-hour instance time limitation. However, it takes longer than 16 hours to reproduce all results here. Therefore you may want to use the `extend` button to extend the instance a bit.

![Extend A Cloudlab Instance](https://i.imgur.com/1p3WpuJ.png)
