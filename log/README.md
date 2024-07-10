# log
- 20240312_154908.log and 20240312_164455.log use $\lambda_1$=0.75 and $\lambda_2$=0.5
- 20240623_111306.log uses both $\lambda_1=0.1$ and $\lambda_2$=0.1, the best mIoU is 86.685.

# Experiment
## Train LandSat08 China dataset
```sh
bash main_demo.sh 0
```

## Test three datasets
```sh
bash LC08_test.sh
```

- [CHLandSat8](https://github.com/HaiLei-Fly/CHLandsat8)
- [38 cloud](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
	- [mask](https://github.com/kunzhan/HR-cloud-Net/blob/main/dataset/38_large_gt.tar.gz)
- [SPARCS](https://emapr.ceoas.oregonstate.edu/sparcs/)

# Citation
We appreciate it if you cite the following paper:
```
@InProceedings{LiJEI2024,
  author =    {Jingsheng Li and Tianxiang Xue and Jiayi Zhao and 
               Jingmin Ge and Yufang Min and Wei Su and Kun Zhan},
  title =     {High-Resolution Cloud Detection Network},
  booktitle = {Journal of Electronic Imaging},
  year =      {2024},
}

```

# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)