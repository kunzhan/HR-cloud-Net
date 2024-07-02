# High-Resolution Cloud Detection Network
The complexity of clouds, particularly in terms of texture detail at high resolutions, has not been well explored by most existing cloud detection networks. This paper introduces the High-Resolution Cloud Detection Network (HR-cloud-Net), which utilizes a hierarchical high-resolution integration approach. HR-cloud-Net integrates a high-resolution representation module, layer-wise cascaded feature fusion module, and multi-resolution pyramid pooling module to effectively capture complex cloud features. This architecture preserves detailed cloud texture information while facilitating feature exchange across different resolutions, thereby enhancing overall performance in cloud detection. Additionally, a novel approach is introduced wherein a student view, trained on noisy augmented images, is supervised by a teacher view processing normal images. This setup enables the student to learn from cleaner supervisions provided by the teacher, leading to improved performance. Extensive evaluations on three optical satellite image cloud detection datasets validate the superior performance of HR-cloud-Net compared to existing methods.

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
