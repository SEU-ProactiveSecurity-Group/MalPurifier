# MalPurifier: Enhancing Android Malware Detection with Adversarial Purification Against Evasion Attacks

## Overview

Welcome to the repository for "[MalPurifier](https://arxiv.org/abs/2312.06423)," a research project aimed at improving Android malware detection through adversarial purification techniques designed to counter evasion attacks. This work seeks to strengthen security on Android platforms by addressing sophisticated attack strategies.

## Project Status

- **Under Review**: The research paper titled "MalPurifier: Enhancing Android Malware Detection with Adversarial Purification Against Evasion Attacks" is currently under submission for peer review.
- **Ongoing Development**: This repository will be regularly updated with code, documentation, and other resources as the project progresses.

## Citation

```
@article{zhou2023malpurifier,
  title={MalPurifier: Enhancing Android malware detection with adversarial purification against evasion attacks},
  author={Zhou, Yuyang and Cheng, Guang and Chen, Zongyao and Yu, Shui},
  journal={arXiv preprint arXiv:2312.06423},
  year={2023}
}
```

## Disclaimer

The source code and specific methodologies of "MalPurifier" are currently withheld due to the innovative and confidential nature of the ongoing research. Full disclosure, including the public release of the code, will be considered following the completion of the review process and the paper's potential publication.

## Dataset

We conduct our experiments on two primary datasets: **Drebin** and **Androzoo**. Both datasets require users to comply with their respective policies to obtain the APK files. The `sha256` checksums for the apps in these datasets are available in the `dataset` directory. APKs can be downloaded directly from Androzoo and Drebin.

To reproduce the experimental results on the **Drebin** or **Malscan** datasets, we provide a portion of intermediate files (e.g., vocabulary, dataset splitting info, etc.), which are available in [dataset](./dataset/). However, please note that data preprocessing is still required, meaning you will need to download the necessary APKs and follow the preprocessing steps before running the experiments. This is crucial for generating realistic attack scenarios.

For more details on dataset construction, refer to the code in `core/defense/dataset.py`. To generate feature vectors, please refer to the code located in `core/droidfeature`. Additionally, we have provided some example feature files in the `core/droidfeature` directory for your reference.

## Contact

For more information or inquiries, feel free to reach out to us at: [yyzhou@seu.edu.cn](mailto:yyzhou@seu.edu.cn).
