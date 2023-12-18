# ArtiFact datasheet

ArtiFact is a *A Large-Scale Dataset with Artificial and Factual Images for Generalizable and Robust Synthetic Image Detection* [[GitHub](https://github.com/awsaf49/artifact), [Kaggle](https://www.kaggle.com/datasets/awsaf49/artifact-dataset)].

The training dataset comprises 8 sources carefully chosen to ensure diversity and includes images synthesized from 25 distinct methods, including 13 Generative Adversarial Networks (GANs), 7 Diffusion, and 5 other miscellaneous generators. It contains 2,496,738 images, comprising 964,989 real images and 1,531,749 fake images. Image resolution is 200x200px.

## Motivation

**Q. For what purpose was the dataset created?**

**A.** Synthetic image generation has opened up new opportunities but has also created threats in regard to privacy, authenticity, and security. Detecting fake images is of paramount importance to prevent illegal activities, and previous research has shown that generative models leave unique patterns in their synthetic images that can be exploited to detect them. However, the fundamental problem of generalization remains, as even state-of-the-art detectors encounter difficulty when facing generators never seen during training.

The purpose is to assess the generalizability and robustness of synthetic image detectors. 

**Q. Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset?**

**A.** The dataset was created by a group of researchers at BUET (Bangladesh University of Engineering and Technology). It was created in accordance with in accordance with the
[IEEE VIP Cup 2022](https://grip-unina.github.io/vipcup2022/) standards. This was a competition hosted by the IEEE Signal Processing Society, University Federico II of Naples (Italy) and NVIDIA (USA). The aim is to distinguish real versus AI-based content in images. It's not clear who funded the dataset collection and research jobs.

 
## Composition

**Q. What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?**

**A.** The dataset includes a diverse collection of real images from multiple categories. The most frequently occurring categories in the dataset are Human/Human Faces, Animal/Animal Faces, Vehicles, Places, and Art.

**Q. How many instances of each type are there?**

**A.** There are 2,496,738 images in total, comprising 964,989 real images and 1,531,749 fake images. The categorical distribution can only be inferred for some objects.

**Q. Is there any missing data?**

**A.** Not really. The dataset is comprehensive and doesn't represent a controlled population. 

**Q. Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?**

**A.** No.

## Collection process

**Q. How was the data acquired?**

**A.** Data was acquired from 8 different sources and synthesizes info from 25 distinct methods, including 13 Generative Adversarial Networks (GANs), 7 Diffusion, and 5 other miscellaneous generators.

**Q. If the data is a sample of a larger subset, what was the sampling strategy?**

**A.** **A.** N/A

**Q. Over what time frame was the data collected?**

This is not clear. Year 2022.

## Preprocessing/cleaning/labelling

**A.** Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.

**A.** Don't know.

**Q. Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?**

**A.** Don't know.
 
## Uses

**Q. What other tasks could the dataset be used for?** 

**A.** There might be multiple use cases for a dataset of real and synthetic images. Thanks ChatGPT for the [suggestions](https://chat.openai.com/share/8f32b45c-f915-4c14-937e-fe9567d089d7):

- Media integrity and verification → Journalism and social media
- Security and forensics → Legal proceedings and digital forensics
- Online content moderation → Content platorms and advertising
- Election integrity → Political campaigns
- Educational settings → Academic integrity
- Healthcare imaging → Medical imaging
- Art authentication → Art market

**Q. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?**

**A.** None that I am aware of.

**Q. Are there tasks for which the dataset should not be used? If so, please provide a description.**

**A.** None that I can think of.

## Distribution

**Q. How has the dataset already been distributed?**

**A.** The dataset is available via [[GitHub](https://github.com/awsaf49/artifact) and [Kaggle](https://www.kaggle.com/datasets/awsaf49/artifact-dataset)].

**Q. Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**

**A.** ArtiFact dataset takes leverage of data from multiple methods thus different parts of the dataset come with different licenses. All the methods and their associated licenses are mentioned in the table avaialble in the [README.md](https://github.com/awsaf49/artifact/blob/main/README.md) file of the project.

## Maintenance

**Q. Who maintains the dataset?**

**A.** Judging from the commits, the maintainer is a GitHub user called Awsaf, username [awsaf49](https://github.com/awsaf49), who is a deep learning researcher and competitor.