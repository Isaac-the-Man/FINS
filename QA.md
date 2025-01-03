# Questions

1. **What conservation problem is being addressed and why is it important?**

   There are more than 33k known fish species as of 2023, with roughly 250 new species being discovered every year. This poses a challenge for an Autmatic species Identification System (AIS), since it requires sufficient data to train such a network, which we don't always have. Having a good AIS is crucial to the convservation of these marine species as it provides insights on population, biodiversity, migration patterns, and etc.. This project aims to solve the data deficiencies aspect by generating synthetic images for novel marine species with minimal data.  

2. **Why is AI needed to solve the problem? What are the non-AI alternatives?**

   Besides manually collecting more data (which requires funding, experts, and equipments), the most popular approach is generating more data via data augmentation techniques such as blurring, rotation, mirroring, and etc.. This however, does not provide enough diversity for the models. An deep learning based approach using generative models can better address this problem by generating an infinite amount of diverse synthetic images (in theory).

3. **What datasets are available to develop and test a solution (and in what format)? Are they available for download via API? What is their provenance, meaning under what conditions were they collected and how consistent are these with CARE principles? What dataset(s) did you choose and why?**

   A wide collection of images from different fish species is required. Currently the best dataset we found so far is the Wildfish++ dataset, which is available on BaiduNet (requires a Chinese phone number to register), but we contacted the authors and they gave us a link to a private Google Drive which is not public to the internet. The datasets consists of 10k+ images from more than 2k fish species. This is our choice of dataset as the rest of the dataset on the internet are either focused on a specific fish species or simply does not have high quality images. The data collection process for the dataset involves scraping images from Google/Flickr, and some professional fish databases. For the compliance of the FAIR principles, the link to the paper and the datasets is semi-pubicly available (reasons stated earlier) and fully managed by the authors. All images are clearly labeled and in JPEG format. However, the paper does not acknowledge any licensing of the images. As for the CARE compliance, the dataset is for research purposes that benefits the general public. However, the sources of the images are untraceable as it is scraped from the internet, therefore the ethical implications remains un-adressed. 

4. **What annotations on the data are available? Did you have to get more, and if so, how?**

   The only annotations that is useful to us is the species label, which the dataset provides. There are also some text descriptions available for easily confused species, but we do not utilize this in our project.
   To test the ability of our model to adapt to novel species (not included in Wildfish++) in a few-shot setting, we had to manually collect images from the internet (sources includes articles, Google Images, Journals, etc.), sources of these images are acknowledged.

5. **Evaluate the data: what are its statistical properties? What does it look like? Is it truly representative of the conservation problem that must be solved?**

   The datasets consists of 10k+ images from more than 2k fish species, averaging about 40 images per species. The species of choice includes a wide variety of fishes from different habitats around the world. This is comprehensive enough for our model to learn a good idea about what a fish should look like. However, there are many artificial settings within the database (fish out of water, humans holding fish, drawings of a fish, etc.), so we had to manually clean a subset of the images ourselves before feeding it into the model. 

6. **What baseline did you establish using existing trained models? What AI techniques did you develop and use and how much better are they than the baseline? These should be explained in significant detail.**

   There are no significant baseline generative models for the task of fish image generation currently, so we trained a Deep Convolutional GAN (DCGAN) to serve as our baseline. 
   
   LAWRENCE MIAOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
   DESCRIBE DCGAN

   The two-stage lightweight GAN approach we proposed consists of a pre-trained model that generalize well to a variety of fish species and background, while fine-tuning on the model results in a species-specific model that can be finetuned in a few-shot setting using as little as 6 images. 

   The lightweight GAN architecture proposed in the paper *Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis* introduces significant architectural novelties that achieve high-quality image synthesis while minimizing computational complexity and memory usage. A key innovation is the share discriminator, which processes multi-resolution input images simultaneously. Leveraging shared features at different scales, this approach enhances generalization and eliminates the need for separate discriminators for each resolution, significantly reducing the model’s parameter count and computational overhead. Additionally, the model employs depth-wise separable convolutions, which split standard convolution operations into a depth-wise convolution followed by a point-wise convolution. This drastically reduces computational cost while retaining the network’s representational power, making the GAN more suitable for deployment on resource-constrained devices (shared GPU in our case). 

   Another standout feature is the skip-layer excitation (SLE) mechanism, which allows high-level semantic information from deeper layers to modulate earlier layers in the generator. This preserves global structure and fine details, enhancing the quality of synthesized images. Lightweight GAN also stabilizes training through a dynamic weighted feature matching loss, which adjusts feature alignment weights dynamically. This loss encourages better alignment between generator and discriminator features, improving image quality and reducing the risk of mode collapse.
   By combining these innovations, Lightweight GAN achieves remarkable efficiency, with fewer parameters and lower computational requirements compared to traditional GAN architectures like StyleGAN2. Having fewer parameters is also favorable for few-shot image generation tasks, which is what we aim to achieve. 

   It is hard to quantify the results of a generative model. Instead, we can visually compare the results of our DCGAN and lightweight GAN. Since our DCGAN doesn't even converges, we can conclude that our lightweight GAN performs significantly better than our baseline model. As an experiment we also visually compared our lightweight GAN with DALLE-3 prompted with text description and image samples to perform zero-shot in-context learning. Our lightweight GAN approach is still signficantly better visually. See section 9 below for a visual comparison.

7. **Please describe the training, validation and testing that you did. This also requires significant detail.**

   1. **Diffusion**

      JONATHAN LIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
      DESCRIBE DIFFUSION

   2. **Lightweight GAN**

      The training of lightweight GAN consists of two parts: pre-training and finetuning. 

      The pre-training phase ensures the model learns about good features of various fish species in general (body shape, different habitat settings, poses, point of view, etc..). This is crucial as without the pretraining step, it is harder for the model to converge in a few-shot setting. In our experiments we tried directly training the model from scratch on a target species in a few-shot setting (~10 images), but the model is unable to converge. We can think of the pre-training phase as teaching the model "what is a fish". We pre-trained a lightweight GAN on 20 randomly selected species from the Wildfish++ dataset (manually cleaned and cropped by us) for 35000 epochs, which took about 55 hours. The output image size is 128 x 128, with a latent dimension of 256. Model is trained using the Adam optimizer with a learning rate of 2e-4. Batch size is set to 64 and gradient penalty weight is set to 10. Data augmentation techniques used includes random brightness, random contrast, random saturation, random offset, and random cropout. The augmentation probability is set to 30%. During validation (every 1k epoch), the model is tasked to perform masked image modeling, where an patch of the ground truth is removed, and the model must learn how to in-paint the missing patch back to the original image. These interpolated images are then inspected visually to determine the quality. The result is a pre-trained model that can generate synthetic images of 20 random fish species in a diverse background. 

      The finetuning phase allows the model to adapt to the target species, while still keeping the good features learned during pretraining. Regularization techniques described in the paper *Few-shot Image Generation with Elastic Weight Consolidation* is used to prevent catastrophic forgetting of features learned in the pre-training phase. The pre-trained model is finetuned separately on 4 target species that it has not seen in pre-training before in a few-shot setting. The target species we chose includes 3 well-known species (*Anomalops katoptron*, *Haemulon carbonarium*, and *Liopropoma eukrines*) and 1 novel fish species discovered in 2023 (*Sinocyclocheilus longicornus*). Most hypereparamters are kept the same as pre-training, except batch size and learning rate is lowered to 4 and 1e-4 respectively. The same validation technique in pre-training is also employed. All target species are finetuned using 2000 epochs, which took half an hour on average.

      There are no testing set for our lightweight GAN as evaluating a image generation model is a complex task. Common metrics such as the *Fréchet Inception Distance (FID)* doesn't make sense in a low-data setting since it relies on statistical robustness. Instead, we visually inspect the generated images to manually evaluate on both the quality and diversity of the generated images. 

8. **How might the solution be used to help address the conservation problem? Is the solution accurate enough? How would you employ human guidance to improve, correct, or explain your necessarily-imperfect solution:**

   The synthetic images generated by our model can be used as training materials for an AIS. This should theoretically improve the accuracy of the system, allowing a more precise analysis of fish population, diversity, migration patterns, and etc.. Currently our model is only a prove on concept (trained on a subset of the data and on 128 x 128 resolution) so the diversity and resolution is very limited. It is hard to evaluate the quality of these synthetic images quantitatively, but human experts can be guided to hand-pick good synthetic images to ensure quality control. 

9. **In what ways does this problem challenge the state of the art in AI? To answer this you might need to think about how people will learn to trust your system and use it in new ways.**

   There are no current SOTA generative models specifically for such a task (fish image generations), and training / finetuning the current SOTA generative models are out of scope due to the high resource required (and often has a paywall). Instead, we followed a zero-shot and  in-context learning approach to enter some description and sample images for these models. The generated images are often not realistic and does not visually resemble the target species. Here's an example generate by DALLE-3 given text description and sample images of the novel species *Sinocyclocheilus longicornus*:
  
   ![image-20241201185417631](/media/image-20241201185417631.png)
  
   Our model clearly has better results in terms of realism and the reconstruction of key features of the species.

10. **How might your system be deployed and what are the logistical challenges in doing so?**

   The System can be deployed to any servers that has a high-end GPU. This is acceptable as we do not need any real-time inferences nor do we have to deploy the model on an edge device. An ideal workflow would be to have the model deployed on a server, human experts will then manually collect high quality images for novel fish species of desired, and finetune the model on the target species. The model can then be used to generate synthetic images of the target species, which will later be fed into the training pipeline of the AIS system. 

   The main challenge to deploying our system to production is the required GPU resources, since our model currently stands at 92 million parameters to generate a 128 x 128 resolution images. For higher resolutions (1024 x 1024), which is typically what you want to do in an production setting, this will push the model to roughly 150M paramters. Therefore, a high-end GPU is required for both training and finetuning the model. 

   Another challenge to deploying or model is to include a much larger dataset for the pretraining phase. Currently, due to resource and time constraints, this project only uses 20 species for pre-training of a 128 x 128 generative model as a prove of concept. At production we would ideally want a model that incorporates more diverse species and generates higher resolution images.

11. **What issues are associated with long-term maintenance of your system, further development, and new application domains?**

   Ideally we would want to update the pretrained weights periodically to include more species every once in a while, but this is not always possible as obtaining these images are not trivial, and re-training the model is a costly procedure. Potential future development directions includes the optimization of the model and also ideally including an online learning paradigm to reduce cost of adding a new species. The model can also be applied to a more general domain to include other marine life-form that is not a fish (mollusks, whales, exotic fishes, etc..), with the main challenge being having enough data. 

12. **How are the data, models and weights being shared, and is this in accordance with FAIR principles?**

   Code for the model is avialable on our Github. Our own cleaned subset of Wildfish++ and model weights is available on a public Google Drive link. These artifacts shared by us is mostly complied with the FAIR principles, with the exception of the licensing of images within the datasets, which is derived from unknown sources from the Wildfish++ dataset.

13. **What other ethical, privacy and security implications do you foresee in using your system? How can they be addressed.**

   There are no obvious ethical, privacy, and security concerns for using our system. However, the system is built upon a dataset that has questionable CARE practicies (unknown / unacknowledged sources). There is no quick solution to this besides collecting a new subset of images that follow the principles, which is out of scope for this project.

14. **Finally, please outline the individual contributions of each member of your team. Be as specific as you can. Everyone should contribute both technical content and application content.**

   - Lawrence Miao (undergrad)
      - Dataset EDA
      - Baseline Models
         - WGAN
         - DCGAN
   - Jonathan Li (grad)
      - Diffusion Models
         - Pre-eliminary training a diffusion model on a single species (great white shark)
         - Conditional Diffusion
   - Yu-Kai "Steven" Wang (grad)
      - Manually cleaning / cropping on the subset of Wildfish++
      - Collection and cleaning / cropping of images of the novel species *Sinocyclocheilus longicornus*.
      - GAN Models
         - CNN-based WGAN with gradient penalty (single species *Carcharodon carcharias*)
         - Conditional WGAN (single species *Carcharodon carcharias*)
         - Pretraining of lightweight GAN (20 randomly selected species)
         - Implementation of regularization techniques for finetuning a GAN
         - Finetuning of lightweight GAN (4 species unseen in training, including 3 well-known species and 1 novel marine species *Sinocyclocheilus longicornus*)