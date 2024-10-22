# Project & Team Proposal

FINS: Few-shot Image Generation of Novel Marine Species

### Team Members

- Graduate Team Captain: Yu-Kai "Steven" Wang (wangy80)
- Graduate Member: Jonathan Li (lij47)
- Undergraduate Member: Lawrence Miao

### Descriptions

**What is the Conservation Challenge and why is it important?**

There are more than 33,000 known fish species, with 250 new species being discovered every year. Despite most of the fish species are not endangered, they made up the majority of the biomass in the ocean, which covers roughly 70% of the Earth's surface. Fishes provide food sources, biodiversity, and huge economic value, therefore having a better understanding of these creatures is crucial for both humans and the health of the ecosystem in general. Since there are a huge number of different fish species with more being discovered every year, humans often relies on Automatic Species Identification Systems (AIS) to distinguish them. Having a good AIS for fish can provide valuable insights into the population, biodiversity, migration pattern, and much more. However, it can be challenging to train such a system since as little as 10 images may be available for each species. We can say that the limitation of our current AIS system is limited by the amount of annotated data that humans can gather by hand.   

**What are the data sources and what is needed to acquire it, process it, and annotate it?**

The [WildFish++](https://github.com/PeiqinZhuang/WildFish) dataset contains more than 2k annotated fish images (at least 10+ images per species) for fish recognition in the wild. Each fish contains a morphology diagnosis and a biological description. We can utilize the images within the dataset to train our model without further annotation.

**How might the problem be formulated and what technical approaches might be used?**

To solve the data shortage problem, we propose the machine learning framework "FINS: Few-shot Image Generation of Novel Marine Species" to generate diverse synthetic images of a given fish using a couple of images. We plan on utilizing Conditional Generative Adversarial Networks (CGANs) as our main model architecture, using a image-based encoder to create a latent representation of the fish which can then be used as the input for the generative network. To aid in the development of the latent representations, a seperate language based model can be trained to encode the text descriptions of each species of fish. A clip based architecture could be used to further finetune the image/text embedding pairs to develop a robust and meaningful latent representation for each fish. The model will first be pretrained on a subset of the dataset for the purpose of learning the features and representation of what a fish is. Once the model successfully generates high quality fishes, we will train the model to conditionally generate specific species of fish using the latent representation as the conditional prior. The performance of our CGANs will be evaluated using the GAN-Train GAN-Test evaluation method [How good is my GAN?](https://arxiv.org/pdf/1807.09499), which essentially takes a subset of the real images, and a subset of the fake images, trains a classifier on the real data and evaluated on the fake data (GAN Test), and trains another classifier on the fake data and evaluated on the real data (GAN Train). The idea here is that if the CGAN's is successful, real and fake images should be apart of the same distribution. This would mean that the classifiers should generalize well on the evaluation datasets. Our work will mainly focus on species with very little sightings to imitate real-world use case (Few-Shot image generation). Some potential technical difficulties that we might face are: 1. Model is unable to generate good quality images with little data. 2. Hallucination of features that is not present in the species. 3. Unable to distinguish between highly-confused species.  

**How much could a group accomplish in a semester?**

Realistically, it should be feasable for us to be able to train a CGANs to produce "high-quality" images of select fish species using minimal training examples by the end of December assuming our methodology is correct. 