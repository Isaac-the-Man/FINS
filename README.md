# Project & Team Proposal

FINS: Few-shot Image Generation of Novel Marine Species

### Team Members

- Graduate Team Captain: Yu-Kai "Steven" Wang (wangy80)
- Graduate Member: Jonathan Li (li4j)
- Undergraduate Member: Lawrence Miao

### Descriptions

**What is the Conservation Challenge and why is it important?**

There are more than 33,000 known fish species, with 250 new species being discovered every year. Despite most of the fish species are not endangered, they made up the majority of the biomass in the ocean, which covers roughly 70% of the Earth's surface. Fishes provide food sources, biodiversity, and huge economic value, therefore having a better understanding of these creatures is crucial for both humans and the health of the ecosystem in general. Since there are a huge number of different fish species with more being discovered every year, humans often relies on Automatic Species Identification Systems (AIS) to distinguish them. Having a good AIS for fish can provide valuable insights into the population, biodiversity, migration pattern, and much more. However, it can be challenging to train such a system since as little as 10 images may be available for each species. We can say that the limitation of our current AIS system is limited by the amount of annotated data that humans can gather by hand.   

**What are the data sources and what is needed to acquire it, process it, and annotate it?**

The [WildFish++](https://github.com/PeiqinZhuang/WildFish) dataset contains more than 2k annotated fish images (at least 10+ images per species) for fish recognition in the wild. We can utilize the images within the dataset to train our model without further annotation. 

**How might the problem be formulated and what technical approaches might be used?**

To solve the data shortage problem, we propose a machine learning framework "FINS: Few-shot Image Generation of Novel Marine Species" to generate new images of fish species given only a few real images for feature extraction. We plan on adopting the Conditional GAN as our main model architecture, using a vision encoder as the input feature condition for the generative network. The model will first be pretrained on the dataset to learn how to generate a good image of a fish. Our work will mainly focus on species with very little sightings to imitate real-world use case (Few-Shot image generation). Some potential technical difficulties that we might face are: 1. Model is unable to generate good quality images with little data. 2. Hallucination of features that is not present in the species. 3. Unable to distinguish between highly-confused species.  

**How much could a group accomplish in a semester?**

We'll l hopefully be able to train a CGAN to effectively generate images of a selected few species with minimal training examples by the end of December. A discriminator for real and fake images can be trained to evaluate the quality of our generated images.