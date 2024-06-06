# ML2Project
## 1. Project Motivation
   What is the problem you are trying to solve? What is the motivation behind it? Why is your project relevant?
   
   Agricultural productivity is the backbone of global food security, and maize is one of the world's most important crops. However, maize crops are constantly threatened by various     diseases    that can severely affect yield and quality. This is where the classification of maize diseases becomes crucial.
   
### Accurate disease classification in maize is important for several reasons:

**1. increasing crop yield:**
early and accurate identification of diseases enables timely intervention, which can significantly reduce crop losses. By accurately classifying diseases, farmers can apply targeted treatments, ensuring healthier harvests and higher yields.

**2. sustainable agriculture:**
disease classification helps in the adoption of sustainable farming practices. By understanding the specific diseases that affect corn, researchers and farmers can develop environmentally friendly treatments that reduce reliance on chemical pesticides and promote biodiversity.

**3. economic impact:**
Corn is an important economic crop for many countries. Proper disease management through accurate classification can prevent devastating economic losses, ensure the stability of the agricultural sector and secure farmers' livelihoods.

**4. promotion of scientific research:**
Disease classification of maize is essential for scientific research and contributes to the development of resistant plant varieties. This research is critical to long-term food security, especially in the face of climate change and evolving pathogens.

**5. global food security:**
Maize is a staple food for millions of people around the world. Effective disease management ensures a steady supply of this important crop and plays a critical role in combating hunger and ensuring food security worldwide.

In summary, the classification of diseases in maize is not just a scientific endeavor, but a task with far-reaching implications for agriculture, the economy and global food security.     By investing in advanced data science techniques for maize disease classification and management, we can pave the way for a more resilient and sustainable future.

## 3. Data Collection or Generation

Dataset Source: [OSF](https://osf.io/s6ru5/files/osfstorage#) **Important: _only download "Dataset_Original" Folder from "CD&S" Folder._**


## 4. Modeling

I decided to use a pre-trained model from Keras for the task and to fine-tune it. For the base model I use the EfficientNetB0 model which was developed by Google with 1.2 million images in 1000 classes. It is already quite accurate with a top-1 accuracy of 77.1%. The model has about 5.3 million parameters and an input resolution of 224x224 pixels (Documentation: [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function))

## 5. Interpretation and Validation
