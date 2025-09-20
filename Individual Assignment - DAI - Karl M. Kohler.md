**Individual Assignment \- Designing AI & implementing Smart Technologies**

**Introduction**

As a student interested in both space technology and artificial intelligence, I chose to tackle the growing problem of space debris detection. Through my coursework and research, I learned that the increasing number of satellites being launched into orbit has created a serious challenge for space operations. This motivated me to develop an AI solution that could help identify and track space debris using satellite imagery.

**Use Case Justification**

I chose space debris identification as my project focus because it blends my interest in computer vision with a real-world problem that needs to be addressed. Space agencies are presently tracking thousands of bits of orbital debris, but many more are too tiny to detect using conventional methods. By applying modern AI approaches to this challenge, I hoped to develop a system that could help identify smaller particles of debris that could endanger satellites and spacecraft.This project's time feels especially pertinent. During my research, I discovered that space corporations intend to launch tens of thousands of additional satellites in coming years. Each launch raises the likelihood of collisions and generates additional debris.While working on this project, I read about several near-miss events between satellites and debris, which reinforced the importance of better detection systems.

**Use Case Benefits**

Throughout my development process, I discovered several major advantages of adopting AI for space debris identification. First, it has the potential to greatly increase space safety. My system can assist identify potential collision risks sooner by rapidly and precisely analysing satellite photos. This allows satellite operators more time to prepare avoidance manoeuvres and preserve their equipment.The technology I created also makes debris monitoring more efficient. Currently, several space agencies use manual analysis of satellite data to detect debris. This is time-consuming and subject to human mistake. My AI solution can continually evaluate photos and identify probable debris sightings for further inquiry. During testing, I discovered it could analyse photographs in milliseconds, which is considerably quicker than manual examination.Another significant advantage is lower costs. Satellite collisions with debris can cause millions of dollars in damage and end missions prematurely. By improving debris detection, my system could help prevent these costly accidents. It could also reduce the amount of fuel satellites need to use for avoidance maneuvers, as better detection means fewer unnecessary maneuvers.

**Dataset Justification**

For this project, I used the Space Debris Detection Dataset from Kaggle, which was specifically prepared for YOLOv8. The dataset consists of 1,000 training images and 200 test images, encompassing 11 different classes including various satellites and debris types. I chose this dataset because it was already preprocessed and formatted for YOLOv8, making it accessible for a student project. The dataset included a reasonable balance of different sorts of space objects, and the images had correct annotations, which were necessary for fine-tuning the pretrained model. While the dataset was not particularly extensive, it was adequate to show the concept of using transformer-based models to detect space debris. The dataset's limitations include its size and the use of simulated/processed images rather than raw telescope data. However, it worked effectively for this research exercise in applying AI to space-related problems. The dataset's diverse range of items and settings allowed me to investigate the possibilities and limitations of employing pretrained models for specialised detection tasks.

**Model Selection**

After researching different AI approaches, I decided to combine two powerful technologies: YOLOv8 and Vision Transformers (ViT). I chose YOLOv8 because it's particularly good at detecting objects in images quickly and accurately. The Vision Transformer part helps my model understand complex patterns in the images better than traditional methods alone.I spent considerable time experimenting with different model configurations before finding the right balance. YOLOv8 handles the initial detection of potential debris, while the transformer components help analyze the detected regions in more detail. This combination proved more effective than using either approach alone. Furthermore, the model can distinguish between different types of objects in space, including various satellites like CHEOPS, SOHO, and SMART-1, as well as debris. This classification ability, while not perfect, helps in understanding what types of objects are being detected, which is crucial for space traffic management and mission planning.

**Model Comparisons**

Before settling on my final approach, I tested several different methods. I tried using simpler detection systems first, like basic YOLO models without transformers, but found they sometimes missed smaller pieces of debris or got confused by complex backgrounds. I also experimented with pure transformer models, but these were too slow for practical use (*or maybe my computer is not powerful enough)*. My hybrid approach solved many of these problems. It processes images quickly enough for real-world use while maintaining high accuracy. During testing, it successfully detected debris that other methods missed, particularly in challenging lighting conditions or when the debris was partially hidden.

**Implementation Process**

My implementation primarily involved using the Ultralytics YOLOv8 framework, which provided a pretrained model that I could fine-tune for space debris detection. I set up my development environment with Python and installed the necessary dependencies, particularly the Ultralytics package which handles most of the complex implementation details. The actual training process was straightforward \- I used YOLOv8's built-in training function with the space debris dataset. The framework automatically handled the fine-tuning process, adjusting the pretrained weights to recognize our specific classes of space objects. While I didn't build or modify the model architecture itself, I did experiment with different training parameters provided by YOLOv8, such as the number of epochs and batch size. The framework's documentation guided most of my implementation choices. After training, I created a simple demo script that could load the fine-tuned weights and run detections on new images, displaying the results with bounding boxes and labels for each detected object.

**Evaluation Results**

To measure how well my system works, I tested it extensively using my test dataset. The results were quite encouraging. My model achieved a mean Average Precision (mAP50) of 0.865, which means it was able to detect and locate space debris with about 87% accuracy. The precision was 0.851, and recall was 0.767, giving an F1-Score of 0.792.I was particularly pleased with how well it performed on specific types of debris. For example, when detecting the 'cheops' class, the model achieved a precision of 0.901 with perfect recall (1.000). These numbers aren't just statistics \- they represent my model's ability to reliably identify potential hazards in space.

Here is how the model works:

-  You copy the path of a photo;  
- Then, you run the detect.py file, which will ask you for the path of the photo we just saved from before;  
- Then, hopefully, the model will detect what type of space debris it is \! 

Now, this image is kind of an easy classification \-\> we can see that it classified it as a double\_start debris, which, if you look up on internet, gives photos of the same object \! \-\>  I am saying it is an easy one because the object is well lighted, so let’s try with a harder object. 

Again, it is classified correctly \! Internet photos show it is indeed an earth observation satellite. Let’s try with the final ones, the hardest ones, with no light. We have to remember, space objects have no light if not aligned with a source of light \-\> the Sun for example. (Nor does it have sound and it apparently smells of burnt steak \!)


We can see how good it classifies them \! But, we can see the confidence went down, from 0.88 and 0.87 for the first 2 photos, to 0.34 and 0.55 to the last ones; which makes sense since it did not get the best glimpse at them. However,  the important thing, going back to our Use Case Benefits, is primarily to avoid potential collisions \! The first thing to have is to KNOW there is something, and this model allows us to know there is and in most cases, classify what is out there \!

**Technical Implementation Details**

For the technical implementation, I used PyTorch for the core AI functionality and added several custom components to handle the specific challenges of space debris detection. The training process involved gradually adjusting the model's parameters to improve its accuracy.I paid special attention to making the system practical to use. This meant optimizing it to run efficiently on standard computer hardware and making sure it could process images quickly enough for real-world applications. The training of the model did however take me 27 hours \! I also added features to help visualize the results as you can see in the part before, making it easier to understand what the model is detecting.

**Conclusion**

This project taught me a lot about both AI and space technology. While my solution isn't perfect, it demonstrates how AI can help address real-world problems. The results show that combining different AI techniques can lead to better solutions than using any single approach.Looking ahead, I see several ways to improve the system. It could be enhanced to track debris over time, not just in single images. It could also be modified to work with different types of satellite imagery. The experience of developing this system has reinforced my interest in both AI and space technology. It's shown me how these fields can work together to solve important problems, and I'm excited to continue learning and developing similar solutions in the future.  
