# minifig-classification
Lego Minifigure Classification is a deep learning project designed to identify different Lego Star Wars minifigures using a Convolutional Neural Network (CNN). The model was trained using TensorFlow and Keras to classify five specific characters. The goal of this project was to create an accurate classifier despite working with a relatively small dataset.

One of the biggest challenges was overfitting—the model quickly reached 100% accuracy on training data, but validation accuracy remained significantly lower. This indicated that the model memorized the training images rather than learning general patterns. To improve generalization, future steps include increasing the dataset size by collecting and labeling more images.

This project was built using TensorFlow and Keras for model training, NumPy for numerical operations, Pandas for data management, and Matplotlib/Seaborn for visualization. OS and Shutil were used to organize the dataset, and TensorFlow’s ImageDataGenerator was initially used for data augmentation experiments. The biggest issue encountered was overfitting due to the small dataset. The training accuracy quickly hit 100%, while validation accuracy remained low, meaning the model was not generalizing well to new images.

To address this, I experimented with data augmentation techniques like random rotations, flips, and zooms to artificially increase the dataset size. However, these transformations altered the Lego minifigures in ways that made classification harder rather than improving performance. As a result, I decided to remove data augmentation. The best solution moving forward is to collect more real images to increase the dataset size.

Future Improvements:
-Expanding the dataset by collecting and labeling more Lego minifigure images.
-Fine-tuning a pre-trained CNN instead of training from scratch to improve generalization.
-Exploring techniques like dropout and batch normalization to further reduce overfitting.

This project was inspired by my passion for AI and Lego Star Wars. I learned valuable skills in TensorFlow and CNNs by following a tutorial by Patrick Loeber, which provided a great starting point for building this model. 

I'm Stefano, a grade 10 student from Toronto with a strong interest in AI and computer vision. In my free time, I enjoy playing basketball, 3D printing, and working on deep learning projects. Feel free to reach out with any questions at stefanoedwards@icloud.com.
