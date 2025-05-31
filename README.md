# Large-Scale Face Matching
In this work, we propose a comprehensive approach to address various real-world challenges in face matching within face search systems. The implementation has been done using Python.

## Challenges in Large-Scale Face Matching
1. Variations in lighting, pose, age progression, occlusions (e.g., glasses, masks), image quality
2. With large-scale databases, real-time search performance becomes a critical issue, requiring efficient similarity search algorithms and data structures to handle millions of facial embeddings while maintaining accuracy.

## Proposed approach: 
1. Face detection and feature extraction: using **MTCNN (Multi-task Cascaded Convolutional Networks), InceptionResnetV1**
2. Efficient indexing and similarity search: using **Facebook Faiss Vector Database**
3. Dynamic matching thresholds to enhance robustness and scalability

## Step 1: Face Detection & Feature Extraction
1. For initial face localization, we apply an object detection model (**MTCNN**) that can handle variations in in pose, lighting, and occlusions. This step isolates faces from backgrounds, providing normalized images for consistent feature extraction. <br>
   <img src="/MTCNN_architecture.png" alt="MTCNN Architecture" width="400"/>
   <img src="/mtcnn_image.png" alt="feature extraction" height="360">
   
2. After detection, each face is converted into a feature embedding using a pre-trained model (**Inception Resnet V1**). <br>
   <img src="/inception_resnet_v1.png" alt="Inception Resnet Architecture" width="400"/>
   
3. This embedding captures distinctive facial features in a high-dimensional space, allowing for accurate similarity comparisons with other embeddings in the database.
4. MTCNN will return images of faces all the same size, enabling easy batch processing with the Resnet recognition module.
5. Thus, a Facial Database is created for face searching.

## Step 2: Indexing & Similarity Searching
1. **FAISS (Facebook AI Similarity Search)** was used, an optimized library for similarity search that enables high-speed, large-scale embedding comparisons.
2. **Indexing technique:** Structure embeddings using a combination of IVF (Inverted File Index) and PQ (Product Quantization). Multi-level index that partitions the embedding space, reducing the number of comparisons required by clustering similar embeddings together. Product quantization further compresses each embedding to reduce storage requirements, making the system both memory and speed efficient. <br>
   <img src="/faiss.png" alt="FAISS" width="370"/>
4. **Approximate Nearest Neighbor Search:** To quickly identify the most similar embeddings, we apply ANN search. The IVF-PQ index enables efficient search by approximating the nearest neighbors, significantly reducing search time compared to brute-force methods. For each query, only a subset of the dataset is compared, accelerating the retrieval process without sacrificing much accuracy.

## Step 3: Dynamic Matching Thresholds
1. **Threshold Calibration:** Using a validation dataset, we perform threshold calibration to determine optimal thresholds across various conditions. For instance, lower thresholds are applied for images with significant variations (e.g., low light conditions or partial occlusions), while higher thresholds are used for clear, well-lit images.
2. **Adaptive Matching:** During face matching, the system evaluates each query image’s characteristics (e.g., brightness, contrast) and dynamically adjusts the threshold accordingly. This adaptive approach minimizes errors by allowing more lenient matching for challenging images while maintaining strict thresholds for clearer images, balancing precision and recall.

## Results
**Algorithm Comparison:** <br>
Five machine learning algorithms have been applied to a fresh dataset. <br>
  <img src="/results_metrics.png" alt="Results" width="400"/>

**End-to-end Latency & Elapsed time graphs:** <br>
  <img src="/latency_graph.png" alt="Latency" height="200"/>
  <img src="/elapsed_time_graph.png" alt="Latency" height="200"/>
