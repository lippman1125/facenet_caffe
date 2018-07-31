FaceNet
=======

Data Preprocess
---
>>We introduce a new large-scale face dataset named VGGFace2. The dataset contains 3.31 million images of 9131 subjects (identities),
>>with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose,
>>age, illumination, ethnicity and profession (e.g. actors, athletes, politicians).

>>(1)人脸对齐  
>>使用vggface_align.py脚本进行人脸对齐，对齐方法：读取loose_bb_train.csv标注文件中人脸框，人脸框4个方向各外扩20%，抠取人脸，然后把人脸框的最小边缩放到128（宽高等比例缩放）。

>>vggface2_face的人脸样图如下：  
>>![vggface2_face](https://github.com/lippman1125/github_images/blob/master/facenet_images/vggface2_face.jpg)

>>对齐后的人脸如下:  
>>![vggface2_face_aligned](https://github.com/lippman1125/github_images/blob/master/facenet_images/vggface2_face_aligned.jpg)

>>(2)人脸列表  
>>利用face_labels_gen.py脚本生成face label。生成的列表如下：
>>![vggface2_list](https://github.com/lippman1125/github_images/blob/master/facenet_images/vggface2_list.jpg)

>>(3)生成LMDB  
>>利用create_vggface2.sh脚本生成caffe训练时用的lmdb数据。

Train
---
>>./build/tools/caffe train --solver examples/face_recog/inception_resnet_v2_tiny_solver.prototxt  --gpu 0,1

Test
---
>>(1)LFW数据集测评：  
>>利用face_similarity.py脚本生成LFW人脸对的余弦相似度得分。  
>>python face_similarity.py   --image_dir lfw_mtcnnpy_160  
>>                            --pair_file face_list_lfw.txt  
>>                            --result_file  facenet_vggface2_inception_resnet_v2_lfw.txt  
>>                            --network inception_resnet_v2_tiny_deploy.prototxt  
>>                            --weights facenet_inception_resnet_v2_tiny_iter_192000.caffemodel  

>>(2)LFW性能分析：  
>>利用roc_curve.py脚本生成结果。    
>>python roc_curve.py face_list_lfw_gt.txt  facenet_vggface2_inception_resnet_v2_lfw.txt  
>>![ROC](https://github.com/lippman1125/github_images/blob/master/facenet_images/ROC.jpg)
>>![FAR-FRR](https://github.com/lippman1125/github_images/blob/master/facenet_images/FAR_FRR.jpg)
>>![HISTOGRAM](https://github.com/lippman1125/github_images/blob/master/facenet_images/Hist.jpg)
