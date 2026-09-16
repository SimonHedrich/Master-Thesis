<!-- image -->

Contents lists available at ScienceDirect

## Computer Vision and Image Understanding

[journal homepage: www.elsevier.com/locate/cviu](http://www.elsevier.com/locate/cviu)

## Anovel algebraic solution to the perspective-three-line pose problem ✩

Ping Wang a , Guili Xu a, ∗ , Yuehua Cheng b

- a College of Automation Engineering, Nanjing University of Aeronautics and Astronautics, Nanjing 211106, China
- b College of Astronautics, Nanjing University of Aeronautics and Astronautics, Nanjing 211106, China

## A R T I C L E I N F O

Communicated by Nikos Paragios

MSC: 41A05 41A10 65D05 65D17

Keywords:

Perspective-three-line problem (P3L) Absolute position and attitude Pose estimation Imaging geometry Computer vision

## 1. Introduction

Determining the pose of a calibrated camera from three correspondences between 3D reference features and their 2D projections has numerous applications in computer vision and robotics. Examples include robot localization (Gai et al., 2016), augmented reality (Marchand et al., 2016), and spacecraft pose estimation during descent and landing (Mourikis et al., 2009; Ryan et al., 2015).

When the point features are employed, it becomes the well-known Perspective-Three-Point (P3P) problem, which has been well studied in recent years (Gao et al., 2003; Shiqi and Chi, 2011; Kneip et al., 2011; Masselli and Zell, 2014; Ke and Roumeliotis, 2017; Wang et al., 2018). Meanwhile for line features, it corresponds to the Perspective-ThreeLine (P3L) problem, which remains a challenging topic. P3L attracts many researchers, because it is the smallest subset of Perspective𝑛 -Line (P 𝑛 L) problem, and its solution plays a fundamental role in dealing with the general P 𝑛 L problem. As one of the earliest analytical methods, Dhome et al. (1989) proposed a polynomial method to solve the P3L problem. They transformed 3D lines and 2D lines into a model coordinate system and a virtual viewer coordinate system, respectively, and then derived a eighth order polynomial to determine the closed-form solution of the P3L problem. However, one of the weaknesses of Dhome's method (Dhome et al., 1989) is that its

## A B S T R A C T

In this work, we present a novel algebraic method to the perspective-three-line (P3L) problem for determining the position and attitude of a calibrated camera from features of three known reference lines. Unlike other methods, the proposed method uses an intermediate camera frame F and an intermediate world frame E, with sparse known line coordinates, facilitating formulations of the P3L problem. Additionally, the rotation matrix between the frame E and the frame F is parameterized by using its orthogonality, and then a closed-form solution for the P3L pose problem is obtained from subsequent substitutions. This algebraic method makes the processes more easily followed and significantly improves the performance. The experimental results show that the proposed method offers numerical stability, accuracy and efficiency comparable or better than that of state-of-the-art method.

mathematical expression is too complex to solve. Later, Liu et al. (1990) presented an iterative method to the P3L problem. However, this iterative method is usually time-consuming, and is easily trapped into a local minimum, which will lead to poor results. Chen (1991) investigated the necessary conditions under which the P3L problem has finite number of solutions, and proposed another polynomial method to the P3L problem. Unfortunately, Chen's method is highly unstable in the presences of noise. Caglioti (1993) addressed a special case of the P3L problem, where three lines lie in a common plane and intersect at a common point, which is called the planar 3-line junction perspective problem. Qin and Feng (2008) addressed another special case of the P3L problem, where three non-coplanar lines intersect at two points, i.e., two lines are parallel and the third line intersects with both of them. Recently, Xu et al. (2017) decomposed the overall rotation from the world frame to the camera frame into a sequence of simple rotations by using the sine and cosine of two angles ( 𝛼 and 𝛽 ), and proposed a geometric method to the P3L problem. To the best of our knowledge, Xu's method (Xu et al., 2017) may be the best version to the P3L problem, offering high numerical stability, accuracy and efficiency. However, the parametric process of rotation between the model frame and camera frame is unclear for readers. Besides, Xu's method (Xu et al., 2017) also need to take into account the sign of solutions, because they depend on the sine and cosine of two rotation angles.

✩ No author associated with this paper has disclosed any potential or pertinent conflicts which may be perceived to have impending conflict with this work. For full disclosure statements refer to https://doi.org/10.1016/j.cviu.2018.08.005.

∗ Corresponding author. E-mail addresses: pingwangsky@gmail.com (P. Wang), guilixu2002@163.com (G. Xu).

<!-- image -->

<!-- image -->

In contrast to all previous methods, in this paper we provide a novel algebraic method to the P3L problem. The advantages of our method are summarized as follows:

- The proposed method uses an algebraic approach to solve the P3L problem, which simplifies understanding of the processes and improves overall performance . Our method relies on the definition of an intermediate camera frame 𝐹 and an intermediate world frame 𝐸 , in which the coordinates of 3D lines and 2D lines are sparse, facilitating derivations of the P3L problem. Additionally, according to the projection model from 3D lines to 2D lines in the normalized image plane (Zhang and Koch, 2011), and taking into account orthogonality constraints, the rotation matrix between the frame 𝐸 and the frame 𝐹 is parameterized, and then a closedform solution for the camera pose is obtained by subsequent substitutions of the parameterized matrix. All derivations of our method are based on the algebraic approach, which makes the processes easier to understand and improves overall performance.
- The proposed method has high accuracy and efficiency . Experimental results show that our method offers accuracy comparable or better than existing state-of-the-art method. The computational efficiency of our method is increased by about 10% compared with Xu's method (Xu et al., 2017).
- The proposed method has universal applicability to the P3L problem . Our method is applicable to both general and special cases (e.g. three lines are orthogonal to each other orthogonality). In contrast, some existing methods only work for special cases (Caglioti, 1993; Qin and Feng, 2008).

The remainder of this paper is organized as follows. Section 2 presents the derivations of the proposed method. Section 3 provides a thorough analysis of the proposed method by simulated experiments and real images. Section 4, finally, provides conclusions.

## 2. Theory

## 2.1. Problem statement

The problem considered in this paper is illustrated in Fig. 1. Given a calibrated camera and three known 3D reference lines 𝐿𝑖 ( 𝑖 = 1 , 2 , 3) with their corresponding 2D projections on the image plane as 𝑙 𝑖 . The goal is to recover the rotation 𝑅𝑐𝑤 and translation 𝑇 𝑐𝑤 of the camera frame with respect to the world frame. Let 𝐿𝑖 = ( 𝑉 𝑤 𝑖 , 𝑃 𝑤 𝑖 ) be a known 3D line, where 𝑉 𝑤 𝑖 is the normalized vector giving the direction of the line and 𝑃 𝑤 𝑖 is any point on the line. Let 𝑙 𝑖 = ( 𝑠 𝑖 , 𝑝 𝑖 ) be the corresponding 2D projection of 𝐿𝑖 on the image plane, where 𝑠 𝑖 and 𝑝 𝑖 are the endpoints of 𝑙 𝑖 . For a given 𝑙 𝑖 , a projection plane 𝛱𝑖 is determined which passes through the projection center 𝑂 , 𝑙 𝑖 and 𝐿𝑖 . The normal of 𝛱𝑖 is denoted as 𝑛 𝑐 𝑖 , which can be easily calculated by using the cross product of 𝑠 𝑖 and 𝑝 𝑖 . Note that the superscript indicates the different coordinate frame, i.e., 𝑤 and 𝑐 indicate the world frame and camera frame, respectively.

## 2.2. Building an intermediate camera frame

The first step involves the definition of a new, intermediate camera frame F from the normal vector 𝑛 𝑐 1 . As shown in Fig. 1, the new camera frame is defined as ( 𝐹 - 𝑓 𝑥 , 𝑓 𝑦 , 𝑓 𝑧 ) , where

<!-- formula-not-decoded -->

Fig. 1. Illustration of the P3L problem. 𝐿 1 , 𝐿 2 and 𝐿 3 are three known 3D reference lines, and 𝑙 1 , 𝑙 2 and 𝑙 3 are corresponding 2D projections. The goal is to retrieve the rotation 𝑅𝑐𝑤 and translation 𝑇 𝑐𝑤 of a camera with respect to a world frame.

<!-- image -->

<!-- formula-not-decoded -->

Via the transformation matrix 𝑇 𝑐𝑓 = [ 𝑓 𝑥 , 𝑓 𝑦 , 𝑓 𝑧 ] 𝑇 , normal vectors 𝑛 𝑐 1 , 𝑛 𝑐 2 and 𝑛 𝑐 3 can be easily transformed into F frame using

<!-- formula-not-decoded -->

where 𝑛 𝑓 𝑖 = [ 𝑛 𝑥,𝑖 , 𝑛 𝑦,𝑖 , 𝑛 𝑧,𝑖 ] 𝑇 . Since 𝑛 𝑐 1 = 𝑓 𝑥 , 𝑓 𝑥 ⟂ 𝑓 𝑦 and 𝑓 𝑥 ⟂ 𝑓 𝑧 , we can get sparse 3D coordinates of 𝑛 𝑓 𝑖 ( 𝑖 = 1 , 2 , 3) as

<!-- formula-not-decoded -->

## 2.3. Building an intermediate world frame

The second step involves the definition of a new world frame E from the direction vector 𝑉 𝑤 1 . As shown in Fig. 1, the new world frame is defined as ( 𝐸 - 𝑒 𝑥 , 𝑒 𝑦 , 𝑒 𝑧 ) , where

<!-- formula-not-decoded -->

Via the transformation matrix 𝑇 𝑤𝑒 = [ 𝑒 𝑥 , 𝑒 𝑦 , 𝑒 𝑧 ] 𝑇 , the reference lines 𝐿 1 , 𝐿 2 and 𝐿 3 can finally be transformed into E frame using

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar to the derivation of 𝑛 𝑓 𝑖 , we use 𝑉 𝑤 1 = 𝑒 𝑧 , 𝑒 𝑥 ⟂ 𝑒 𝑧 , 𝑒 𝑦 ⟂ 𝑒 𝑧 in Eq. (7) and Eq. (8) results in

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now if we are able to obtain the rotation 𝑅𝑓𝑒 and translation 𝑇 𝑓𝑒 of the 𝐹 frame with respect to the 𝐸 frame, the complete rotation 𝑅𝑐𝑤 and translation 𝑇 𝑐𝑤 are obviously also given by 𝑇 𝑐𝑓 and 𝑇 𝑤𝑒 .

## 2.4. Estimation the rotation and translation

According to the projection from 3D lines to 2D lines in the normalized image plane (Zhang and Koch, 2011), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

By substituting ( 𝑛 𝑓 1 ) 𝑇 = [1 , 0 , 0] and 𝑉 𝑒 1 = [0 , 0 , 1] 𝑇 into Eq. (11), we have By expanding Eq. (16) and Eq. (17), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

𝑟

6

(

𝑏

1

𝑟

1 +

𝑏

2

𝑟

2 +

𝑏

3

) +

𝑟

9

(

𝑏

4

𝑟

1 +

𝑏

5

𝑟

2 +

𝑏

6

) +

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

By solving Eqs. (18) and (19), 𝑟 6 and 𝑟 9 can be expressed as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

𝐵

2 = -

𝑎

1

𝑏

8 -

𝑎

2

𝑏

7 +

𝑎

7

𝑏

2 +

𝑎

8

𝑏

1

,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Eq. (13), we can get 𝑟 3 = 0 . Since 𝑅𝑓𝑒 is an orthogonal matrix, we have

<!-- formula-not-decoded -->

By substituting 𝑟 3 = 0 and Eq. (14) into 𝑅𝑓𝑒 , we have

<!-- formula-not-decoded -->

It is clear that 𝑅𝑓𝑒 is now parameterized by 𝑟 1 , 𝑟 2 , 𝑟 6 and 𝑟 9 . By substituting 𝑛 𝑓 2 , 𝑛 𝑓 3 , 𝑉 𝑒 2 , 𝑉 𝑒 3 and Eq. (15) into Eq. (11), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In consideration of the orthogonality constraints of Eq. (15), we have two additional nonlinear constrains as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By substituting Eq. (20) into Eq. (22), we get

<!-- formula-not-decoded -->

Replacing 𝑟 2 2 = 1 - 𝑟 2 1 in Eq. (23), and rearranging the terms, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

P. Wang, G. Xu and Y. Cheng

<!-- formula-not-decoded -->

Based on Eq. (24), 𝑟 2 can be expressed as

<!-- formula-not-decoded -->

After plugging 𝑟 2 back into Eq. (21), expanding and collecting, we can get a eighth order polynomial of the form

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

𝑟 1 can be easily solved from Eq. (26) by using the eigenvalue method (Flannery et al., 1986), which has eight real solutions at most. By substituting 𝑟 1 into Eq. (25), we can get 𝑟 2 . After plugging 𝑟 1 and 𝑟 2 back into Eq. (20), 𝑟 6 and 𝑟 9 can be computed, respectively. By substituting 𝑟 1 , 𝑟 2 , 𝑟 6 and 𝑟 9 into Eq. (15), and then the rotation 𝑅𝑓𝑒 is obtained.

By substituting Eq. (15) into Eq. (12), expanding and collecting, we have

<!-- formula-not-decoded -->

in which

<!-- formula-not-decoded -->

By solving Eq. (27), we have

<!-- formula-not-decoded -->

Now the rotation 𝑅𝑓𝑒 and translation 𝑇 𝑓𝑒 between the 𝐹 frame and the 𝐸 frame have been acquired. Hence the complete rotation 𝑅𝑐𝑤 and 𝑇 𝑐𝑤 are finally given as

<!-- formula-not-decoded -->

## 2.5. Missing solutions

The number of the rotation 𝑅𝑐𝑤 and 𝑡 𝑐𝑤 is directly corresponding to the number of real roots of Eq. (26). However, the number of real solutions would be inconsistent with the ground truth due to the noise. As shown in Fig. 2, 𝐹𝑡𝑟𝑢𝑒 denote the 8th order polynomial corresponding to the ground truth, and 𝐹 𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 denote the polynomial derived form the measurement. Assuming 𝐹 𝑡𝑟𝑢𝑒 has three real roots and the root 𝑥 3 being tangent with the 𝑥 -axis is the true one, 𝑥 3 would probably vanish when the measurement error is introduced. The result is that 𝐹𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 returns only two solutions, and the third solution is missed.

Fig. 2. The missing solution. 𝐹𝑡𝑟𝑢𝑒 denote an unknown polynomial corresponding to the ground truth, and 𝐹𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 denote the polynomial derived form the measurement. 𝐹𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 would be disturbed due to the measurement error.

<!-- image -->

To overcome this problem, we use an easy trick, which is proposed by Shiqi and Chi (2011), to compensate the missing solutions. Firstly, the extrema 𝑥 𝑖 of 𝐹 𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 is determined by finding the roots of its derivation. There exist at most seven candidates. Secondly, for each candidate, if 𝑥 𝑖 is minimum and 𝐹 𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 ( 𝑥 𝑖 ) &gt; 0 , or if 𝑥 𝑖 is a maximum and 𝐹𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑 ( 𝑥 𝑖 ) &lt; 0 , then 𝑥 𝑖 is probably to be the missing root. Finally, 𝑥 𝑖 is checked by the re-projection error on image plane, and the smaller the re-projection error corresponding to 𝑥 𝑖 is the missing solution.

## 3. Results

In this section, we tested the performance of the proposed method by means of synthetic data and real images, and compared the numerical stability, accuracy and efficiency with the Xu's P3L method, which is the particular case of ASPnL method (Xu et al., 2017). All methods are MATLAB and C++ implementations without any additional optimizations, and are executed on a quad-core notebook with 2.5 GHz CPU and 8 GB RAM. The source codes can be downloaded from http://pingwang.sxl.cn/.

## 3.1. Synthetic data

We synthesized a virtual perspective camera with an image size of 640 × 480 pixels, focal length 800 pixels, and principle point at the image center. Then, we generated three 3D reference lines, which are randomly distributed in the range of [-4,4] × [-4,4] × [8,16], in the camera frame, and transformed these 3D lines into the world frame using the ground-truth of rotation 𝑅𝑡𝑟𝑢𝑒 and translation 𝑇 𝑡𝑟𝑢𝑒 . Finally, we projected these 3D lines into the 2D image plane using the virtual calibrated camera. Depending on the experiment, a different level of white Gaussian noise was added to the 2D image plane. For a line triplet in the world frame, we generated four possible configurations:

(1) Configuration 1 . This is the general case that three lines are randomly distributed in space.

- (2) Configuration 2 . This is a special case that three lines are orthogonal to each other: 𝐿 1 ⟂ 𝐿 2 , 𝐿 1 ⟂ 𝐿 3 and 𝐿 2 ⟂ 𝐿 3 .

(3) Configuration 3 . This is a special case that two lines are parallel, and the third line is orthogonal to them. In this configuration, there are three cases: 𝐿 1 ∥ 𝐿 2 , 𝐿 1 ∥ 𝐿 3 or 𝐿 2 ∥ 𝐿 3 .

(4) Configuration 4 . Three lines are parallel to each other: 𝐿 1 ∥ 𝐿 2 ∥ 𝐿 3 .

For the fourth configuration, since all three lines are parallel in space, only one equation can be acquired from Eq. (11), which is underconstraint for solving the P3L problem, so we consider only the first three configurations in our simulation results. The estimated rotation and translation were defined as 𝑅 and 𝑇 , respectively, and the errors of each were calculated as

Fig. 3. The distribution of average running time.

<!-- image -->

Table 1 The total (1st line) and single (2nd line) running time.

|              |      Xu |   The proposed method |
|--------------|---------|-----------------------|
| Total time/s | 55.7320 |               49.4740 |
| Each time/ms |  0.5573 |                0.4974 |

Table 2 The mean stability.

|               | Xu               | The proposed method   |
|---------------|------------------|-----------------------|
| Rotation/rad  | 4 . 8500 × 10 -5 | 9 . 1897 × 10 -7      |
| Translation/m | 1 . 1849 × 10 -4 | 4 . 2481 × 10 -6      |

<!-- formula-not-decoded -->

where 𝑟 𝑘,𝑡𝑟𝑢𝑒 and 𝑟 𝑘 are the 𝑘 th column of 𝑅𝑡𝑟𝑢𝑒 and 𝑅 , respectively. Note that this error metric might not be standard, but we use it in order to remain consistent with previous works.

## 3.2. Computational time

The first simulated experiment was designed to determine the evaluation of the computational cost of both methods. This experiment consisted of 1000 runs, where one run comprised 100 evaluations of the same set of 3D reference lines. We tested the time of 1000 runs and show the histograms of computing time in Fig. 3. The data shows that our method is faster than Xu's method in efficiency. The total and single running time of all methods are shown in Table 1. The speed of our method, which takes only 0.4974 ms, is faster by about 10% compared with Xu's method.

## 3.3. Numerical stability

This second simulated experiment tested the numerical stability of the proposed method using noise-free data. We performed 50,000 runs without the addition of noise to the projected image lines. For each run, we evaluated all solvers and calculated the camera rotation and translation. If a solver produced more than one feasible solution, we selected the one closest to the ground-truth camera as the estimated pose. The distributions and the means of the rotation and translation errors are depicted in Fig. 4(a), Fig. 4(b) and Table 2. As evident, the distributions of our method were more concentrated around the smallest error region, which means that our method has better numerical stability than Xu's method.

Fig. 4. Numerical stability of all methods. The horizontal axis shows the 𝑙𝑜𝑔 10 value of the absolute rotation error (a) and the absolute translation error (b).

<!-- image -->

## 3.4. Noise sensitivity

The last simulated experiment tested the effects of noise on the accuracy and precision of the computation for all methods. We added different levels of Gaussian noise from 0 to 10 pixels onto the image points and generated 1000 triplets of 2D-3D lines for each noise level. The mean and median rotation and translation errors depending on the noise for different configurations are shown in Fig. 5, and show that the error increases approximately linearly with the noise for all configurations. Our method provided accuracy and precision comparable or better than Xu's methods in most configurations. The reason possibly is that our method uses the sparse coordinates of lines to facilitate formulations of the P3L problem, which simplifies the derivations. Note that there are several peaks appearing in the mean rotation and translation errors with the increasing of noise levels. This behavior is caused by single outlier results with high error.

## 3.5. Real images

We also tested our method on two image sequences with known 3D line model. As shown in Fig. 6, three lines of the known model, represented by blue in Fig. 6, were selected to calculate the camera rotation and translation, and the 3D line model was back-projected onto the image by computed rotation and translation. In order to visualize the results, we used those reprojected lines to construct a series of yellow virtual boxes into the real images. As shown, the proposed method can recover the camera pose.

## 4. Conclusion

In this paper, we developed an new algebraic method to solve the P3L problem. The main idea is to simplify the known 3D lines and 2D lines coordinates by using an intermediate camera frame and an intermediate world frame, and to parameterize the rotation matrix by employing its orthogonality. The derivations are easy to understand, and the final method offers numerical stability, accuracy, and efficiency comparable or better than that of state-of-the-art method.

Fig. 5. Results are obtained over 1000 runs, and the mean and median errors of rotation and translation are shown as a function of the noise level. The mean rotation errors of our method are significantly better than Xu's method for all configurations, and the other deviations of our method are comparable to Xu's method.

<!-- image -->

Fig. 6. Images that is augmented with the projected lines by using the estimated pose. The blue lines are the used line segments, and the yellow lines are the projection of the known 3D line model using the estimated pose. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

<!-- image -->

## Acknowledgments

This study was supported by the National Natural Science Foundation of China (Nos. 61473148), and the Foundation of China Shipbuilding Industry Company Limited (Nos. 6141B04050103).

## References

[Caglioti, V., 1993. The planar three-line junction perspective problem with application to the recognition of polygonal patterns. Pattern Recognit. 26, 1603-1618.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb1)

[Chen, H.H., 1991. Pose determination from line-to-plane correspondences: existence condition and closed-form solutions. IEEE Trans. Pattern Anal. Mach. Intell. 13, 530-541.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb2)

- [Dhome, M., Richetin, M., Lapreste, J.T., Rives, G., 1989. Determination of the attitude of 3d objects from a single perspective view. IEEE Trans. Pattern Anal. Mach. Intell. 11, 1265-1278.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb3)

[Flannery, B.P., Flannery, B.P., Teukolsky, S.A., Vetterling, W.T., 1986. Numerical Recipes: The Art of Scientific Computing. Cambridge University Press.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb4)

[Gai, S., Jung, E.J., Yi, B.J., 2016. Multi-group localization problem of service robots based on hybrid external localization algorithm with application to shopping mall environment. Intell. Serv. Robotics 9, 257-275.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb5)

- [Gao, X.S., Hou, X.R., Tang, J., Cheng, H.F., 2003. Complete solution classification for the perspective-three-point problem. IEEE Trans. Pattern Anal. Mach. Intell. 25, 930-943.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb6)
- Ke, T., Roumeliotis, S., 2017. An efficient algebraic solution to the perspectivethree-point problem. In: Computer Vision and Pattern Recognition. CVPR, pp. 4618-4626.

[Kneip, L., Scaramuzza, D., Siegwart, R., 2011. A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation. In: Computer Vision and Pattern Recognition. CVPR, IEEE, pp. 2969-2976.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb8)

- [Liu, Y., Huang, T.S., Faugeras, O.D., 1990. Determination of camera location from 2-d to 3-d line and point correspondences. IEEE Trans. Pattern Anal. Mach. Intell. 12, 28-37.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb9)
- [Marchand, E., Uchiyama, H., Spindler, F., 2016. Pose estimation for augmented reality: a hands-on survey. IEEE Trans. Vis. Comput. Graphics 22, 1-1.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb10)
- Masselli, A., Zell, A., 2014. A new geometric approach for faster solving the perspectivethree-point problem. In: International Conference on Pattern Recognition, pp. 2119-2124.
- [Mourikis, A.I., Trawny, N., Roumeliotis, S.I., Johnson, A.E., Ansar, A., Matthies, L., 2009. Vision-aided inertial navigation for spacecraft entry, descent, and landing. IEEE Trans. Robot. 25, 264-280.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb12)
- [Qin, L.J., Feng, Z., 2008. A new method for pose estimation from line correspondences. Acta Automat. Sinica 34, 130-134.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb13)
- [Ryan, J., Hubbard, A., Box, J., Todd, J., Christoffersen, P., Carr, J., Holt, T., Snooke, N., 2015. Uav photogrammetry and structure from motion to assess calving dynamics at store glacier, a large outlet draining the greenland ice sheet. Cryosphere 9, 1-11.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb14)
- [Shiqi, L.I., Chi, X.U., 2011. A stable direct solution of perspective-three-point problem. Int. J. Pattern Recognit. Artif. Intell. 25, 627-642.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb15)
- [Wang, P., Xu, G., Wang, Z., Cheng, Y., 2018. An efficient solution to the perspective-three-point pose problem. Comput. Vis. Image Underst. 166, 81-87.](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb16)
- [Xu, C., Zhang, L., Cheng, L., Koch, R., 2017. Pose estimation from line correspondences: A complete analysis and a series of solutions. IEEE Trans. Pattern Anal. Mach. Intell. 39 (1209).](http://refhub.elsevier.com/S1077-3142(18)30192-9/sb17)
- Zhang, L., Koch, R., 2011. Hand-held monocular slam based on line segments. In: Machine Vision and Image Processing Conference, pp. 7-14.