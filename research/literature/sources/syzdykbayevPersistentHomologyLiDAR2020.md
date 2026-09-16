<!-- image -->

[Remote Sensing of Environment 246 (2020) 111816](https://doi.org/10.1016/j.rse.2020.111816)

Contents lists available at ScienceDirect

## Remote Sensing of Environment

[journal homepage: www.elsevier.com/locate/rse](https://www.elsevier.com/locate/rse)

## Persistent homology on LiDAR data to detect landslides

Meirman Syzdykbayev a, ⁎ , Bobak Karimi b , Hassan A. Karimi a

- a Geoinformatics Laboratory, School of Computing and Information, University of Pittsburgh, 135 North Bellefield Avenue, Pittsburgh, PA, USA

b Department of Environmental Engineering and Earth Sciences, Wilkes University, 84 West South Street, Wilkes-Barre, PA, USA

## A R T I C L E I N F O

Edited by Jing M. Chen

Keywords: GIS LiDAR DTM Persistent homology

Landslide detection

## 1. Introduction

Mass movements, commonly generalized with the term landslides, are potentially catastrophic geologic events that can take lives, cause economic loss, and have negative environmental impact (Highland and Bobrowsky, 2008). To mitigate the loss of lives, damages, and costs associated with mass movements, susceptibility maps, maps that describe the relative chance of future slope failures based on intrinsic characteristics of a locale or site (e.g., hillslope gradient, bedrock geology, or soil type), are some of the most useful tools available to planners and the public (Burns and Mickelson, 2016). The creation of susceptibility maps requires a landslide inventory, or dataset identifying individual past slope failure events within the area of study (Santangelo et al., 2015; Yalcin, 2008). Despite the overwhelming value of these inventories necessary to produce susceptibility maps, landslide inventory maps are surprisingly rare (Booth et al., 2009; Guzzetti et al., 2012). The rarity of landslide inventories is likely due to the timeconsuming process of fi eld investigations and/or manual mapping from remote imagery for large areas, access to quality data, and subjective methods in detecting, characterizing, and mapping landslides (Booth

⁎ Corresponding author.

E-mail address: mis180@pitt.edu (M. Syzdykbayev).

Available online 22 May 2020

0034-4257/ © 2020 Elsevier Inc. All rights reserved.

A B S T R A C T

Landslides can result in loss of lives, cause damage to property, infrastructure, utilities, and residential structures and can block transportation routes. Landslide inventory maps can provide spatial-temporal information about past and recent landslides and are used for analysis to create models that can characterize susceptibility. As such, these maps are considered an essential source for risk management tasks. In this paper, we propose a persistent homology method applied on LiDAR-derived digital terrain model data to detect landslides for landslide inventory maps. In testing the hypothesis that persistent homology, a method for computing topological features of a space at di ff erent resolutions, can be used to accurately detect landslides, we applied the method on LiDARderived digital terrain models to detect shapes and patterns that are indicative of landslide surface expressions. We validated our test results by comparing them to currently available landslide inventory maps for selected locations in Pennsylvania, Oregon, Colorado and Washington. The results show a di ff erent performance for each state; the accuraces were 0.79, 0.71, 0.53, 0.69, and 0.77 for fi ve study areas. Variations in performance are linked to varying surface roughness between di ff erent types of landslides, their size, shape, ages, composition, and a possible history of reactivations of landslides that would further pronounce surface expressions. To overcome some of these challenges that may hinder performance of our method, we recommend that other datasets, containing landslides, be considered as additional n th -order dimensions beyond the spatial and elevation dimensions inherent in topographic datasets.

; Guzzetti et al., 2012). Many of these challenges can be overcome with higher-resolution data, utilization of automated methods in mass movement detection (Booth et al., 2009), and deploying protocols for landslide detection, characterization, and mapping tailored to reduce errors sourced from subjective methods by di ff erent researchers (Burns and Mickelson, 2016).

In recent years, high-resolution digital imagery and elevation datasets have become much easier to acquire, allowing researchers to map smaller and more subtle slope failure events. Light Detection and Ranging (LiDAR) technology has been able to penetrate vegetative cover to generate high-resolution Digital Terrain Model (DTM), revealing morphologic features of landslides hidden beneath the dense foliage of forests (Ardizzone et al., 2007; Corsini et al., 2009; Schulz, 2007). Small Unmanned Aerial Systems (UAS) have also been e ff ectively used to assess inaccessible fi eld locations and generate data ' ondemand ' (Ahmad et al., 2017; Fonstad et al., 2013; Lucieer et al., 2014; Niethammer et al., 2012, 2008). These advances help overcome many limitations in developing complete landslide inventories, but experts will often disagree about features that indicate the presence of landslides in datasets, so detection and mapping of slope failures remains somewhat subjective (Booth et al., 2009). Additionally, subjectivity is attributed to the physical limitations and di ff erences between users embedded in the visual inspection of data products; thus, automated methods are necessary to eliminate as much human bias as possible. Computational recognition methods reduce the subjective nature of such studies and are capable of working with larger and complex datasets. Most automated landslide detection methods utilize change detection, identifying di ff erences between similar datasets taken at different times, but are at most 70% accurate in detecting landslides due to limits in spatial resolution (Booth et al., 2009; Hölbling et al., 2012; Mayr et al., 2018; Nichol and Wong, 2005). Additionally, the cost of some data collection methods limits the availability of multiple, temporally o ff set data. Current algorithms and processes are designed to pronounce and detect landslides but are not as of yet advanced enough to identify landslides with high accuracy (&gt; 50%) (Booth et al., 2009; Hölbling et al., 2012; Nichol and Wong, 2005).

<!-- image -->

<!-- image -->

Persistent homology (PH), as an advanced mathematical (topological) method, has recently gained the attention of researchers for automatic detection of objects but its use for detection of geospatial objects has not been fully explored yet. Our contributions in this paper are the development of a PH method designed speci fi cally to be used on LiDAR-derived DTM and the evaluation of its accuracy for the detection of landslides. The structure of the paper is as follows. In Section 2, backgrounds on PH, LiDAR, and landslides detection are given. The study areas used in this work are discussed in Section 3. Section 4 is devoted to the detailed description of the PH method. In Section 5, the experiments conducted to detect landslides in the fi ve study areas are discussed. The accuracy of the PH method, using speci fi c metrics, is analyzed in Section 6. Conclusions and future works are given in Section 7.

## 2. Background

Here we use ' landslide ' as the general term for all downslope movement of a mass of rock, ground, or debris under gravitational forces and also to describe the landform that results from such movement (Highland and Bobrowsky, 2008). Mass movements are a common response to gravity (driving force) in weak and fractured rock, steep and slope materials. The failure itself is a result of natural or anthropogenic causes (external or internal) which function to either increase the driving forces or reduce the resisting forces (Popescu, 1994). Past studies have revealed that human alterations from overloaded slopes, excavation at the base of slopes, or altered drainage conditions have also accelerated the natural processes of erosion and land shift (Alexander, 1992). Removal of vegetation, mining, and vibrations caused by construction also contribute to slope instability (Pomeroy, 1982). The type of soil, hillslope angle and orientation, levels of precipitation, earthquakes, presence of old landslides (prior failure), and the over-steepening of slopes by stream erosion or engineered modi fi cations are all factors in determining the development of landslides. Di ff erent materials combined with di ff erent causes and local conditions can in fl uence the overall way in which a slope may fail. These types of failures are best described by Varnes (1978), who classi fi ed slope failures based on how the mass moved and the type of material involved. Based on these criteria, slope failures are classi fi ed into fi ve major groups: falls, topples, slides (translational and rotational), lateral spreads, and fl ows (Casale et al., 1994; Varnes, 1978) (Table 1). A sixth classi fi cation (complex) describes combinations of these mass movement types (Varnes, 1978). When considering mass movement detection, it is important to understand the classi fi cation types as they will exhibit di ff erent surface expressions. Not all types of mass movements will share similar surface expressions, which is one of the reasons why their automated detection is currently limited.

Mass movements leave discernible signs on the landscape, such as altered slope shape, position, or appearance of the surface (i.e., increased surface roughness or hummocky terrain). These morphological

2

Remote Sensing of Environment 246 (2020) 111816

Table 1 Abbreviated and modi fi ed version of Varnes (1978) slope movement classi fi - cation. This system taking into account both type of movement and the material being moved.

| Type of movement   | Type of movement         | Type of material                                       | Type of material                                       | Type of material                                       |
|--------------------|--------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
|                    |                          | Bedrock                                                | Engineering soils                                      | Engineering soils                                      |
|                    |                          | Bedrock                                                | Predominantly coarse                                   | Predominantly fi ne                                    |
| Falls Topples      | Falls Topples            | Rock fall Rock topple                                  | Debris fall Debris topple                              | Earth fall Earth topple                                |
| Slides             | Rotational Translational | Rock slide                                             | Debris slide                                           | Earth slide                                            |
| Lateral Spreads    | Lateral Spreads          | Rock spread                                            | Debris spread                                          | Earth spread                                           |
| Flows              | Flows                    | Rock flow (Deep creep)                                 | Debris flow (Soil creep)                               | Earth flow                                             |
| Complex            | Complex                  | Combination of two or more principal types of movement | Combination of two or more principal types of movement | Combination of two or more principal types of movement |

expressions are often expressed as changes in surface roughness and curvature on or at the base of slopes (Fig. 1). These topographic features can be created at the top, bottom, left, or right side of the landslide. The size of features varies depending on the shape and type of the landslides and the sharpness of the features depends on the material involved and on the erosion process. Field surveys, manual inventories, and automated or semi-automated methods using aerial and satellite imagery are currently used to identify morphologic features and map landslide occurrences. Features that might indicate the presence of a mass movement include a crown, headscarp, lateral margins, and/or toe (Varnes, 1978). These features are curvilinear changes in curvature that de fi ne the perimeter of landslides. Other slide indicators, such as hummocky landforms internal scarps, fl ow lines, displaced slide mass, or the chaotic masses of rock rubble at the base of failure along a slope, can also indicate mass movement types. These slide indicators can be expressed as increasd surface roughness over the areal extent of the landslide. However, not all morphological indicators are always developed or preserved through time. In this paper, we employed PH, a novel topological method, to identify boundaries of landslides, which can be thought of as topological holes.

## 2.1. Persistent homology

Topology is a branch of mathematics that addresses qualitative geometric information. Topology can be used to classify dataset if it has speci fi c shape within a space such as loops and higher dimensional shapes (Carlsson, 2009). Topological invariants of spaces and mapping between these spaces are used to record the essential qualitative features of the space and are insensitive to coordinate changes and deformations. Homology is an algebraic compression scheme that removes all except necessary topological features,such as connected components and loops, from a speci fi c class of data structures occurring naturally from topological spaces (Ghrist, 2018).

We applied our PH method to LiDAR-derived DTM by fi rst identifying ridge and scarp lines that help enhance the presence of ' persistent ' features indicative of a mass movement body. This is best accomplished by looking at the birth and death ( ' persistence ' ) of topological loops, where smaller scale loops could indicate a mass movement landform feature.

PH is an algebraic tool for recording the topological features (Edelsbrunner and Harer, 2008) and reducing the features of a dataset to their most simple forms, providing information on the length of feature persistence. The persistence of features provides insights into their size and type, e.g., a point, a hole, or a void. Features that form a large hole or a void persist longer. To analyze persistent homological

3

Remote Sensing of Environment 246 (2020) 111816

Fig. 1. LiDAR 1-m DTM shaded relief map from Pennsylvania along the southern slope of Shickshinny Mountain fl anking the Susquehanna River in Luzerne County with annotations identifying morphological expressions of a traditional slide type mass movement.

<!-- image -->

data, the data are represented as simplicial complexes, which are combinatorial structures used to compute homology because they approximate the topological space. Simplicial complexes are created from points, edges, and polygons. From these simplicial complexes, PH records appearance (birth) and disappearance (death) of topological features.

PH is used to analyze the shape of high-dimensional data. The shape data can be retrieved at a di ff erent scale, both small and large connected components, holes and voids. Hence PH allows retrieving small holes that are inside or at the border of large holes. Another advantage of PH is its stability under perturbation. Adding noise or slightly changing the dataset will not change the output dramatically.

For the reasons that computation of PH requires data to be represented as simplicial complexes and 0-dimensional simplex is a point, point clouds are suitable for PH methods. It is also worth mentioning that topological data analysis may not be useful for all types of data, so care must be given to selecting the right type of data, or explicitly preparing data for PH methods. For example, the images need to be converted to combinatorial structures called ' cubical complexes ' that is similar to simplicial complexes before PH can be computed. Weighted undirected networks need to be converted to point clouds (Otter et al., 2017).

PH covers a wide range of applications in a variety of fi elds. Li et al. (2014) presented a computer vision framework for object detection using topological persistence and showed that PH can provide informative descriptors for shapes and images. In the fi eld of medicine, Ferri et al. (2017) used PH and a k-Nearest Neighbour search algorithm in order to detect melanoma from the images. Xia and Wei (2014) applied PH for extracting molecular topological fi ngerprints that used for characterization, identi fi cation, and classi fi cation of the protein. In the data analysis fi eld, Islambekov and Gel (2019) proposed a clustering algorithm that is based on PH and compared results with current unsupervised clustering algorithms. Pereira and De Mello (2015) applied PH on spatial and time series data clustering and discussed topological features that can identify signi fi cant patterns. To the best of our knowledge, PH has not yet been applied, along with LiDAR-derived

4

Remote Sensing of Environment 246 (2020) 111816

Fig. 2. United States landslide susceptibility map with four selected states: Pennsylvania, Oregon, Colorado and Washington. (For interpretation of the references to color in this fi gure, the reader is referred to the web version of this article.)

<!-- image -->

DTM, to topographic data to detect landslides as a means for creating landslide inventory maps.

## 2.2. LiDAR

Airborne LiDAR is a remote-sensing method that is used to acquire digital representations of a topographic surface (Guzzetti et al., 2012). LiDAR can penetrate terrain that is covered in vegetation and thus is able to provide quantitative descriptions of a topographic surface in heavily covered areas. This ability provides an advantage over other approaches, such as optical aerial or satellite images, which are based on visual understanding and are not able to penetrate the vegetation canopy. This ability is speci fi cally important when attempting to detect and map landslides in heavily vegetated areas (Guzzetti et al., 2012).

Most landslide inventories utilize aerial and/or satellite imagery to locate and detect landslides which are then may be con fi rmed by fi eld surveys. Researchers have also attempted to implement LiDAR DTM for the automatic or semi-automatic rapid identi fi cation of landslide characteristics Guzzetti et al. (2012). LiDAR-derived DTM approaches have potential uses for assessment, mitigation, and post-event recovery from hazards.

## 2.3. Landslide detection

The use of satellite or airborne technologies allows the development of a variety of new automated, semi- automated, machine learning and object-oriented techniques. Currently, many studies are focused on detecting landslides using machine learning techniques that incorporate features derived from Terrestrial Laser Scanning (TLS) (Mayr et al.,

2017), airborne LiDAR (Mezaal et al., 2017), radar (Mahrooghy et al., 2015), and orthophotos (Ghorbanzadeh et al., 2019). Mezaal and Pradhan (2018) implemented a data mining approach to detect landslides in a rain forested area of Malaysia and compared classi fi cation accuracies of di ff erent classi fi ers. Fanos et al. (2018) proposed a hybrid approach for detecting landslides that is based on the Gaussian mixture model and random forests in Malaysia. Casagli et al. (2016) employed satellite interferometric synthetic aperture radar (InSAR) and objectbased image analysis in di ff erent areas around the world to detect and map landslide locations.

However, these supervised machine learning methods require prelabelled data to train a landslide-detection model. This is di ff erent than unsupervised machine learning which is object-oriented and does not need pre-labled data. Leshchinsky et al. (2015) proposed a Contour Connection Method (CCM) that applies contours and nodes to a map to evaluate landslide features and gradient. Bunn et al. (2019) proposed a new semi-automated Scarp Identi fi cation and Contour Connection Method (SICCM) by adapting CCM. SICCM, automatically or semi-automatically, adapts to diverse geologic settings and can be improved by considering curvature to identify scarps.

The methods described above rely heavily on topographic datasets or datasets that are heavily in fl uenced by topography (e.g., InSAR and LiDAR). Traditional methods of detecting landslides, including automated methods, are limited in the range of datasets they utilize (Booth et al., 2009), which is a likely contributing factor that limits their accuraces. PH methods can include multiple datasets. Bivariate susceptibility methods, such as the frequency-ratio method, can correlate landslide locations to various factors, such as vegetative cover, bedrock lithology, soils, etc. (Chalkias et al., 2014). This correlation further serves as a prediction of where landslides are likely to occur. In PH methods, these factors can be used to further di ff erentiate points that de fi ne landslide features by n-dimensions besides latitude, longitude, and elevation, where n refers to a new quantitative measure, such as the factors mentioned above. The ability of PH to identify landslides using a multitude of datasets, while remaining computationally e ffi cient makes it a powerful approach. For the purposes of this paper, we only use topographic datasets so as to evaluate the success of PH methods on a commonly used dataset (topography).

## 3. Study areas

We chose fi ve locations to test our PH method. The criteria for the selection of the study areas are high landslide density, availability of LiDAR data, and availability of a landslide inventory map. We selected areas with high landslide susceptibility from a United States landscape susceptibility map (U.S. Geological Survey (USGS), n.d.), shown in Fig. 2, in Pennsylvania, Oregon, Colorado, and Washington states in which both LiDAR data and a landscape inventory map exist and distinct di ff erences in vegetative cover, climate, or near-surface materials are represented.

For Study Area 1, marked with a blue pin in Fig. 2 and shown also in Fig. 3, we chose a site along the southern face of Shickshinny Mountain on the northern fl ank of the Susquehanna River in Luzerne County, Pennsylvania (latitude 41°12 ′ 39 ″ N and longitude 76°03 ′ 44 ″ W). In an area of 25,572,981 square meters, there are seven identi fi ed landslides (Karimi et al., 2019), and the combined area of mapped landslides is 2,140,041 square meters. The study area is characterized by a dense vegetation cover mostly with deciduous forest, which may work to stabilize slopes, depending on soil/weathered bedrock thickness, and annual rainfall average based on a 30-year normal (1981 - 2010: 1156.2 mm) according to data from a station in Shickshinny (USC00368057) (Arguez et al., 2010). The elevation ranges from 151 m to 471 m with a standard deviation of 100 m. The slope inclination in the area ranges from 0 to 71.8° with a mean of 4.9° and a standard Remote Sensing of Environment 246 (2020) 111816

deviation of 3.92°, which overall does not indicate increased driving shear stresses on slopes but could at site localities where inclination is high (Table 2). The area is predominantly covered by soil whose formation is promoted by the wet climate and biological processes, including deep-rooted vegetation that more rapidly weathers the underlying bedrock (Reybold and TeSelle, 1989).

Study Area 2, marked with a red pin in Fig. 2 and shown also in Fig. 4, is located in Oregon (latitude 45°34 ′ 0 ″ N and longitude 123°11 ′ 0 ″ W). The total area is 135,587,076 square meters. We chose this location because it was one of the study areas in Bunn et al. (2019) allowing us to compare fi nal outputs. There are 738 documented landslides (Burns et al., 2008) and the combined area of mapped landslides is 26,283,520 square meters. The local climate is similar in annual average rainfall to Study Area 1 with 1157.4 mm of rain, based on climate data from a station in the nearby town of Forest Grove, OR (USCC00352997) (Arguez et al., 2010). The study area is characterized by grassland, shrub, but mostly evergreen/mixed forest on slopes (Homer et al., 2004). The elevation ranges from 48 m to 548 m with a standard deviation of 110 m. The slope inclination in the area ranges from 0 to 84.22° with a mean of 3.68° and a standard deviation of 3.7° which does not indicate any major topographic changes ranging from fl at terrain to hilly, and similar to Study Area 1, suggesting only localized areas of high-slope where driving shear stresses may be higher (Table 2). There is some evidence of bedrock exposure at the surface, indicating a thinner cover of soil (Reybold and TeSelle, 1989), and further suggesting that bedrock, may be involved in the mass of moved materials.

Study Area 3, marked with a green pin in Fig. 2 and shown also in Fig. 5, is located in Mesa County, Colorado (latitude 39°10 ′ 44.6 ″ N and longitude 107°50 ′ 58.0 ″ W). This site was chosen because of its proximity to the location of the historical West Salt Creek landslide. The total area is 167,225,478 square meters. There are 206 documented landslides and the combined area of landslides is 42,828,063 square meters (Colorado Geological Survey, 2017). The study area is characterized by a dense vegetation cover of deciduous forest and evergreen forest and

Fig. 3. Study Area 1(Pennsylvania) with locations of manually identi fi ed landslides.

<!-- image -->

5

Table 2 Characteristics of study areas.

|                                                                                                                                                                                                                                                                                         | Study Area 1                                                                                                           | Study Area 2                                                                                                         | Study Area 3                                                                                                                  | Study Area 4                                                                                                       | Study Area 5                                                                                                      |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Location (latitude, longitude) State Area (square meters) Landslide Area (square meters) Percentage of Landslide Area Number of Landslides Annual Rainfall Average (millimetres) Elevation range (meters) Slope range (degrees) Slope mean (degrees) Slope standard deviation (degrees) | 41°12 ′ 39 ″ N 76°03 ′ 44 ″ W Pennsylvania 25,572,981 2,140,041 8.37% 7 1156.2 From 151 to 471 From 0 to 71.8 4.9 3.92 | 45°34 ′ 0 ″ N 123°11 ′ 0 ″ W Oregon 135,587,076 26,283,520 19.38% 738 1157.4 From 48 to 548 From 0 to 84.22 3.68 3.7 | 39°10 ′ 44.6 ″ N 107°50 ′ 58 ″ W Colorado 167,225,478 42,828,063 25.61% 206 397.51 From 1985 to 3174 From 0 to 74.7 4.81 3.66 | 45°42 ′ 0 ″ N 122°53 ′ 0 ″ W Oregon 134,963,016 61,104,222 45.27% 1664 1087.88 From 7 to 521 From 0 to 89.41 15 11 | 47°36 ′ 28 ″ N 122°20 ′ 6 ″ W Washington 216,603,450 21,470,606 9.91% 783 1219.2 From 0 to 159 From 0 to 88.6 6 7 |

Fig. 4. Study Area 2 (Oregon) with locations of previously mapped landslides.

<!-- image -->

6

Remote Sensing of Environment 246 (2020) 111816

7

Remote Sensing of Environment 246 (2020) 111816

Fig. 5. Study Area 3 (Colorado) with locations of previously mapped landslides.

<!-- image -->

annual precipitation averages, based on a 30-year normal (1981 - 2010: 397.51 mm) from a station in Collbran (USC00051743) (Arguez et al., 2010). The drier climate (relative to Study Area 1 and 2) can limit the growth of deep-rooted vegetation (e.g., tall trees whether deciduous or evergreen) and the rate of weathering, as drier climates do not have the water to support deep roots (Guswa, 2008). However, this root-depth versus wetness generalization is an assumption that can vary based on total rainfall, plant and soil characteristics, and evapotranspiration within any given area (Guswa, 2008; Schenk and Jackson, 2002). In addition, some parts of the area are covered by grasses (pasture), shrub and woody wetlands (Homer et al., 2004). The elevation ranges from 1985 m to 3174 m with a standard deviation of 269 m. The mean slope inclination is 4.81° and the standard deviation is 3.66°, suggesting more pronounced hills than in prior described areas (Table 2). Bedrock in this area is either exposed or covered with minimal soil (Reybold and TeSelle, 1989).

Study Area 4, marked with a yellow balloon in Fig. 2 and shown also in Fig. 6, is located in Oregon (latitude 45°42 ′ 0 ″ N and longitude 122°53 ′ 0 ″ W). The total area is 134,963,016 square meters. We chose this location because it was one of the study areas in Booth et al. (2009) allowing us to compare fi nal outputs. There are 1664 documented landslides (Burns et al., 2008) and the combined area of mapped landslides is 61,104,222 square meters. The annual average rainfall is 1087.88 mm, based on climate data from a station in the Scappoose industrial airport, OR (USW00004201) (Arguez et al., 2010). The study area is characterized by grassland, shrub, but mostly evergreen/mixed forest on slopes (Homer et al., 2004). The elevation ranges from 7 m to 521 m with a standard deviation of 123 m. The slope inclination in the area ranges from 0 to 89.41° with a mean of 15° and a standard deviation of 11° (Table 2). This indicates a lack of any major topographic changes ranging from fl at terrain to hilly, and similar to Study Area 1, suggesting only localized areas of high slope where driving shear stresses may be higher.

Study Area 5, marked with a purple balloon in Fig. 2 and shown also in Fig. 7, is located in Washington (latitude 47°36 ′ 28 ″ N and longitude 122°20 ′ 6 ″ W). The total area is 216,603,450 square meters. We chose this location because it was one of the study areas in Booth et al. (2009) allowing us to compare fi nal outputs. There are 783 documented landslides and the combined area of mapped landslides is 21,470,606 square meters (Sarikhan and Stanton, 2008). The annual average rainfall is 1219.2 mm, based on climate data from a station in the Seattle portage bay (USW00024281) (Arguez et al., 2010). The elevation ranges from 0 m to 159 m with a standard deviation of 39 m. The slope inclination in the area ranges from 0 to 88.6° with a mean of 6° and a standard deviation of 7° (Table 2).

## 4. Methods

## 4.1. Ridge and scarp detection

Morphological expressions of landslides, particularly of deep-seated landslides, can be characterized as a collection of small ridges and scarps, which are expressed as changes in curvature. Considerable research has been conducted on detection of terrain morphology features including ridges and scarps. Jenson and Domingue (1988) proposed a fl ow accumulation algorithm for detecting only surface ridge lines. This

8

Remote Sensing of Environment 246 (2020) 111816

Fig. 6. Study Area 4 (Oregon) with locations of previously mapped landslides.

<!-- image -->

algorithm can identify the linear terrain features but cannot distinguish between ridges and scarps. Rana (2006) proposed a curvature-based semi-automated iterative channel and ridge identi fi cation algorithm that is simple and provides reliable results. The algorithm can identify ridges and scarps but requires determination of a threshold value. Pirotti and Tarolli (2010) applied multiplication of the curvature's standard deviation as a threshold value to identify ridges and channels. Jasiewicz and Stepinski (2013) proposed a new approach to identify landform elements called ' geomorphon ' . This approach does not require a threshold value and is based on the principle of pattern recognition.

To evaluate the performance of our PH method, we compared results of three other methods: two curvature-based methods (Pirotti and Tarolli, 2010; Rana, 2006) and one pattern recognition-based method (Jasiewicz and Stepinski, 2013). The curvature-based methods involve a combination of fi rst derivative (slope) and second derivative (curvature of elevation). As the basis for classi fi cation, these methods use the variation in curvature and consist of two steps. The fi rst step removes noise in the input dataset by iteratively taking an average of the grid. Deriving precise DTM from LiDAR data has limitations and still remains a challenging process. The noise increases in complicated terrain, which is distinguished by sharp variation of elevation and dense trees and may lead to incorrect interpretation (Chen et al., 2017). Hence, noisy DTM can increase false positive rates by detecting ridges and scarps that do not exist (see Fig. 10 (b1, c1)). The second step assumes that elevation z is f(x,y), as shown in Fig. 8, and the curvature value is derived by the following algorithm (Moore et al., 1991; Zevenbergen and Thorne, 1987):

Curvature = Plan Curvature - Profile Curvature = 2E+2D Plan Curvature = 2((DG 2 + EH 2 + FGH)/(G 2 + H 2 )) Profile Curvature = 2((DH 2 + EG 2 - FGH)/(G 2 + H 2 )) D = [(Z4 + Z6)/2 - Z5]/L 2 E = [(Z2 + Z8)/2 - Z5]/L 2 F = ( - Z1 + Z3 + Z7 - Z9)/4L 2

G = ( - Z4 + Z6)/2 L

H = (Z2 - Z8)/2 L

To identify common local morphological elements such as fl ats, peaks, ridges, shoulders, spurs, slopes, hollows, footslopes, valleys, and pits, geomorphon uses the concept of Local Ternary Patterns (LTP) (Liao, 2010). The uniqueness of geomorphon is that it applies methods of computer vision to identify morphological elements from the pattern of the local terrain. Also, for the determination of LTP, instead of using a fi xed size neighbourhood, geomorphon uses a neighbourhood with size and shape that adjusts to the local topography. Hence, it can identify landforms at various spatial scales and is computationally effi cient (Jasiewicz and Stepinski, 2013).

9

Remote Sensing of Environment 246 (2020) 111816

Fig. 7. Study Area 5 (Washington) with locations of previously mapped landslides.

<!-- image -->

## 4.2. Computation of persistent homology (PH)

PH measures the multidimensional connectivity of a dataset in multiscale to examine the natural shape of the data using a distance. Even though PH can examine shape of high-dimensional objects, in this paper, we apply PH to examine shape of small and large landslides using two-dimensional objects. The most common data input format for PH is a point cloud presented as a table where columns represent coordinates and rows represent a number of points. In our proposed PH method, the data are ridge and scarp lines converted to two-dimensional points at line section vertices: long straight lines with few sections have sparse points, while lines made of many sections have more densely spaced points. The points are then converted to a graph structure and used as a proxy for the shape. This graph structure is Remote Sensing of Environment 246 (2020) 111816

<!-- image -->

Fig. 8. 3 × 3 cell window representation of a surface.

<!-- image -->

called a simplicial complex, which consists of simplices, or two, three and higher-dimensional building blocks. For example, a vertex (0-dimensional simplex) consists of one vertex, an edge (1-dimensional simplex) consists of two vertices and its face is de fi ned by two vertices and one edge, a triangle (2-dimensional simplex) consists of three vertices and three edges and its face is de fi ned by three vertices and three edges. This pattern continues on for higher dimensional simplexes.

The next step is to build a simplicial complex that uses the original data as the vertex set and describes the structure of the data. Although there are several types of simplicial complexes, we selected alpha complexes (Otter et al., 2017) to build our simplicial complex. Alpha complexes are sub-complexes of the Delaunay triangulation that creates a circle with a radius (R) around each point. If two circles intersect, the points form an edge, and if three circles intersect, the points form a triangle (polygon) and so on. If the radius is too small, none of the circles will interest, and if the radius is too large, all of them will intersect. PH checks all possible R values to detect when the structure emerges and vanishes and records them (Otter et al., 2017). The kdimensional homological ' holes ' are used as structure descriptors. For example, a set of points would eventually form a hole, or a 1-dimensional persistent homological loop, when the circles intersect, as shown in the smaller circle in Fig. 9 radius 1 or Fig. 10(d). This phenomenon is the birth point or appearance of the hole feature. As the circles grow larger (Fig. 9 radius 3), smaller holes close, sometimes resulting in a 2dimensional homological ' void ' or ' cavity ' , and eventually the hole in the center of the points disappears, which signi fi es the death of the hole feature (Fig. 9 radius 4 or Fig. 1(e)). Of the two circles in Fig. 9, the larger hole survives the longest, as can be seen in the bar graph of persistence (birth to death) for each pattern set. Topographic features created on the top, bottom, left, or right side of the landslide such as crown, headscarp, lateral margins, and toe de fi ne the perimeter of landslides. PH can detect landslides by identifying these boundaries as topological holes.

The creation and evolution of topological features are recorded in what is called a ' persistence diagram ' (PD), which is a scatter plot that assigns a single point with coordinates (birth or appearance, death or disappearance). In the diagram, each point represents a feature, and all points are mapped to the half-space above the diagonal line, as shown in Fig. 10.

Fig. 9. Representation of a point cloud on two di ff erent scales as a union of balls. Upper left-hand numbers correspond to the radius (R).

## 5. Experimentation

## 5.1. Datasets

We acquired the dataset for each of the fi ve study areas from different sources. Speci fi cally, we obtained data from the respective state government o ffi cial website for Pennsylvania, Oregon, Colorado, and Washington. Most of the data are open source, except the data for Colorado, which requires a formal request to download data. As the data source for Study Area 1, we used LiDAR data for Pennsylvania from 2006 to 2008 that are publicly available through Pennsylvania Spatial Data Access (PASDA). We obtained data on existing landslides from the landslide inventory database for north-eastern Pennsylvania (Karimi et al., 2019). We obtained LiDAR data for Study Areas 2 and 4 for 2007 from the State of Oregon Department of Geology and Mineral Industries public fi le transfer protocol (FTP) site. We obtained data on existing landslides from Oregon's Statewide Landslide Information Database (Burns et al., 2008). This database includes landslides that have been located and identi fi ed in published maps. Study Area 2 is located in a quadrangle referred to as the Gales Creek and Study Area 4 is located in a quadrangle referred to as the Dixie Mountain (Burns et al., 2012). In this dataset, landslide deposits and scarp fl anks are separated. To utilize data for our work, we merged scarp fl anks and deposits into one landslide object. We obtained LiDAR data for Study Area 3 from 2015 to 2016 and downloaded them with the request/permission from the Colorado Geological Survey (CGS). We obtained data on existing landslides with scarps and deposits from the CGS web portal (Colorado Geological Survey, 2017). In this paper, we used landslide information that was digitized from 1:24000-scale maps by the CGS and United States Geological Survey. For Study Area 5, we obtained high-resolution DTM with 1.8 m point spacing from Puget Sound LiDAR Consortium (Haugerud et al., 2003). We obtained data on existing landslides with scarps and deposits from the Washington State Department of Natural Resources web portal (Sarikhan and Stanton, 2008) (Table 3).

## 5.2. Landslide detection

To identify ridges and scarps related to landslides, we applied a fi lter to the raw LiDAR data to obtain only the point clouds that were returned from bare ground. Appropriate pixel size and smoothing parameters are vital parts of object detection techniques. With large pixels or high smoothing iterations, small ridges and scarps can be

10

11

Remote Sensing of Environment 246 (2020) 111816

Fig. 10. (a) Slopeshade of the surface, (b1) curvature of the surface with pixel size 5 m, (b2) curvature of the surface with 5 smoothing iterations and with pixel size 5 m, (c1, c2) ridge and scarp lines converted to points overlaying the curvature of the surface, (d) the birth (appearance) of the hole feature when the circles intersect (light blue), (e) the death of the hole feature (dark blue), and (f) the detected landslide shown as a polygon. (For interpretation of the references to colour in this fi gure legend, the reader is referred to the web version of this article.)

<!-- image -->

averaged out and seen as a fl at surface. With small pixels and without smoothing, the data will be noisy, where each small bump may be perceived as a ridge or scarp line (Fig. 10 (c1, c2)). Both the size of a rectangular fi lter and the number of smoothing iterations depend on the pixel size.

In this work, to test the robustness of our proposed PH method, we compared the results of our PH method using di ff erent pixel sizes (1 m,

5 m, 10 m) and several smoothing iterations (0, 2, 5, 10, 15, 20) with the results of three other ridge and scarp detection methods. For Study Area 5, we resample from 1.8 m to 2 m, 5 m, and 10 m pixel size DTM. For smoothing, we implemented a 3*3 rectangular fi lter, since larger fi lter sizes increase the level of blur, taking an averaged values approach (Babic and Mandic, 2003). The three ridge and scarp detection methods used for comparisons are:

Table 3

Characteristics of study areas.

| Study Area 5            | 2000 - 2005 Geology Puget Sound LiDAR Consortium Yes 1.8 m                                                                                      | 2017 Washington State Department of Natural Resources web portal by Compiling landslide inventory data through di ff erent methods and scales Yes No                                    |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Study Area 4            | 2007 State of Oregon Department of and Mineral Industries Yes 1 m quadrangle                                                                    | 2019 Dixie Mountain Oregon's Statewide Landslide Information Database digitized in Compiling landslide inventory data created by using LiDAR and protocol Burns and Madin, 2009 Yes Yes |
| Study Area 3            | 2015 - 2016 Colorado Geological Survey No 1 m Survey                                                                                            | 2015 Colorado Geological Compiling landslide information from 1:24,000-scale maps published geologic hazard maps of Colorado Yes No                                                     |
| Study Area 2            | of Oregon Department of Geology Mineral Industries quadrangle Statewide Landslide Database landslide inventory data using LiDAR and protocol by | Madin, 2009 Yes Yes                                                                                                                                                                     |
| Study Area 1            | 2007 Spatial Data Access State and Yes 1 m 2019 2019)                                                                                           | Gales Creek Oregon's Information LiDAR-derived protocol by Burns and Compiling created by Burns and                                                                                     |
|                         | 2006 - 2008 Pennsylvania (PASDA) Yes 1 m 2019 (Karimi et al.,                                                                                   | Detecting visually DTM. Mimics Madin, 2009 No No                                                                                                                                        |
| LIDAR (or LIDAR derived | Acquisition time Source Open source Horizontal ground resolution Existing landslides Acquisition time                                           | Source Acquisition method Open source Landslide deposits and scarp fl anks are separated                                                                                                |

12

Remote Sensing of Environment 246 (2020) 111816

- Method 1: Curvature algorithm using a threshold (Rana, 2006)
- Method 3: Geomorphon (Jasiewicz and Stepinski, 2013)
- Method 2: Curvature algorithm using a threshold calculated from multiplication of standard deviation (Pirotti and Tarolli, 2010)

For Methods 1 and 2, the next step after smoothing is calculating slope curvature algorithm, as shown in Fig. 10 (b1, b2). For the ridge and scarp lines, the surface shape should be convex with positive raster values, whereas for the channels, it should be concave with negative raster values. The ridge and scarp lines were retrieved by selecting pixels with high values from the output of the curvature algorithm. The threshold value depends on the pixel size and smoothing iteration by visually inspecting hillshades and other derivatives of DTM. In Method 1, we manually selected a suitable threshold value for each smoothing iteration and pixel size. In Method 2, the threshold value is a multiplication of standard deviation. We compared di ff erent multiplications of curvature's standard deviations (standard deviations *2, *1, *0.5, *0.25). Method 3 does not require any threshold values. The output of Method 3 is a raster with identi fi ed common local morphological elements where we selected ridges, shoulders, spurs to get ridge and scarp lines. Then, for all methods, as shown in Fig. 10 (c1, c2), the retrieved ridge and scarp lines were converted to points, and pixel sizes were used for distances between the points.

We used coordinates of the points stored in the matrix as input to compute the PH. We used the Topological Data Analysis (TDA) ' R ' package developed by Fasy et al. (2014) that contains tools for computing PD and analyzing the PH of the point data. The inputs to the algorithm are the ridge and scarp lines converted to a set of points, as shown in Fig. 10 (c1, c2). These data can be visualized in a PD, as presented in Fig. 11, that shows the birth (Fig. 10(d), light blue) and death (Fig. 10(e), dark blue) and therefore the lifespan (size of the circles or holes), or persistence, of the features.

Fig. 11. Persistence diagram of the points from Study Area 1 showing the birth and death of several persistent features. The points farther away from the diagonal line have longer ' lifespans ' .

<!-- image -->

13

Remote Sensing of Environment 246 (2020) 111816

Fig. 12. Representation of two circles as a point cloud (left) and persistence diagram of these circles (right).

<!-- image -->

Fig. 13. (a) Detected small landslides, (b) detected medium landslide, and (c) detected large landslide overlaid on landslides from landslide inventory maps.

<!-- image -->

The typical size of a landslide varies from few meters to several hundred meters in length or width. We use size and shape to detect persistent features of a landslide in an area. In order to retrieve circles from the output of the PH tool, we selected two parameters: birth and distance. The birth value indicates the time when a hole was created (Fig. 10(d)). If the birth value is too large, the tool will collect holes from points that are too far away from each other. If the birth value is too small, it collects only holes bound with very close points. This distance parameter can work as a noise reduction function. The distance parameter, which indicates the period between the birth and death values of an object, is represented by the size of the hole. In the PD shown in Fig. 12, this parameter is represented as the distance from the diagonal.

Once all holes are detected, we construct polygons (Fig. 10(f)) by their coordinates and overlay them on polygons of existing landslides: this requires prior knowledge about landslide size we are trying to identify. In addition, we can set a noise level, the distance between points, to form a landslide. Since this process is not computationally expensive, it is possible to set an average landslide size, if there is no prior knowledge, or several landslide sizes if the purpose is to detect landslides with di ff erent sizes. We set three sizes to identify landslides based on size and noise level. The fi rst size is small landslides with a little or no noise level, Fig. 13(a). This means that the distance between points for creating a circle is small. The second size is medium landslides with medium noise level, see Fig. 13(b). The third size is large landslide with large noise level, see Fig. 13(c). For large landslides we allow distance between points to be larger in comparison with smaller landslides.

## 6. Evaluation

## 6.1. Validation metrics

To evaluate our PH method, we overlaid PH detected landslides on mapped landslides and computed them through a confusion matrix for each study area. From the confusion matrix four common landslide validation metrics - accuracy, precision, recall, and Cohen's Kappa coe ffi cient - were measured:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

14

Remote Sensing of Environment 246 (2020) 111816

Fig. 14. Graphs representing comparison of the three methods with di ff erent pixel values and smoothing iterations as a measure of accuracy, precision, recall, and Cohen's Kappa coe ffi cient. Study Area 1: Pennsylvania.

<!-- image -->

<!-- formula-not-decoded -->

′ = - - P P P Cohen s Kappa coefficient 1 c exp exp

where

= P Accuracy c

<!-- formula-not-decoded -->

TP stands for true positive, an areas with existing landslides that were detected as landslides. FP stands for false positive, an areas with no existinglandslides that were detected as a landslides. FN stands for false negative, an areas with landslides that were not detected as a landslides. TN stands for true negative, an areas with no existing landslides that were not detected as a landslides. The values for all four metrics range from 0 to 1 where 0 represents worst and 1 represents best. Accuracy is a ratio of detected landslides and not landslides to the whole area. Depending on percentage of landslide areas in the whole area, accuracy can be biased. Precision is a ratio of correctly detected landslides to all detected landslides. Recall is a ratio of correctly detected landslides to all existing landslides. Cohen's Kappa coe ffi cient (Cohen, 1960; Tsangaratos and Ilia, 2016) is a measure of agreement between detection model and reality or a measure of how the result is signi fi cantly better than random (Jensen, 1996).

Table 4 Method, confusion matrix values, number of existing, detected, and intersected landslides, accuracy, precision, recall, Cohen's Kappa coe ffi cient for each study area.

|                                | Study Area 1   | Study Area 2   | Study Area 3   | Study Area 4   | Study Area 5   |
|--------------------------------|----------------|----------------|----------------|----------------|----------------|
| Method                         | 2              | 1              | 3              | 2              | 2              |
| Pixel size                     | 5 m            | 1 m            | 1 m            | 1 m            | 10 m           |
| Number of smoothing iterations | 5              | 2              | 5              | 15             | 1              |
| True positive                  | 5.17%          | 12.55%         | 14.51%         | 29.2%          | 6.%            |
| False Positive                 | 17.16%         | 22.21%         | 34.82%         | 14.66%         | 20.06%         |
| False Negative                 | 3.22%          | 6.54%          | 11.37%         | 15.96%         | 2.33%          |
| True negative                  | 74.44%         | 58.68%         | 39.28%         | 40.16%         | 71.59%         |
| Accuracy                       | 0.79           | 0.71           | 0.53           | 0.69           | 0.77           |
| Precision                      | 0.23           | 0.36           | 0.29           | 0.66           | 0.22           |
| Recall                         | 0.61           | 0.66           | 0.56           | 0.64           | 0.72           |
| Cohen's Kappa coe ffi cient    | 0.24           | 0.29           | 0.07           | 0.38           | 0.25           |

## 6.2. Results

We compared PH detected landslides to landslides inventory maps and computed accuracy, precision, recall, Cohen's Kappa coe ffi cient by using several smoothing iterations and di ff erent pixel sizes.

For Study Area 1 in Pennsylvania, we derived 55 polygons, with highest Cohen's Kappa coe ffi cient, that were deemed potential locations of existing landslides, as shown in Fig. 15. The highest Cohen's Kappa coe ffi cient was derived by using Method 2 with pixel size 5 m and with 5 smoothing iterations as shown in Fig. 14(k). The accuracy, precision, and recall are 0.79, 0.23, and 0.61, respectively. The Cohen's Kappa coe ffi cient is 0.24 (Table 4) which indicates fair agreement (Landis and Koch, 1977).

For Study Area 2 in Oregon, we derived 1514 polygons, with highest Cohen's Kappa coe ffi cient, that were deemed potential locations of existing landslides, as shown in Fig. 17. The highest Cohen's Kappa coe ffi cient was derived by using Method 1 with pixel size 1 m and with 2 smoothing iterations, as shown in Fig. 16(j). The accuracy, precision, and recall are 0.71, 0.36, and 0.66, respectively. The Cohen's Kappa coe ffi cient is 0.29 (Table 4) which indicates fair agreement (Landis and Koch, 1977).

For Study Area 3 in Colorado, we derived 494 polygons, with highest Cohen's Kappa coe ffi cient, that were deemed potential locations of existing landslides, as shown in Fig. 19. The highest Cohen's Kappa coe ffi cient was derived by using Method 3 with pixel size 1 m and with 5 smoothing iterations, as shown in Fig. 18(j). The accuracy, precision, and recall are 0.53, 0.29, and 0.56, respectively. The Cohen's Kappa coe ffi cient is 0.07 (Table 4) which indicates slight or no agreement (Landis and Koch, 1977).

For Study Area 4 in Oregon, we derived 1514 polygons, with highest

Fig. 15. Locations of the PH detected landslides and the mapped landslides. Study Area 1: Pennsylvania.

<!-- image -->

15

Remote Sensing of Environment 246 (2020) 111816

16

Remote Sensing of Environment 246 (2020) 111816

Fig. 16. Graphs representing comparison of the three methods with di ff erent pixel values and smoothing iterations as a measure of accuracy, precision, recall, and Cohen's Kappa coe ffi cient. Study Area 2: Oregon.

<!-- image -->

Cohen's Kappa coe ffi cient, that were deemed potential locations of existing landslides, as shown in Fig. 21. The highest Cohen's Kappa coe ffi cient was derived by using Method 2 with pixel size 1 m and with 15 smoothing iterations, as shown in Fig. 20(k). The accuracy, precision, and recall are 0.69, 0.66, and 0.64, respectively. The Cohen's Kappa coe ffi cient is 0.38 (Table 4) which indicates fair agreement (Landis and Koch, 1977).

For Study Area 5 in Washington, we derived 5274 polygons, with highest Cohen's Kappa coe ffi cient, that were deemed potential locations of existing landslides, as shown in Fig. 23. The highest Cohen' s Kappa coe ffi cient was derived by using Method 2 with pixel size 10 m and with one smoothing iteration, as shown in Fig. 22(k). The accuracy, precision, and recall are 0.77, 0.22, and 0.72, respectively. The Cohen's Kappa coe ffi cient is 0.25 (Table 4) which indicates fair agreement

(Landis and Koch, 1977).

## 6.3. Discussion

The results indicate that our proposed PH method is able to detect mass movements, but with a low mean accuracy of 0.7 when compared to existing landslides. The accuracy, precision, recall, and Cohen's Kappa coe ffi cient vary depending on the study area, ridge and scarp detection method, pixel size and smoothing iteration. For Study Areas 1 and 5, we obtain the highest accuracy of 0.91 with large pixel sizes and smoothing iterations (Figs. 14, 22). Study Area 2 has the highest Cohen's Kappa coe ffi cient of 0.29 and the accuracy is 0.80 (Method 1, pixel size 10 and iteration 2). The landslide detection approach from Bunn et al. (2019) that used the same dataset in the same study area provides an accuracy of 0.85. The di ff erence is that they focus on detection of landslide deposits while our work attempts to detect the whole landslide area. Study Area 3 has the lowest Cohen's Kappa

17

Remote Sensing of Environment 246 (2020) 111816

Fig. 17. Locations of the PH detected landslides and the mapped landslides. Study Area 2: Oregon.

<!-- image -->

coe ffi cient of 0.07. This area has the highest percentage of false positives (35%). This may be due to the lack of complete landslide inventory maps that do not include landslides that happened in 2014. Note that West Salt Creek landslide (White et al., 2015) is not in the landslide inventory maps (Fig. 24). Study Areas 4 and 5 were chosen to compare results with Booth et al. (2009) where Study Area 4 is balanced (45% total area is landslides) and Study Area 5 is unbalanced (only 9%). The obtained accuracies were 0.69 and 0.77 for Study Areas 4 and 5, respectively. The landslide detection approach from Booth et al. (2009) that used the same dataset in the same study area provides an accuracy of 0.68 for Study Area 4 and 0.87 for Study Area 5. In Study Area 4, the PH method slightly outperformed the approach by Booth et al. (2009) whereas in Study Area 5 the accuracy of the PH method was lower. However, in Study Area 5, several methods with di ff erent pixel sizes and smoothing iterations were able to get an accuracy higher than 0.9, see Fig. 22(a,b,c).

The pixel size and the number of smoothing iterations a ff ect the

18

Remote Sensing of Environment 246 (2020) 111816

Fig. 18. Graphs representing comparison of the three methods with di ff erent pixel values and smoothing iterations as a measure of accuracy, precision, recall, and Cohen's Kappa coe ffi cient. Study Area 3: Colorado.

<!-- image -->

ridge and scarp detection, wherewith a high pixel size smoothing fails to detect small ridges and scarps. Therefore, our PH method fails to detect small landslides and the number and areas of PH detected landslides decrease. This trend can be seen from the recall graph where recall drops when pixel size or number of iterations increases, see Figs. 14, 16, 18 (g, h, i). In addition to the pixel size, the prior knowledge of terrain and landslide size can increase detection accuracy.

PH method can be adjusted to detect landslides with di ff erent sizes.

It is worth mentioning that since areas with and without landslides, are di ff erent, data is unbalanced. Therefore, the high accuracy value can be misleading. For example, for Study Area 1 the accuracy is 0.91 with a large number of smoothing iterations and both small and large pixel sizes (Fig. 14(a,b,c)). This can be due to the fact that the unbalanced dataset detects fewer landslides increasing the values of TN

19

Remote Sensing of Environment 246 (2020) 111816

Fig. 19. Locations of the PH detected landslides and the mapped landslides. Study Area 3: Colorado.

<!-- image -->

and decreasing the values of FP. With pixel size 10 m and 20 smoothing iterations we cannot fi nd any landslides, the recall drops to 0, see Fig. 14 (i). Since one of the co-authors of this paper was the original mappers for the inventory used in Study Area 1 (see Karimi et al., 2019), we were able to perform additional assessments of false positive areas. Most false positives in the area are a product of man-made structures, or topographic patterns sourced from exposed and highly fractured, tilted bedrock; however, of interest is the large polygon that overlaps with the largest rockslide in the central part of Study Area 1. In the false positive portion of this large polygon is an area roughly equal to that of the largest rock-block slide that somewhat resembles a hummocky surface, though the resolution of the data makes it hard to distinguish, as well as a bulge on the downslope end that together might suggest a landslide in the pre-failure, or early failure, stages of development (Rotaru et al., 2007) (Fig. 25). Further fi eld investigation aided by geophysical surveys and high-resolution photogrammetry will be required to validate this further. If true, the accuracy estimates would then be 0.81. If validated as a true landslide, this could further suggest that PH method proposed herein is a promising fi rst step towards automated landslide inventory development.

The comparison results of the three methods show that we are able to derive ridge and scarp lines from all three methods and detect landslides from these lines. Method 1 requires more time and human intervention because it requires a threshold value which varies depending on pixel size and smoothing iteration. For Method 2 we compared several multiplications of curvature's standard deviation (std*2 *1 *0.5*0.25), where std.*0.5 gave the best results for all three study areas.

The choice of pixel size depends on the size of the detected landslides. With a small pixel size, we fi rst identify small and large ridgelines and scarps since they can precisely capture the shape of small landslides. However, small pixel sizes can fail to capture the boundaries of large landslides, or in addition to boundaries they may capture small ridgelines and scarps that are inside of large landslides (Fig. 24), generating false negatives. The smoothing iteration can remove some small ridgelines and scarps, but the process is not as e ff ective as using larger pixel size DTM.

The main purpose of smoothing iteration is to reduce noise which originated during the process of deriving DTM from LiDAR data. In addition, smoothing iteration removes small ridges and scarps and gives

20

Remote Sensing of Environment 246 (2020) 111816

Fig. 20. Graphs representing comparison of the three methods with di ff erent pixel values and smoothing iterations as a measure of accuracy, precision, recall, and Cohen's Kappa coe ffi cient. Study Area 4: Oregon.

<!-- image -->

higher values to large ridges and scarps. Hence, for Method 1 the threshold values for the same pixel vary based on smoothing iterations. Too many smoothing iterations can remove ridges and scarps that correspond to the boundaries of landslides and can a ff ect the detection accuracy. Except for the threshold values for Method 1, the only threshold values that are required are those for selecting landslide size and distance between points (noise). If there is no prior knowledge about landslide size, the average size or several landslide sizes can be used since the process is not computationally expensive.

While the PH method is quite accurate in detecting landslides with well-pronounced surface expressions at the top, bottom, left or right side of the landslide, such as a crown, headscarp, lateral margins, and toe, it is less accurate on landslides expressed by subtle terrain features (such as fl ow types). Strong surface features of landslides persist longer

21

Remote Sensing of Environment 246 (2020) 111816

Fig. 21. Locations of the PH detected landslides and the mapped landslides. Study Area 4: Oregon.

<!-- image -->

than those landslides where ridges/scarps are undetectable or those landslides so small that the persistence of their holes is limited.

The overall low accuracy is likely attributed to a combination of factors related to (1) the preservation and permanence of mass movements based on vegetation, climate, and human related contributions; (2) the types of mass movements being considered, and (3) the accuracy of landslide inventories used for assessment. In general, increased vegetative cover and rainfall, seen in Study Ares 1 and 2, increased the accuracy of the results in this study. This might seem somewhat surprising given that vegetation increases rates of weathering, particularly in areas of high physical erosion (Drever, 1994) that can allow surface morphological expressions of slope failures to be erased more quickly and e ff ectively, especially when in areas of higher total rainfall to act as an eroding and transport agent. Greater precipitation can also increase rates of silicate weathering (Brady et al., 1999) that could lead to thicker and deeper layers of highly weathered bedrock and soil. When saturated during a rain event, these thicklayers could result in mass movement events that are more deeply-seated, and therefore larger in area (Larsen et al., 2010), and can remain relatively well-preserved over time because of their size. These deeper landslides are likely to have di ff erent, often more-pronounced, surface features that are distinct from more common debris fl ow types seen in areas with thinner soil and weathered bedrock materials (Colorado), and their overall size means their features are expressed across a greater number of pixels within the DTMs, resulting in 1-dimensional loops with longer persistence. In such areas where deep-seated landslides are more frequent, rainfall can also trigger their reactivation (Floris and Bozzano, 2008), thus allowing for surface features to remain pronounced over time. Furthermore, while vegetation surfaces are removed from LiDAR data to arrive at DTM, vegetation size and local density negatively impact the resolution of the DTM as LiDAR points associated with the ground surface will be limited and sparse in distribution. The reason for the direct relationship between vegetative cover and rainfall versus detection accuracy could also have more to do with such limited and sparse

22

Remote Sensing of Environment 246 (2020) 111816

Fig. 22. Graphs representing comparison of the three methods with di ff erent pixel values and smoothing iterations as a measure of accuracy, precision, recall, and Cohen's Kappa coe ffi cient. Study Area 5: Washington.

<!-- image -->

point cloud density, where more sparse LiDAR points ignore locally subtle ridges and scarps that might not be relevant to the overall scale of di ff erent mass movements. An alternative reason might be that moved materials would be more predominantly engineering soils and debris in highly vegetated areas, as the biologic cover would promote soil forming processes. These Earth materials would be highly susceptible to erosion and their signal in topographic datasets dampened with rainfall events common in such vegetated areas. Since the proposed PH method is being used on elevation datasets, there is a higher likelihood of detecting deep-seated and relatively young landslides, because they result in a longer persistence of 1-dimensional homological loops (holes), while the opposite is true for older

23

Remote Sensing of Environment 246 (2020) 111816

Fig. 23. Locations of the PH detected landslides and the mapped landslides. Study Area 5: Washington.

<!-- image -->

landslides. Adjustments to the PH method to account for this discrimination in accuracy versus landslide age will require further consideration and testing as part of continued research in the future. Additionally, landslide preservation and permanence can be interrupted by the construction of roads, agricultural processes, and general land surface development by humans, which are present in all study areas.

Mass movement types and the nature of the moved material are highly variable, and thus can result in vastly di ff erent shapes, sizes, and pronunciation of surface expressions. Flow type mass movement deposits are often fl uidized by water and are therefore less likely to generate pronounced ridges and scarps de fi ning their bounds. The nature of the methodology and the resolution of the dataset can also ignore or mask the presence of such slope failures. Future adaptations of this PH methodshould consider soil type, slope, elevation, land cover, resolution, and even mean annual precipitation to better detect slope failures with di ff erent types of likely expressions. The current PH methodis most accurate for deep-seated landslides, or those with longlasting, or well-pronounced, surface expressions.

Many early landslide inventories were developed by digitizing old paper maps, possibly laden with spatial accuracy problems, with traditional fi eld techniques, that are not always regularly updated after the completion of the initial inventory. Inventories that rely on fi eld investigation can be hindered by accessibility and scale. Site accessibility can be limited to fi eld geologists identifying mass movements because of road access, forest density, and even landowner permissions. Additionally, the size of a landslide may be so large that in the fi eld it is confused with local trends in topography (Fig. 19). Since the development of slope failures is continuous over time, fi eld observations would

24

Remote Sensing of Environment 246 (2020) 111816

Fig. 24. Results of visual analysis of PH detected West Salt Creek landslide overlaid on satellite image. Colorado (Landslides were detected using Method 1, pixel size 1, and iteration 5).

<!-- image -->

have to be conducted regularly over all areas to properly maintain a landslide inventory, which is a living/evolving document. This is not feasible for geological surveys given limited resources and humanpower, and they must therefore turn to more modern methods that can meet, if not surpass, the accuracy of established protocols and methods. However, the accuracy of landslide inventories can also contribute to decreased landslide detection accuracy, as earlier inventories are used to assess the quality of new methods of detection.

Computational landslide inventories can be generated using aerial/ satellite imagery, traditional fi eldwork, and sometimes semi-automated/automated methods (Booth et al., 2009). However, even with these advancements careful consideration must be given to the datasets and algorithms deployed; imagery with poor resolution can mask the presence of slope failures, and the data type itself can also do the same. While standard visible spectrum data, and elevation data are common digital imagery, other types of data may add to the accuracy of detection. In highly vegetated areas that mask surface expressions of landslides, detection might bene fi t from looking at vegetative patterns over time, or temporal changes in the normalized di ff erence vegetation index (NDVI) that requires re fl ected visible and near-infrared (NIR) wavelengths (Roessner et al., 2014). Thermal infrared measurements can be used to derive the apparent thermal inertia (ATI) of a material (Xue and Cracknell, 1995), which is a proxy for real thermal inertia, or the measure of the rate at which heat is gained or lost to its environment. Measurements of the thermal inertia of soils/materials at the ground surface of a moved body may be able to detect di ff erences in moisture content, identifying pathways for increased fl uid fl ow by failure planes that de fi ne the bounds and internal architecture of slope failures.

In comparison with current landslide detection methods, which are focused on detection of scarps and deposits separately, the proposed PH method detects the shape of the whole landslides. Another advantage of the proposed method is that it can detect landslides at di ff erent scales. In other words, the proposed PH method allows detecting smaller new landslides within boundaries or next to the older landslides. Finally, PH is stable under perturbation in that the noise in the input data (ridge and scarp lines) will not dramatically change the detection accuracy.

25

Remote Sensing of Environment 246 (2020) 111816

Fig. 25. The Shickshinny Mountain slope in Study Area 1 with the largest rock-block slide identi fi ed inside the red polygon. The blue polygon identi fi es a suspected landslide that is in the initial phases of development. (For interpretation of the references to colour in this fi gure legend, the reader is referred to the web version of this article.)

<!-- image -->

## 7. Conclusions and future works

Landslides are a natural hazard that have a signi fi cant impact on lives and property; therefore, the ability to detect recent and possible landslide locations is critically important. In this paper, we tested the hypothesis that PH on LiDAR data can accurately detect landslides. The accessed LiDAR data was processed and provided detailed point cloud data about the terrain of the three study areas. We experimented by applying PH on these point cloud datasets to detect landslides and compared results with existing landslide inventory maps of Oregon, Washington, and Colorado and manually detected Landslides from Pennsylvania. For validation of the results, we measured four metrics: accuracy, precision, recall, and Cohen's Kappa coe ffi cient. The results show that for Study Area 1, the accuracy, precision, recall and Cohen's Kappa coe ffi cient are 0.79, 0.23, 0.61 and 0.24, respectively. For Study Area 2, the accuracy, precision, recall and Cohen's Kappa coe ffi cient are 0.71, 0.36, 0.66 and 0.29, respectively. For Study Area 3, the accuracy, precision, recall and Cohen's Kappa coe ffi cient are 0.53, 0.29, 0.56 and 0.07, respectively. For Study Area 4, the accuracy, precision, recall and Cohen's Kappa coe ffi cient are 0.69, 0.66, 0.64 and 0.38, respectively and for Study Area 5, the accuracy, precision, recall and Cohen's Kappa coe ffi cient are 0.77, 0.22, 0.72 and 0.25, respectively. We discussed our reasons for these results, such as the low accuracy of existing landslide inventory maps. Considering the importance of accurately determining ridges and scarps for detection of landslides, one future work is to take a convolutional fi lters approach and apply it to DTM. Another future work is to compare and analyze the result of this current work with the result of an extended version of the proposed PH where it takes fused data from LiDAR and other datasets, such as NDVI, visual/NIR satellite imagery, apparent thermal inertia, vegetative cover, bedrock lithology, and/or soil types to detect landslides. A third future work is to use features of landslides detected through PH as labels for classi fi cation machine learning techniques for a fully automated method.

## Author contribution statement

Meirman Syzdykbayev: Visualization, Original Draft, Validation, Software, Conceptualization, Methodology, Data Curation, Formal analysis

Bobak Karimi: Visualization, Original Draft, Investigation, Validation, Conceptualization, Formal analysis

Hassan A. Karimi: Supervision, Writing - Review &amp; Editing, Formal analysis

## Declaration of competing interest

The authors declare that they have no known competing fi nancial interests or personal relationships that could have appeared to in fl uence the work reported in this paper.

## References

- Ahmad, Z.A., Guey, C.W., Sze, L.T., Chet, K.V., 2017. Scalable and cost e ff ective high resolution digital elevation model extraction method for slope ' s stability assessment. Adv. Sci. Lett. https://doi.org/10.1166/asl.2017.8367.
- Alexander, D., 1992. On the causes of landslides: human activities, perception, and natural processes. Environ. Geol. Water Sci. https://doi.org/10.1007/BF01706160.
- Ardizzone, F., Cardinali, M., Galli, M., Guzzetti, F., Reichenbach, P., 2007. Identi fi cation and mapping of recent rainfall-induced landslides using elevation data collected by airborne Lidar. Nat. Hazards Earth Syst. Sci. https://doi.org/10.5194/nhess-7-6372007.
- [Arguez, A., Durre, I., Applequist, S., Squires, M., Vose, R., Yin, X., Bilotta, R., 2010. NOAA ' s US Climate Normals (1981 - 2010). NOAA Natl. Centers Environ. Inf. 10, V5PN93JP.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0020)
- Babic, Z.V., Mandic, D.P., 2003. An e ffi cient noise removal and edge preserving convolution fi lter. In: 6th International Conference on Telecommunications in Modern Satellite, Cable and Broadcasting Service, TELSIKS 2003- Proceedings, https://doi. org/10.1109/TELSKS.2003.1246284.
- Booth, A.M., Roering, J.J., Perron, J.T., 2009. Automated landslide mapping using spectral analysis and high-resolution topographic data: Puget Sound lowlands, Washington, and Portland Hills, Oregon. Geomorphology. https://doi.org/10.1016/j. geomorph.2009.02.027.
- Brady, P.V., Dorn, R.I., Brazel, A.J., Clark, J., Moore, R.B., Glidewell, T., 1999. Direct measurement of the combined e ff ects of lichen, rainfall, and temperature onsilicate weathering. Geochim. Cosmochim. Acta. https://doi.org/10.1016/S0016-7037(99) 00251-3.
- Bunn, M.D., Leshchinsky, B.A., Olsen, M.J., Booth, A., 2019. A simpli fi ed, object-based framework for e ffi cient landslide inventorying using LIDAR digital elevation model derivatives. Remote Sens. https://doi.org/10.3390/rs11030303.
- [Burns, W.J., Mickelson, K.A., 2016. Protocol for Deep Landslide Susceptibility Mapping. Oregon Department of Geology and Mineral Industries.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0045)
- [Burns, W.J., Madin, I., 2009. Protocol for inventory mapping of landslide deposits from light detection and ranging (LiDAR) imagery. Oregon Department of Geology and Mineral Industries, Portland, Portland, OR.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf3850)
- [Burns, W.J., Maidin, I.P., Ma, L., 2008. Statewide landslide information database for Oregon (SLIDO), release 1. In: 2008 Joint Meeting of the Geological Society of America, Soil Science Society of America, American Society of Agronomy, Crop Science Society of America, Gulf Coast Association of Geological Societies with the Gulf Coast Section of SEPM.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0050)
- [Burns, W.J., Duplantis, S., Mickelson, K.A., Spritzer, J.M., Wells, R.E., 2012. Landslide inventory maps of the Gales Creek Quadrangle, Washington County, Oregon. In: Oregon Dep. Geol. Miner. Ind. Portland, Oregon, Interpret. Map Ser. Scale. 1. pp. 8000.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0055)
- Carlsson, G., 2009. Topology and data. Bull. Am. Math. Soc. https://doi.org/10.1090/ S0273-0979-09-01249-X.
- Casagli, N., Cigna, F., Bianchini, S., Hölbling, D., Füreder, P., Righini, G., Del Conte, S., Friedl, B., Schneiderbauer, S., Iasio, C., Vlcko, J., Greif, V., Proske, H., Granica, K., Falco, S., Lozzi, S., Mora, O., Arnaud, A., Novali, F., Bianchi, M., 2016. Landslide mapping and monitoring by using radar and optical remote sensing: examples from the EC-FP7 project SAFER. Remote Sens. Appl. Soc. Environ. https://doi.org/10. 1016/j.rsase.2016.07.001.
- [Casale, R., Fantechi, R., Flageollet, J.-C., 1994. Temporal Occurrence and Forecasting of Landslides in the European Community. European Commission.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0070)
- Chalkias, C., Ferentinou, M., Polykretis, C., 2014. GIS-based landslide susceptibility mapping on the Peloponnese Peninsula, Greece. Geosci. https://doi.org/10.3390/ geosciences4030176.

Chen, Z., Gao, B., Devereux, B., 2017. State-of-the-art: DTM generation using airborne LIDAR data. Sensors (Switzerland). https://doi.org/10.3390/s17010150. Cohen, J., 1960. A coe ffi cient of agreement for nominal scales. Educ. Psychol. Meas.

26

Remote Sensing of Environment 246 (2020) 111816

[https://doi.org/10.1177/001316446002000104.](https://doi.org/10.1177/001316446002000104)

- Colorado Geological Survey, 2017. Colorado landslide inventory [WWW document]. Database. URL. http://coloradogeologicalsurvey.org/geologic-hazards/landslides/ colorado-landslide-inventory/.
- Corsini, A., Borgatti, L., Cervi, F., Dahne, A., Ronchetti, F., Sterzai, P., 2009. Estimating mass-wasting processes in active earth slides - earth fl ows with time-series of highresolution DEMs from photogrammetry and airborne LiDAR. Nat. Hazards Earth Syst. Sci. 9, 433 - 439. https://doi.org/10.5194/nhess-9-433-2009.
- Drever, J.I., 1994. The e ff ect of land plants on weathering rates of silicate minerals. Geochim. Cosmochim. Acta. https://doi.org/10.1016/0016-7037(94)90013-2. Edelsbrunner, H., Harer, J., 2008. Persistence Homology - a Survey. Contemp. Math.
- Fanos, A.M., Pradhan, B., Mansor, S., Yuso ff , Z.M., Abdullah, A.F. bin, 2018. A hybrid model using machine learning methods and GIS for potential rockfall source identifi cation from airborne laser scanning data. Landslides. https://doi.org/10.1007/ s10346-018-0990-4.
- [Fasy, B.T., Kim, J., Lecci, F., Maria, C., 2014. Introduction to the R Package TDA. arXiv Prepr. arXiv1411.1830.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0115)
- Ferri, M., Tomba, I., Visotti, A., Stanganelli, I., 2017. A feasibility study for a persistent homology-based k-nearest neighbor search algorithm in melanoma detection. J. Math. Imaging Vis. https://doi.org/10.1007/s10851-016-0680-6.
- Floris, M., Bozzano, F., 2008. Evaluation of landslide reactivation: a modi fi ed rainfall threshold model based on historical records of rainfall and landslides. Geomorphology 94, 40 - 57. https://doi.org/10.1016/j.geomorph.2007.04.009.
- Fonstad, M.A., Dietrich, J.T., Courville, B.C., Jensen, J.L., Carbonneau, P.E., 2013. Topographic structure from motion: a new development in photogrammetric measurement. Earth Surf. Process. Landforms. https://doi.org/10.1002/esp.3366.
- Ghorbanzadeh, O., Blaschke, T., Gholamnia, K., Meena, S.R., Tiede, D., Aryal, J., 2019. Evaluation of di ff erent machine learning methods and deep-learning convolutional neural networks for landslide detection. Remote Sens. https://doi.org/10.3390/ rs11020196.
- Ghrist, R., 2018. Homological algebra and data. https://doi.org/10.1090/pcms/025/06. Guswa, A.J., 2008. The in fl uence of climate on root depth: a carbon cost-bene fi t analysis. Water Resour. Res. https://doi.org/10.1029/2007WR006384.
- Guzzetti, F., Mondini, A.C., Cardinali, M., Fiorucci, F., Santangelo, M., Chang, K.T., 2012. Landslide inventory maps: new tools for an old problem. Earth-Science Rev. https:// doi.org/10.1016/j.earscirev.2012.02.001.
- [Haugerud, R.A., Harding, D.J., Johnson, S.Y., Harless, J.L., Weaver, C.S., Sherrod, B.L., 2003. High-resolution lidar topography of the Puget Lowland, Washington. GSA Today 13, 4 - 10.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0155)
- [Highland, L.M., Bobrowsky, P., 2008. The landslide Handbook - a guide to understanding landslides. US Geol. Surv. Circ. 3 - 5.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0160)
- Hölbling, D., Füreder, P., Antolini, F., Cigna, F., Casagli, N., Lang, S., 2012. A semi-automated object-based approach for landslide detection validated by persistent scatterer interferometry measures and landslide inventories. Remote Sens. https://doi. org/10.3390/rs4051310.
- Homer, C., Huang, C., Yang, L., Wylie, B., Coan, M., 2004. Development of a 2001 National Land-Cover Database for the United States. Photogramm. Eng. Remote Sensing. 73, 337 - 341. https://doi.org/10.14358/PERS.70.7.829.
- [Islambekov, U., Gel, Y.R., 2019. Unsupervised space - time clustering using persistent homology. Environmetrics 30, e 2539.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0175)
- Jasiewicz, J., Stepinski, T.F., 2013. Geomorphons-a pattern recognition approach to classi fi cation and mapping of landforms. Geomorphology. https://doi.org/10.1016/j. geomorph.2012.11.005.
- [Jensen, J.R., 1996. Introductory Digital Image Processing: A Remote Sensing Perspective. Second Edition, Introductory Digital Image Processing: A Remote Sensing Perspective, Second edition. .](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0185)
- [Jenson, S.K., Domingue, J.O., 1988. Extracting topographic structure from digital elevation data for geographic information system analysis. Photogramm. Eng. Remote Sensing 54, 1593 - 1600.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0190)
- Karimi, B., Yanchuck, M., Foust, J., 2019. A new landslide inventory and improved susceptibility model for northeastern Pennsylvania. Environ. Geosci. https://doi.org/10. 1306/eg.09191919008.
- Landis, J.R., Koch, G.G., 1977. The measurement of observer agreement for categorical data. Biometrics. https://doi.org/10.2307/2529310.
- Larsen, I.J., Montgomery, D.R., Korup, O., 2010. Landslide erosion controlled by hillslope material. Nat. Geosci. https://doi.org/10.1038/ngeo776.
- Leshchinsky, B.A., Olsen, M.J., Tanyu, B.F., 2015. Contour connection method for automated identi fi cation and classi fi cation of landslide deposits. Comput. Geosci. https:// doi.org/10.1016/j.cageo.2014.10.007.
- Li, C., Ovsjanikov, M., Chazal, F., 2014. Persistence-based structural recognition. In: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, https://doi.org/10.1109/CVPR.2014.257.
- Liao, W.H., 2010. Region description using extended local ternary patterns. In: Proceedings - International Conference on Pattern Recognition, https://doi.org/10. 1109/ICPR.2010.251.
- Lucieer, A., Jong, S.M.d., Turner, D., 2014. Mapping landslide displacements using structure from motion (SfM) and image correlation of multi-temporal UAV photography. Prog. Phys. Geogr. https://doi.org/10.1177/0309133313515293.
- Mahrooghy, M., Aanstoos, J.V., Nobrega, R.A.A., Hasan, K., Prasad, S., Younan, N.H., 2015. A machine learning framework for detecting landslides on earthen levees using spaceborne SAR imagery. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. https://doi. org/10.1109/JSTARS.2015.2427337.
- Mayr, A., Rutzinger, M., Bremer, M., Oude Elberink, S., Stumpf, F., Geitner, C., 2017. Object-based classi fi cation of terrestrial laser scanning point clouds for landslide monitoring. Photogramm. Rec. https://doi.org/10.1111/phor.12215.
- Mayr, A., Rutzinger, M., Geitner, C., 2018. Multitemporal analysis of objects in 3D point

- clouds for landslide monitoring, in: International Archives of the Photogrammetry. Remote Sensing and Spatial Information Sciences - ISPRS Archives. https://doi.org/ 10.5194/isprs-archives-XLII-2-691-2018.
- [Mezaal, M.R., Pradhan, B., 2018. Data mining-aided automatic landslide detection using airborne laser scanning data in densely forested tropical areas. 대 한 원 격 탐 사 학 회 지 34, 45 - 74.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0245)
- Mezaal, M.R., Pradhan, B., Sameen, M.I., Shafri, H.Z.M., Yuso ff , Z.M., 2017. Optimized neural architecture for automatic landslide detection from high-resolution airborne laser scanning data. Appl. Sci. https://doi.org/10.3390/app7070730.
- Moore, I.D., Grayson, R.B., Ladson, A.R., 1991. Digital terrain modelling: a review of hydrological, geomorphological, and biological applications. Hydrol. Process. https://doi.org/10.1002/hyp.3360050103.
- Nichol, J., Wong, M.S., 2005. Satellite remote sensing for detailed landslide inventories using change detection and image fusion. Int. J. Remote Sens. https://doi.org/10. 1080/01431160512331314047.
- [Niethammer, U., Rothmund, S., Joswig, M., 2008. UAV-based remote sensing of the slowmoving landslide Super-Sauze. In: Proc. Int. Conf. Landslide Process. From Geomorpholgic Mapp. To Dyn. Model. Strasbourg, CERG Ed.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0265)
- Niethammer, U., James, M.R., Rothmund, S., Travelletti, J., Joswig, M., 2012. UAV-based remote sensing of the Super-Sauze landslide: evaluation and results. Eng. Geol. https://doi.org/10.1016/j.enggeo.2011.03.012.
- Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., Harrington, H.A., 2017. A roadmap for the computation of persistent homology. EPJ Data Sci. https://doi.org/10.1140/ epjds/s13688-017-0109-5.
- Pereira, C.M.M., De Mello, R.F., 2015. Persistent homology for time series and spatial data clustering. Expert Syst. Appl. https://doi.org/10.1016/j.eswa.2015.04.010.
- Pirotti, F., Tarolli, P., 2010. Suitability of LiDAR point density and derived landform curvature maps for channel network extraction. Hydrol. Process. https://doi.org/10. 1002/hyp.7582.
- [Pomeroy, J.S., 1982. Landslides in the Greater Pittsburgh Region, Pennsylvania. US Government Printing O ffi ce.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0290)
- Popescu, M.E., 1994. A suggested method for reporting landslide causes. Bull. Int. Assoc. Eng. Geol. - Bull. l ' Association Int. Géologie l ' Ingénieur. https://doi.org/10.1007/ BF02594958.
- Rana, S., 2006. Use of plan curvature variations for the identi fi cation of ridges and channels on DEM. In: Progress in Spatial Data Handling -12th International Symposium on Spatial Data Handling, SDH 2006, https://doi.org/10.1007/3-54035589-8\_49.

27

Remote Sensing of Environment 246 (2020) 111816

- Reybold, W.U., TeSelle, G.W., 1989. Soil geographic data bases. J. Soil Water Conserv. 44, 28 - 29.
- [Roessner, S., Behling, R., Segl, K., Golovko, D., Wetzel, H.-U., Kaufmann, H., 2014. Automated remote sensing based landslide detection for dynamic landslide inventories. In: Landslide Science for a Safer Geoenvironment. Springer, pp. 345 - 350.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0310)
- Rotaru, A., Oajdea, D., R ă ileanu, P., 2007. Analysis of the landslide movements. Int. J. Geol.
- Santangelo, M., Marchesini, I., Bucci, F., Cardinali, M., Fiorucci, F., Guzzetti, F., 2015. An approach to reduce mapping errors in the production of landslide inventory maps. Nat. Hazards Earth Syst. Sci. https://doi.org/10.5194/nhess-15-2111-2015.
- [Sarikhan, I.Y., Stanton, K.M., 2008. Washington Geological Survey GIS statewide landslide database - from design to implementation. In: Digital Mapping Techniques ' 08 - Workshop Proceedings, Moscow Idaho, pp. 1298 - 2009.](http://refhub.elsevier.com/S0034-4257(20)30186-3/rf0320)
- Schenk, H.J., Jackson, R.B., 2002. The global biogeography of roots. Ecol. Monogr. https://doi.org/10.1890/0012-9615(2002)072[0311:TGBOR]2.0.CO;2.
- Schulz, W.H., 2007. Landslide susceptibility revealed by LIDAR imagery and historical records, Seattle, Washington. Eng. Geol. https://doi.org/10.1016/j.enggeo.2006.09. 019.
- Tsangaratos, P., Ilia, I., 2016. Landslide susceptibility mapping using a modi fi ed decision tree classi fi er in the Xanthi Perfection, Greece. Landslides. https://doi.org/10.1007/ s10346-015-0565-6.
- U.S. Geological Survey (USGS), user\_community E Landslide susceptibility [WWW document]. URL n.d.. https://www.arcgis.com/home/item.html?id= b3fa4e3c494040b491485dbb7d038c8a.
- Varnes, D.J., 1978. Slope movement types and processes. Spec. Rep. 176, 11 - 33. White, J.L., Morgan, M.L., Berry, K.A., 2015. The West Salt Creek landslide: a catastrophic rockslide and rock/debris avalanche in Mesa County, Colorado. Color. Geol. Surv. Bull. 55, 45.
- Xia, K., Wei, G.W., 2014. Persistent homology analysis of protein structure, fl exibility, and folding. Int. j. numer. method. biomed. eng. https://doi.org/10.1002/cnm.2655.
- Xue, Y., Cracknell, A.P., 1995. Advanced thermal inertia modelling. Int. J. Remote Sens. https://doi.org/10.1080/01431169508954411.
- Yalcin, A., 2008. GIS-based landslide susceptibility mapping using analytical hierarchy process and bivariate statistics in Ardesen (Turkey): comparisons of results and con fi rmations. Catena. https://doi.org/10.1016/j.catena.2007.01.003.
- Zevenbergen, L.W., Thorne, C.R., 1987. Quantitative analysis of land surface topography. Earth Surf. Process. Landforms. https://doi.org/10.1002/esp.3290120107.