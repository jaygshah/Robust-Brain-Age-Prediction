# Robust Brain Age Prediction
Official PyTorch implementation of [**Ordinal Classification with Distance Regularization for Robust Brain Age Prediction**](https://openaccess.thecvf.com/content/WACV2024/html/Shah_Ordinal_Classification_With_Distance_Regularization_for_Robust_Brain_Age_Prediction_WACV_2024_paper.html) [WACV 2024]

[Jay Shah](https://www.public.asu.edu/~jgshah1/)<sup>1,2</sup>,
[Md Mahfuzur Rahman Siddiquee](https://mrahmans.me/)<sup>1,2</sup>
[Yi Su](https://scholar.google.com/citations?user=vdZKSEIAAAAJ&hl=en)<sup>1,2,3</sup>,
[Teresa Wu](https://labs.engineering.asu.edu/wulab/person/teresa-wu-2/)<sup>1,2</sup>
[Baoxin Li](https://www.public.asu.edu/~bli24/)<sup>1,2</sup>

<sup>1</sup>ASU-Mayo Center for Innovative Imaging,
<sup>2</sup>Arizona State University,
<sup>3</sup>Banner Alzheimer’s Institute

---
## Abstract 
Age is one of the major known risk factors for Alzheimer's Disease (AD). Detecting AD early is crucial for effective treatment and preventing irreversible brain damage. Brain age, a measure derived from brain imaging reflecting structural changes due to aging, may have the potential to identify AD onset, assess disease risk, and plan targeted interventions. Deep learning-based regression techniques to predict brain age from magnetic resonance imaging (MRI) scans have shown great accuracy recently. However, these methods are subject to an inherent regression to the mean effect, which causes a systematic bias resulting in an overestimation of brain age in young subjects and underestimation in old subjects. This weakens the reliability of predicted brain age as a valid biomarker for downstream clinical applications. Here, we reformulate the brain age prediction task from regression to classification to address the issue of systematic bias. Recognizing the importance of preserving ordinal information from ages to understand aging trajectory and monitor aging longitudinally, we propose a novel **ORdinal Distance Encoded Regularization (ORDER)** loss that incorporates the order of age labels, enhancing the model's ability to capture age-related patterns. Extensive experiments and ablation studies demonstrate that this framework reduces systematic bias, outperforms state-of-art methods by statistically significant margins, and can better capture subtle differences between clinical groups in an independent AD dataset.

<p align="center">
<img src="imgs/order_loss.png" width=62% height=62% 
class="center">
</p>
Cross entropy (left) encourages the model to learn high entropy feature representations where embeddings are spread out but fails to capture ordinal information from labels. ORDER loss + cross entropy (right) preserves ordinality by spreading the features proportional to Manhattan distance between normalized features weighted by absolute age difference.

<p align="center">
<img src="imgs/all_losses.png" width=62% height=62% 
class="center">
</p>
t-SNE visualization of embeddings from models’ penultimate layer