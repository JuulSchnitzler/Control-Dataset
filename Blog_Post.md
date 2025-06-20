# Control Dataset for Testing Scaling Invariance in Random Forest
Juul Schnitzler
5094917 

## Motivation

Tree-based models such as Decision Trees and Random Forests are widely used in practice. Often people assume that these models are invariant under monotonic transformations of input features, for example, rescaling a feature should not affect the model’s predictions.

In a paper by Su et al. (2011), the authors look into the use of tree-based models in nursing research. They explicitly mention the advantage of tree-based methods being invariant to monotone transformations of predictors. This means the paper relies on the claim being true and promotes the use of tree-based methods in applications such as quality-of-life data.

A paper by Galili & Meilijson (2016) examines the same claim more critically. They mention that while it's widely believed that tree models are invariant under monotonic transformation of features, *“this statement may be false when predicting new observations with values that were not seen in the training-set and are close to the location of the split point of a tree rule”* (Galili & Meilijson, 2016, p. 1). In other words, they hypothesise that monotonic transformations can in fact impact model behaviour in some cases, especially when the test data lies close to a learned decision threshold.

Verifying claims about machine and deep learning models is important. If the commonly assumed invariance of Random Forests to feature scaling does not hold in practice, this can lead to incorrect assumptions when preprocessing data or interpreting model outputs. Since Random Forests are often chosen for their ease of use, it is worth testing whether the invariance holds under controlled conditions. In this blog post, we create a control dataset to evaluate the aforementioned claim made by Galili & Meilijson (2016). Specifically, we create a dataset that contains test data that are near split points.


## Control Dataset
The code for creating the control dataset can be found in the [code repository](https://github.com/JuulSchnitzler/Control-Dataset). 

### Creating the Control Dataset
To test the invariance of Random Forests to feature scaling, specifically for unseen test samples near a learned decision threshold, we constructed a controlled dataset using the following steps:

1) **Training Set**
   We generate a training set with two features, `X1` and `X2`, drawn independently from a uniform distribution over [0, 10]. The binary target variable `y` is defined as:

    ```python
        if X1 + X2 > 10:
            y = 1
        else: 
            y = 0
    ```
2. **Random Test Set**  
   A test set `test_random` is created using the same distribution as the training data. It provides a baseline for evaluating the effect of monotonic transformations. 

3. **Near-Split Test Set**  
   A test set `test_near` is created by sampling `X1` and `X2` independently from a uniform distribution over [4.9, 5.1], so that `X1 + X2` is likely to fall close to the split at `X1 + X2 = 10`.

4. **Monotonic Transformation**  
   We apply a strictly monotonic transformation only to the test sets, scaling `X1` by 2 and `X2` by 0.5. This results in `test_random_scaled`, and `test_near_scaled`. These transformations preserve input ordering but change the feature scales.  

5. **Saving Datasets**  
   All five datasets are saved as `.csv` files. 


With this setup, we train a Random Forest model on the original training set and evaluate its predictions across all test variants. Since the transformation is applied only to the test sets, we can directly assess whether a monotonic change in input representation affects model behavior. By comparing results on `test_random` vs. `test_random_scaled`, and `test_near` vs. `test_near_scaled`, we can isolate whether monotonic transformations affect predictions, especially for inputs near learned decision thresholds. 

### Example
The following figures show (part of) what the datasets can look like in the described setup.


**Figure 1:** Original training set:
<p float="left">
  <img src="https://github.com/user-attachments/assets/a580b014-9317-4847-b84a-34427324e7bc" width="20%" />
</p>


**Figure 2:** Original test set (left) and near-split test set (right):
<p float="left">
  <img src="https://github.com/user-attachments/assets/c02e46a1-f289-4f4f-8e31-40008be39e13" width="20%" />
  <img src="https://github.com/user-attachments/assets/0980a385-8de0-46ad-a92d-59eed9f139db" width="21.5%" />
</p>


**Figure 3:** Scaled test set (left) and near-split scaled test set (right):
<p float="left">
  <img src="https://github.com/user-attachments/assets/8e40ecbe-0698-41ff-9a54-a455af996dae" width="20%" />
  <img src="https://github.com/user-attachments/assets/46ededf3-965d-49f7-b993-bfbeabb69730" width="20%" />
</p>
 

## Conclusion
A control dataset allows us to test a specific claim about the behavior of a machine or deep learning model. We created a control dataset for testing the hypothesis raised by Galili & Meilijson (2016), who argue that even monotonic transformations, while preserving the order of inputs, may still affect model predictions when test samples lie close to a decision boundary.

We constructed the dataset so that the training data was straightforward, and the test data deliberately included inputs near the learned decision threshold. This setup provides the conditions under which Random Forests may not be invariant to scaling. The provided control dataset is rather simple, but it can be easily extended (adding more features, changing scalars, changing dataset size, or placing more samples around the split) to explore the effects further.

While our experiment targeted Random Forests, the same approach can be used to test other models that rely on tree-based splits or make similar assumptions about invariance to feature scaling.




## References
* Su X, Azuero A, Cho J, Kvale E, Meneses KM, McNees MP. An introduction to tree-structured modeling with application to quality of life data. *Nurs Res.* 2011 Jul-Aug;60(4):247-255. [https://doi.org/10.1097/NNR.0b013e318221f9bc](https://doi.org/10.1097/NNR.0b013e318221f9bc)

* Galili T, Meilijson I. Splitting matters: Decision trees, monotone transformations and tree-based models. *The American Statistician.* 2016;70(1):32-36. [https://doi.org/10.1080/00031305.2015.1086684](https://doi.org/10.1080/00031305.2015.1086684)
