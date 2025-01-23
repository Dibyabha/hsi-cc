# hsi-cc
This is the official repository for the paper "CORRELATION-BASED BAND SELECTION FOR HYPERSPECTRAL IMAGE CLASSIFICATION" and has been submitted to the IGARSS 2025 conference (IEEE International Geoscience and Remote Sensing Symposium).

# Abstract
Hyperspectral images offer extensive spectral information about ground objects across multiple spectral bands. However, the large volume of data can pose challenges during processing. Typically, adjacent bands in hyperspectral data are highly correlated, leading to the use of only a few selected bands for various applications. In this work, we present a correlation-based band selection approach for hyperspectral image classification. Our approach calculates the average correlation between bands using correlation coefficients to identify the relationships among different bands. Afterward, we select a subset of bands by analyzing the average correlation and applying a threshold-based method. This allows us to isolate and retain bands that exhibit
lower inter-band dependencies, ensuring that the selected bands provide diverse and non-redundant information. We evaluate our proposed approach on two standard benchmark datasets: Pavia University (PA) and Salinas Valley (SA), focusing on image classification tasks. The experimental results demonstrate that our method performs competitively with other standard band selection approaches.

# Methodology
The proposed work computes band correlation using Correlation Coefficient (CC) to determine the relationships between different spectral bands of the datasets. This metric was chosen to quantify the inter-band dependencies of the benchmark datasets. CC quantifies the linear relationship between two variables (or bands in our study) and describes how they move in relation to each other. It ranges from -1 to 1, where -1 indicates a perfect negative correlation (as one variable increases, the other decreases), and 1 indicates a perfect positive correlation (as one variable increases, the other also increases). Mathematically, CC can be represented as $r_{XY}$ where X and Y are the two variables (or bands) and can be formulated as

![Formula](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}&space;r_{XY}=\frac{\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2\sum_{i=1}^{n}(Y_i-\bar{Y})^2}})

where Xi and Yi are the individual samples and $\bar{X}$ and $\bar{Y}$ are the mean values of variables X and Y respectively, and n is the number of samples (pixels) in each variable (or band) after pre-processing. The band correlation data was stored as a matrix for future access. Subsequently, Average Band Correlation (ABC) is computed. The ABC for band i is defined as the mean of the absolute correlation of band i (Bi) with band j (Bj ), where j varies from 1 to N and j!=i.

![Formula](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\text{ABC}_i=\frac{1}{N-1}\sum_{j=1,j\neq&space;i}^{N}\left|r_{B_i,B_j}\right|)

where r(Bi, Bj ) represents the correlation coefficient between Bi and Bj , and | · | denotes the absolute value. This process is repeated for each Bi, where i ∈ {1, . . . , N }. We experimentally set a threshold of 0.65 for the average band correlation (ABC). Bands with ABC less than the threshold were selected and, these selected bands were then extracted from the datasets. This approach allowed us to isolate and retain bands that exhibited lower inter-band dependencies, ensuring that the retained bands provided diverse and nonredundant information.

# Support
Feel free to contact : dibyabhadeb@gmail.com
