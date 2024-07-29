# NeuroBeatsDL
A deep learning exploration of EEG-processed brain topographical maps, uncovering the impact of binaural beats on neural activity.
## Abstract

This project integrates deep learning with image processing and computer vision techniques to quantify the impact of binaural beats on attention span by harnessing EEG data collected across 42 electrodes on the human scalp. The methodology extends beyond traditional signal analysis by preprocessing the EEG signals into brain heat maps—topographical representations of brain activity—leveraged as input into neural network architectures like conditional Generative Adversarial Networks (GANs), 3D Convolutional Neural Networks, autoencoder transformer models, and diffusion models. The networks are designed to anticipate how the brain heat map would evolve in the ensuing moment, hence providing a dynamic measure of attention span, indicated by the color-coded contour lines of the active brain regions. By comparing these predictions in the presence and absence of binaural audio, the project performs a nuanced exploration of how auditory stimuli modulate human neural activity, highlighting the potential of combining deep learning with image processing to understand attention mechanisms.

## Dataset

The dataset involves raw EEG signal files recorded in BIOSEMI Active Two Brain recorder and exported in (.BDF) format. The data consists of deidentified data for 80 participants, recorded for two sessions of 30 minutes each, where one session had just pure tone, while the other session had binaural beats. The EEG signals have a sampling frequency of 512 Hz. Each session consisted of 33 minutes, where the participant had to answer 1200 cognitive ability questions, and their brain activity was recorded via EEG, and their wrong answers were counted. These 1200 questions had a stimulus window of 1500 ms and a response window of 150 ms, hence the total duration of the question was 1.65 seconds.
<table>
  <tr>
    <td><img src="EEG_3D_Rotation.gif" alt="Sample Brain map" width="300"/></td>
    <td><img src="Project pictures/10-20 system.png" alt="EEG System" width="300"/></td>
  </tr>
</table>


## Methodology

### Data Preprocessing
1. **Brain Rate Calculation:**
   - Brain rate is defined as a sum of the mean frequencies of brain oscillations weighted over the EEG bands (delta, theta, alpha, beta, and gamma) of the power spectrum for each channel.
   - The EEG data is preprocessed using Fast Fourier Transform (FFT) to transform time-domain signals into the frequency domain.
   - Power Spectral Density (PSD) is computed to quantify the power present at each frequency component of the EEG signal.

2. **Normalization:**
   - Brain rate values are normalized using the baseline brain rate value computed during the relaxation phase of the session to correct any inherent biases or non-task-related activity.

### Neural Network Architectures
1. **Model 1: CNN+LSTM2D**
   - Predicts the next frames of brain topographical maps using a sequence of ConvLSTM2D layers followed by a Conv3D layer.
   - The model is trained on 500 seconds of images consisting of normal and binaural brain topographical maps.

2. **Model 2: Conditional GAN**
   - Focuses on next frame prediction at two time steps: t+0, t+15.
   - The input training data encapsulates the entire brain topographical maps (including all the frequency bands: alpha, beta, gamma, delta, theta).

3. **Model 3: Next Frame Prediction**
   - Similar architecture to Model 1, but focuses solely on next frame prediction at a future time step.

## Results

### Observations from Model 1: CNN+LSTM2D
- The model consistently captures the general structure and spatial distribution of activity across the human brain.
- Some noticeable differences indicate that certain areas have not been learned well, particularly in the frontal and occipital regions.
- Improved alignment in the frontal areas in the second second but discrepancies in the temporal regions.
- Misjudges the intensity and exact locations of peak activities, particularly in the central and parietal areas.

### Observations from Model 2: Conditional GAN
- Generated artifacts are visually distinguishable as irregular, noisy patches that do not conform to typical smooth gradients.
- The frame generated at T+15 shows an absence of the earlier noted artifacts, indicating a possible stabilization in the model’s predictions.
- Consistency and smoothness are improved, suggesting that the conditional aspect of the GAN helps guide the generation process.

### Observations from Model 3: Next Frame Prediction
- The model maintains the overall integrity of brain topographical maps but struggles with local precision, particularly in predicting changes in activity intensity.
- Reasonably effective in capturing the general spatial distribution of brain activity but fails to predict concentrated high activity areas accurately.

## Conclusion

The project demonstrates the feasibility of using deep learning models to predict brain activity patterns from EEG data, providing insights into the impact of binaural beats on attention span. The models show potential in capturing the general structure and dynamics of brain activity, but improvements are needed for better local precision and handling of artifacts.

## Future Work
- Implementation of more sophisticated neural network architectures.
- Adjustment of hyperparameters, learning rate, number of layers, and nodes in each layer to improve the model’s sensitivity to subtle changes.

## References
1. [Exploring Binaural Beats Parameters for Enhancing Sustained Attention](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9954819/)
2. [Inria BCI Challenge](https://www.kaggle.com/c/inria-bci-challenge)
3. [Modeling Cognitive Load as a Self-Supervised Brain Rate with Electroencephalography and Deep Learning](https://arxiv.org/abs/2209.10992)
4. [Emotional Stress Recognition Using Electroencephalogram Signals Based on a Three-Dimensional Convolutional Gated Self-Attention Deep Neural Network](https://www.mdpi.com/2076-3417/10/5/1666)

