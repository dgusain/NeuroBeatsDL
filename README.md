# NeuroBeatsDL
A deep learning exploration of EEG-processed brain topographical maps, uncovering the impact of binaural beats on neural activity.
## Abstract

This project integrates deep learning with image processing and computer vision techniques to quantify the impact of binaural beats on attention span by harnessing EEG data collected across 42 electrodes on the human scalp. The methodology extends beyond traditional signal analysis by preprocessing the EEG signals into brain heat maps—topographical representations of brain activity—leveraged as input into neural network architectures like conditional Generative Adversarial Networks (GANs), 3D Convolutional Neural Networks, autoencoder transformer models, and diffusion models. The networks are designed to anticipate how the brain heat map would evolve in the ensuing moment, hence providing a dynamic measure of attention span, indicated by the color-coded contour lines of the active brain regions. By comparing these predictions in the presence and absence of binaural audio, the project performs a nuanced exploration of how auditory stimuli modulate human neural activity, highlighting the potential of combining deep learning with image processing to understand attention mechanisms.

## Dataset

The dataset involves raw EEG signal files recorded in BIOSEMI Active Two Brain recorder and exported in (.BDF) format. The data consists of deidentified data for 80 participants, recorded for two sessions of 30 minutes each, where one session had just pure tone, while the other session had binaural beats. The EEG signals have a sampling frequency of 512 Hz. Each session consisted of 33 minutes, where the participant had to answer 1200 cognitive ability questions, and their brain activity was recorded via EEG, and their wrong answers were counted. These 1200 questions had a stimulus window of 1500 ms and a response window of 150 ms, hence the total duration of the question was 1.65 seconds.
<table>
  <tr>
    <td><img src="EEG_3D_Rotation.gif" alt="Sample Brain map" width="400"/></td>
    <td><img src="cap_32_rotate_large.gif" alt="EEG collection" width="400"/></td>
  </tr>
</table>

## Vital information about the signals

These functions of these electrodes are explained in the below table, along with the positioning of these electrodes on the human brain following the international standard 10-20 system.

| Electrode | Description         | Targeted Brain Area                                                                           |
|-----------|---------------------|-----------------------------------------------------------------------------------------------|
| Fp1, Fp2  | Frontopolar         | Frontal lobe, near the forehead, dealing with executive functions and voluntary movement      |
| AF3, AF4  | Anterior Frontal    | Frontal lobe, responsible for high level cognitive functions                                  |
| F7, F8    | Frontal             | Lateral frontal areas dealing with motor function and language                                |
| F3, F4    | Frontal             | Central frontal areas dealing with decision making and problem solving                        |
| FC1, FC2, FC5, FC6 | Frontocentral | Bridge between frontal and central areas, associated with motor control                    |
| T7, T8    | Temporal            | Middle temporal areas responsible for memory and hearing                                      |
| C3, C4    | Central             | Central brain associated with sensorimotor coordination                                       |
| CP1, CP2, CP5, CP6 | Centroparietal | Linking central and parietal lobes, vital for sensory integration                         |
| P7, P8    | Parietal            | Lateral parietal areas involved with orientation and perception                               |
| P3, P4    | Parietal            | Central parietal areas crucial for sensory input and spatial orientation                      |
| Pz        | Parietal midline    | Top of the head responsible for somatosensory processing                                      |
| PO3, PO4  | Parietal Occipital  | Near the visual cortex, important for visual processing                                       |
| O1, O2    | Occipital           | Back of the head, primary visual processing area                                              |
| Oz        | Occipital midline   | Central occipital, key in visual data processing                                              |
| VEO+, VEO-| Vertical EOG        | Vertical eye movement                                                                         |
| HEOL, HEOR| Horizontal EOG      | Horizontal eye movement                                                                       |
| FCz, Cz   | Central midline     | Central area of the head involved for motor function                                          |
| Iz        | Inion               | Near the back of the skull, used for referencing the EEG                                      |
| M1, M2    | Mastoid reference   | Located near the mastoids behind the ears (left and right earlobes), used for referencing the EEG signals |
| B9, B10   | Additional          | Supplementary electrodes for specific studies                                                 |

<table>
  <tr>
    <td><img src="Project pictures/10-20 system.png" alt="EEG System" width="400"/></td>
  </tr>
</table>

## Methodology
### Signal preprocessing - (Preprocessing pipeline.m)
**Software:** MATLAB (EEGLAB toolbox)

1. The raw EEG dataset for a single user, for a single session is imported into the EEGLAB on MATLAB.
2. Two marker files: Artifact rejection and ocular correction are also imported for that user. These files are obtained after visualizing the signals on an external software, called Brain Vision Analyzer, which is capable of recognizing the noise in the data due to muscle movements, eye blinks, nose twitches etc.
3. Before applying the marker information for artifact rejection, we visualize the raw EEG data to make sure that the signals are static and ready for preprocessing.
4. However, on visualizing the EEG signals, on a single plot, a trend is observed in the data. This means that the brain signals aren’t as static as required for the analysis. Hence, we decide to re-reference the EEG signals, based on the mastoids (M1 and M2) signals, which are referred to as ground signals, and correspond to the left and right earlobe respectively.
5. During re-referencing, it is important to exclude non-brain activity signals from the analysis. Hence, certain EEG channels like VEO+, VEO-, HEOR, HEOL, B9 and B10 are excluded from the data, during re-referencing, but not dropped due to their utility during ICA decomposition.
6. The re-referenced signal shows a much better static trend and then gets pre-processed by applying the marker information of artifact rejection and ocular correction.
7. This final signal is then decomposed using Independent Component Analysis, in order to clean out any extra noise.
8. The data contained in the final clean signal is written into a .csv file, as channel data matrix and time vector.

<table>
  <tr>
    <td><img src="Project pictures/Raw EEG signal.png" alt="Raw EEG signal" width="500"/></td>
    <td><img src="Project pictures/Preprocessed EEG signal.png" alt="Preprocessed EEG signal" width="500"/></td>
  </tr>
</table>

#### Observations:

1. **Baseline correction:** The raw EEG signal has baseline drifts moving upwards with time. These have been corrected in the pre-processed signal by re-referencing the channel signals.
2. **Artifact removal:** Noise due to artifacts such as eye blink (VEOG), cardiac signals, and muscle contractions have been filtered out in the second image as evident by the absence of sharp spikes, resulting in a much more stable pattern of brain activity.
3. **Normalized amplitude fluctuations:** Amplitudes are normalized to a certain range to avoid variations due to electrode impedance, in the second signal.

The signal-to-noise ratio has been increased to approximately 17%. This difference in the two signals is further amplified on illustrating the topographical map of the brain for raw and pre-processed data.
<table>
  <tr>
    <td><img src="Project pictures/2D brain map.png" alt="2D brain map" width="700"/></td>
  </tr>
  <tr>
    <td><img src="Project pictures/3D brain map.png" alt="3D brain map" width="700"/></td>
  </tr>
</table>

### Data Preprocessing
#### Problem with existing labels
The dataset’s cognitive ability questions proved to be an imbalanced label for the EEG processed brain maps. This was because each participant would get on average, only 10 to 20 questions wrong out of 1200 and these wrong answers are scattered throughout the dataset. As a result, the CNN and CNN+LSTM model was unable to learn from the dataset based on the number of correct answered questions as the label. The correct answers were labelled as 1 and wrong answers were labelled as 0. Due to the sparsity of wrong answers, the model is unable to generalize its learning.

#### Defining a novel metric (brain rate) as the label

There was a need to define a single scalar quantity that would describe the EEG data (across 42 channels) for each timestep of the session. Taking inspiration from a novel metric defined in a research paper, we developed a brain rate variable that would be used to describe the data.

<table>
  <tr>
    <td><img src="Project pictures/brain rate calculation flowchart.png" alt="Brain Rate Calculation flowchart" width="700"/></td>
  </tr>
</table>

The above figure illustrates the steps required to calculate the brain rate for an EEG signal of a user. Before we jump into the methodology of how the brain rate variable is calculated, it is necessary to understand the different frequency bands present in EEG signals, and how they influence the cognitive processes. This literature was essential in determining the weights given to each frequency band while calculating the scalar value of the brain rate.

#### Cognitive EEG frequency bands

The electroencephalogram (EEG) signals are divided into different frequency bands, each associated with different types of brain activity, including cognitive processes. The most important frequency bands for finding cognitive brain activity are:

- **Delta Waves (0.5 to 4 Hz):**
  Although primarily associated with deep sleep, delta waves can also be related to certain cognitive processes in some pathological states.
- **Theta Waves (4 to 8 Hz):**
  Theta waves are often linked to memory, learning, and navigation. They are particularly prominent during meditative, drowsy, or sleeping states but also appear during deep emotional experiences and cognitive activities that require concentration and focus.
- **Alpha Waves (8 to 12 Hz):**
  Alpha waves are associated with a state of relaxed alertness and rest. They become prominent when a person is calm and physically and mentally relaxed but still alert. Alpha activity is often related to the brain's idle state or resting state and has been linked to creativity and the reduction of depression.
- **Beta Waves (12 to 30 Hz):**
  Beta waves are associated with active, analytical thought and alertness. This frequency band is most commonly observed during active conversation, problem-solving, decision-making, and focused mental activity. High levels of beta activity are associated with stress, anxiety, or excitement.
- **Gamma Waves (30 Hz and above):**
  Gamma waves are associated with higher mental activity, including perception, problem-solving, fear, and consciousness. High-frequency gamma activity is linked to the processing of information from different brain areas simultaneously, essentially integrating thoughts and experiences.

Each of these frequency bands plays a role in different aspects of cognitive processing and brain activity. For cognitive tasks and states of consciousness, alpha, beta, and gamma waves are particularly significant, as they are directly related to how we think, learn, and process information. However, the importance of a specific band can vary depending on the type of cognitive activity being performed.

#### Fast Fourier transforms & Power spectral density

Fast Fourier transforms is a vital mathematical tool used in analysing EEG data that transforms time-domain signals into frequency domain. This transformation is crucial because it:

- Enables spectral analysis of human brain’s electrical activity in terms of frequency components.
- Efficiently computes the Discrete Fourier Transform (DFT) of a sequence which contributes significantly to handling large volumes of EEG data for quick effective analysis.
- Provides insight into patterns that aren’t visible in the time domain, for example the brain states associated with alertness are associated with certain frequency bands (alpha waves for relaxation, beta waves for alertness).

Power Spectral Density (PSD) is a function that describes the power distribution of a signal across various frequencies. It is computed from FFT results to quantify the power present at each frequency component of the EEG signal. PSD is necessary as it:

- Provides a quantitative measure of signal power across frequencies which is essential to assess brain activity linked to different neurological conditions.
- Offers a more condensed, low dimensional representation of the EEG data hence reducing data complexity while retaining critical information about the signal’s power distribution making it more manageable and interpretable.

### Brain rate calculation
#### Formula

Brain rate is defined as a sum of the mean frequencies of brain oscillations weighted over the EEG bands (delta, theta, alpha, beta, and gamma) of the power spectrum for each channel.

$$ BR = \sum_{ch=1}^n \sum_{b=1}^5 (f_b \cdot P(b,ch)) $$

- \( b \) is the index of the frequency bands (1 for delta, 2 for theta, 3 for alpha, 4 for beta, 5 for gamma).
- \( f_b \) is the weight associated with each frequency band b. This is the mean frequency of that band (given in Hz).
- \( P(b,ch) \) is the mean amplitude of the electrical potential for band b of each channel ch over the mean of all its amplitudes.

In simple terms, \( P(b,ch) \) is the power ratio of a specific EEG frequency band over a channel.

$$ P(b,ch) = \frac{avg_b(FFT_{ch})}{avg(FFT_{ch})} $$

- \( P(b,ch) \) is the power ratio for frequency band b and channel ch.
- \( FFT_{ch} \) is the vector containing the amplitudes of the FFT transformed channel.
- \( avg_b(FFT_{ch}) \) is the average (centroid) of the amplitudes within the frequency band b from the FFT-transformed channel ch. It is the centroid of the band’s amplitudes within that frequency range.
- \( avg(FFT_{ch}) \) is the average of all amplitudes across the entire spectrum from the FFT-transformed channel ch.

\( P(b,ch) \) shows the contribution of the band b to the channel’s overall signal. The higher the ratio, the more dominant that particular frequency band is in the channel’s signal.


#### Methodology

1. Pre-processed EEG datasets (after applying artifact rejection and ocular correction) are loaded into MATLAB’s EEGLAB tool.
2. The EEG data is resampled from 512 Hz to 128 Hz, to reduce the number of samples (brain maps) generated per second, while retaining all crucial information. If we sample to 64 Hz, the Nyquist frequency would be 32 Hz (half of sampling frequency), leading to loss of gamma frequency band data.
3. The Fast Fourier transform is calculated for each channel of the EEG data.
4. The five distinct frequency bands are isolated using bandpass filtering. This separation of the data is essential for analysing specific ranges of brain activity related to different states.
5. Power spectral density of EEG data is calculated for each one-second epoch within the filtered frequency bands, hence quantifying the power present at different frequencies.
6. Centroids for each frequency band are computed which serve as baseline weights, helpful in capturing the dominant frequency within the band during each epoch.
7. Brain rate is calculated for each epoch by weighting the centroids of the frequency bands by their relative power contributions. The formula used effectively integrates information across the frequency spectrum to provide a singular measure of brain activity.
8. Normalization of brain rates by subtraction of baseline brain rate: The cognitive session brain rate values are normalized using the baseline brain rate value computed during the relaxation phase of the session. This is done to correct any inherent biases or non-task related activity.

The sparse nature of incorrect answers in the dataset led to the calculation and employment of brain rate as a continuous metric rather than binary labels, allowing for a more nuanced analysis of the EEG data. This methodology provides a robust framework for understanding and quantifying EEG data in research or clinical settings where standard metrics or labels prove to be inadequate for capturing the complexities of brain activity.

#### Observations: 
We can see the correlation between the brain rate metric created and the brain topographical map over a session. The red lines denote the value of the brain rate metric, while the coloured contour lines depict the activity in the participant's brain at that time. 
<table>
  <tr>
    <td><img src="1003_BR_comparison.png" alt="PPT 1003 brain rate across both sessions" width="800"/></td>
  </tr>
  <tr>
    <td><img src="1066_BR_comparison.png" alt="PPT 1003 brain rate across both sessions" width="800"/></td>
  </tr>
</table>

The below figure shows a comparison between the brain rates obtained over two sessions, for two different participants: 1003 and 1066. 
- Blue line: Brain rate metric during the session with Binaural beats as auditory stimuli 
- Red line: Brain rate metric during the session without any auditory stimuli
<table>
  <tr>
    <td><img src="Project pictures/Correlation map BR.png" alt="Brain rate graph" width="700"/></td>
  </tr>
  <tr>
    <td><img src="Project pictures/Correlation map BM.jpg" alt="Brain map" width="700"/></td>
  </tr>
</table>

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

