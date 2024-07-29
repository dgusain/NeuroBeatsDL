% Load the EEG dataset
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
EEG = pop_loadset('filename', 'Occular_corrected_1003_1.set', 'filepath', 'D:/UB/SEM 02/CV & image processing/Ana/1003');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
EEG = pop_chanedit(EEG, 'lookup','plugins/dipfit/standard_BESA/standard-10-5-cap385.elp');
EEG = pop_resample(EEG, 128); % Resampling down to 128Hz

% Automatically determine the sampling frequency from the EEG data structure
sampling_frequency = EEG.srate;


fft_results = zeros(size(EEG.data));

    for ch = 1:EEG.nbchan
        fft_results(ch, :) = fft(EEG.data(ch, :));  
    end

% Assuming you have fft_results computed as shown previously
N = size(fft_results, 2); % Number of points in FFT
sample_rate = 128; % Sampling rate
frequencies = (0:N-1) * (sample_rate / N); % Frequency vector

% Beta band frequencies
deltaBand = [0.5, 4];
thetaBand = [4, 8];
alphaBand = [8, 12];
betaBand = [12, 30];

% Creating a frequency mask for the beta band
%beta_mask = (frequencies >= betaBand(1) & frequencies <= betaBand(2)) | ...
 %           (frequencies >= (sample_rate - betaBand(2)) & frequencies <= (sample_rate - betaBand(1)));

% Creating frequency masks for each band
delta_mask = (frequencies >= deltaBand(1) & frequencies <= deltaBand(2));
theta_mask = (frequencies >= thetaBand(1) & frequencies <= thetaBand(2));
alpha_mask = (frequencies >= alphaBand(1) & frequencies <= alphaBand(2));
beta_mask = (frequencies >= betaBand(1) & frequencies <= betaBand(2));

% Combine all masks
combined_mask = delta_mask | theta_mask | alpha_mask | beta_mask;
% Apply mask to FFT results to keep only beta frequencies
beta_fft_results = zeros(size(fft_results));
for ch = 1:size(fft_results, 1)
    channel_fft = fft_results(ch, :);
    channel_fft(~combined_mask) = 0; % Zero out all frequencies not in the beta band
    beta_fft_results(ch, :) = channel_fft;
end

% Inverse FFT to convert back to time domain
beta_time_domain = real(ifft(beta_fft_results, [], 2));


% Calculate end_second as the last second in the dataset
total_samples = size(EEG.data, 2);
end_second = floor(total_samples / sampling_frequency);
seconds = 500; % Adjust the total duration as needed
start_second = end_second - seconds + 1; 


%start_second = 2526;
%end_second = 2691;
% Define the output folder for saving the JPEG files
output_folder = 'C:/Users/hp/1003/combined_mask_alpha_beta_delta_theta'; % Update the path as needed
if ~exist(output_folder, 'dir')
    mkdir(output_folder); % Create the folder if it does not exist
end

% Initialize the progress bar
h = waitbar(0, 'Initializing...');

total_seconds = end_second - start_second + 1; % Total number of seconds to process

% Iterate through the specified seconds
for second = start_second:end_second
    % Adjusting block size to 16 for calculating mean of each 16-sample block
    num_blocks = sampling_frequency / 16; % Number of blocks per second
    block_size = 16; % Number of samples per block
    
    for block_index = 1:num_blocks
        start_sample_index = (second - 1) * sampling_frequency + (block_index - 1) * block_size + 1;
        end_sample_index = min((start_sample_index + block_size - 1), total_samples);
        
        % Check if the end_sample_index exceeds total_samples
        if start_sample_index <= total_samples
            % Calculate the mean for each block of 16 samples
            data_at_block = mean(beta_time_domain(:, start_sample_index:end_sample_index), 2);
            
            % Plotting the heat map using the topoplot function
            fig = figure('Color', [1 1 1], 'Renderer', 'Painters', 'Position', [100 100 512 512], 'Visible', 'off');
            topoplot(data_at_block, EEG.chanlocs, 'electrodes', 'on', 'style', 'both');
            
            % Save the figure with the modified file naming convention
            saveas(fig, fullfile(output_folder, sprintf('Sample_%d_%d_128Hz_1.jpg', second, block_index)), 'jpg');
            close(fig); % Close the figure to free up memory
        else
            disp(sprintf('Sample index %d out of range, stopping.', end_sample_index));
            break; % Exit the loop if the sample index is out of range
        end
    end
    % Update the progress bar after completing each second
    waitbar((second - start_second + 1) / total_seconds, h, sprintf('Processing: %d %% complete', floor((second - start_second + 1) / total_seconds * 100)));
    
end

% Close the progress bar
close(h);
