%Brain rate for more than 1 user: both normal and binuaral
% for any more users, we need to preprocess them using the Preprocessing
% pipeline. 
users = ["1004","1005","1006", "1066"];
s = size(users);
n = s(2);
session_time = 1980;
num_of_epochs = zeros(n,n);
BR_per_user = cell(n,n);

figure;
hold on;

for j = 1:s(2) % users loop
    file_paths = strcat("D:/UB/SEM 02/CV & image processing/Ana/", users(j));
    file_paths = char(file_paths);
    for f = 1:2
        input_file = strcat("Occular_corrected_", users(j), "_", num2str(f), ".set");
        input_file = char(input_file);
        
        % booting up EEGLAB
        [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
        EEG = pop_loadset('filename', input_file, 'filepath',file_paths);
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
        channel_names = {EEG.chanlocs.labels};
        EEG = pop_resample(EEG, 128); % resampling to 128Hz
        fft_results = zeros(size(EEG.data)); % FFT
        for ch = 1:EEG.nbchan
            fft_results(ch, :) = fft(EEG.data(ch, :));  
        end

        % defining the EEG frequency bands
        deltaBand = [0.5, 4];
        thetaBand = [4, 8];
        alphaBand = [8, 12];
        betaBand = [12, 30];
        gammaBand = [30, 45];

        % applying bandpass filters
        EEG_delta = pop_eegfiltnew(EEG, 'locutoff', deltaBand(1), 'hicutoff',deltaBand(2),'plotfreqz', 0);
        EEG_theta = pop_eegfiltnew(EEG, 'locutoff', thetaBand(1), 'hicutoff',thetaBand(2),'plotfreqz', 0);
        EEG_alpha = pop_eegfiltnew(EEG, 'locutoff', alphaBand(1), 'hicutoff',alphaBand(2),'plotfreqz', 0);
        EEG_beta = pop_eegfiltnew(EEG, 'locutoff', betaBand(1), 'hicutoff',betaBand(2),'plotfreqz', 0);
        EEG_gamma = pop_eegfiltnew(EEG, 'locutoff', gammaBand(1), 'hicutoff',gammaBand(2),'plotfreqz', 0);

        time_vector = EEG.times / 1000; % time in seconds 

        % Baseline timeframe (0 to (end-1980 seconds)
        baseline_start_index = find(time_vector >= 0, 1, 'first');
        baseline_end_index = find(time_vector <= (time_vector(end) -session_time), 1, 'last');% all seconds before the last 1980

        % Getting the centroids of our bands
        N = length(fft_results); % Length of the FFT
        frequencies = (0:N-1) * (EEG.srate / N); % Frequency vector

        % Initialize matrix to hold BR for each 1-second epoch
        num_epochs = floor(length(EEG.data) / EEG.srate); % Number of whole 1-second epochs in the data
        BR_per_epoch = zeros(num_epochs,1);
        psd_matrix = [];

        for epoch = 1:num_epochs
            epoch_data = EEG.data(:, (epoch-1)*EEG.srate+1:epoch*EEG.srate);
            epoch_fft = fft(epoch_data, [], 2); 
            epoch_psd = abs(epoch_fft).^2 / EEG.srate;
            avg_psd = mean(epoch_psd, 1);
            frequencies = (0:length(avg_psd)-1) * (EEG.srate / length(avg_psd));

            % Computing the centroids
            delta_indices = (frequencies >= deltaBand(1) & frequencies <= deltaBand(2));
            centroid_delta = sum(frequencies(delta_indices) .* avg_psd(delta_indices)) / sum(avg_psd(delta_indices));
            theta_indices = (frequencies >= thetaBand(1) & frequencies <= thetaBand(2));
            centroid_theta = sum(frequencies(theta_indices) .* avg_psd(theta_indices)) / sum(avg_psd(theta_indices));
            alpha_indices = (frequencies >= alphaBand(1) & frequencies <= alphaBand(2));
            centroid_alpha = sum(frequencies(alpha_indices) .* avg_psd(alpha_indices)) / sum(avg_psd(alpha_indices));
            beta_indices = (frequencies >= betaBand(1) & frequencies <= betaBand(2));
            centroid_beta = sum(frequencies(beta_indices) .* avg_psd(beta_indices)) / sum(avg_psd(beta_indices));
            gamma_indices = (frequencies >= gammaBand(1) & frequencies <= gammaBand(2));
            centroid_gamma = sum(frequencies(gamma_indices) .* avg_psd(gamma_indices)) / sum(avg_psd(gamma_indices));

            BR_epoch = 0;
            for b = 1:5  
                if b == 1
                    band = deltaBand;
                    f_b = centroid_delta;
                elseif b == 2
                    band = thetaBand;
                    f_b = centroid_theta;
                elseif b == 3
                    band = alphaBand;
                    f_b = centroid_alpha;
                elseif b == 4
                    band = betaBand;
                    f_b = centroid_beta;
                else
                    band = gammaBand;
                    f_b = centroid_gamma;
                end
                band_indices = find(frequencies >= band(1) & frequencies <= band(2));
                P_b_ch = sum(avg_psd(band_indices)) / length(band_indices);
                total_mean_amplitude = sum(avg_psd) / length(frequencies);
                P_b_ch = P_b_ch / total_mean_amplitude;
                BR_epoch = BR_epoch + f_b * P_b_ch;        
            end
            avg_psd_per_channel = mean(epoch_psd, 2);
            psd_matrix = [psd_matrix; avg_psd_per_channel'];
            BR_per_epoch(epoch) = BR_epoch;
        end

        psd_matrix = [psd_matrix BR_per_epoch (1:num_epochs)'];
        psd_matrix_table = array2table(psd_matrix, 'VariableNames', [channel_names,"BrainRate","Time(s)"]);
        output_file = strcat(file_paths, "/Brain_rate_psd_", users(j), "_", num2str(f), ".csv")
        writetable(psd_matrix_table, output_file);
        
        BR_per_user{j,f} = BR_per_epoch;
        num_of_epochs(j,f) = num_epochs;
        
    end % ending session loop
      
end % ending user loop
