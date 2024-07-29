%Brain rate for 2 user: both normal and binuaral
file_paths = "D:/UB/SEM 02/CV & image processing/Ana/1007";
user_number = extractAfter(file_paths,"Ana/");
file_paths = char(file_paths);
psd_matrices = cell(1, 2);

for f = 1:2
    input_file = strcat("Occular_corrected_", user_number, "_", num2str(f), ".set"); % input file name
    input_file = char(input_file)
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; % booting up EEGLAB
    EEG = pop_loadset('filename', input_file, 'filepath',file_paths); % loading the datset
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
    
    channels_to_drop = {'VEO+', 'VEO-', 'HEOL', 'HEOR','Iz','Fcz'};
    EEG = pop_select(EEG, 'nochannel', channels_to_drop);
    channel_names = {EEG.chanlocs.labels}; % extracting channel names
    EEG = pop_resample(EEG, 128); % resampling data to 128Hz
    fft_results = zeros(size(EEG.data)); % initialization of fft_results

    for ch = 1:EEG.nbchan
        fft_results(ch, :) = fft(EEG.data(ch, :));   % converting EEG time-domain data to frequency-domain (FFT)
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
    baseline_end_index = find(time_vector <= (time_vector(end) -1980), 1, 'last');% all seconds before the last 1980

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
    psd_matrices{f} = psd_matrix;
    psd_matrix = [psd_matrix BR_per_epoch (1:num_epochs)'];
    psd_matrix_table = array2table(psd_matrix, 'VariableNames', [channel_names,"BrainRate","Time(s)"]);
    output_file = strcat(file_paths, "/Brain_rate_psd_", user_number, "_", num2str(f), ".csv");
    writetable(psd_matrix_table, output_file);
    
    if f == 1
        BR_per_epoch1 = BR_per_epoch;
        num_epochs1 = num_epochs;
    elseif f==2
        BR_per_epoch2 = BR_per_epoch;
        num_epochs2 = num_epochs;
    end   
end


% Plotting BR now
seconds_to_plot = 1980;
start_epoch1 = num_epochs1-seconds_to_plot;
start_epoch2 = num_epochs2-seconds_to_plot;
selected_epochs1 = start_epoch1:num_epochs1;
selected_epochs2 = start_epoch2:num_epochs2;

selected_BR_per_epoch1 = BR_per_epoch1(start_epoch1:num_epochs1);
selected_BR_per_epoch2 = BR_per_epoch2(start_epoch2:num_epochs2);
selected_mean_1 = mean(selected_BR_per_epoch1);
selected_mean_2 = mean(selected_BR_per_epoch2);

% baseline
mean_BR_1 = mean(BR_per_epoch1(1:start_epoch1));
mean_BR_2 = mean(BR_per_epoch2(1:start_epoch2));

% normalization by subtraction
selected_BR_per_epoch1 = selected_BR_per_epoch1-mean_BR_1;
selected_BR_per_epoch2 = selected_BR_per_epoch2-mean_BR_2;
% smoothing
smoothed_BR1 = smooth(selected_BR_per_epoch1, 15); 
smoothed_BR2 = smooth(selected_BR_per_epoch2, 15);

% Plotting
figure;
hold on; 
plot(selected_epochs1, smoothed_BR1, '-r', 'LineWidth', 2); % Increased line width
plot(selected_epochs2, smoothed_BR2, '-b', 'LineWidth', 2); % Increased line width
hold off; 
xlabel('Seconds', 'FontSize', 12); % Increased font size
ylabel('Brain Rate (BR)', 'FontSize', 12); % Increased font size
title_name = strcat("Brain Rate (BR) for task period for PPT ", user_number);
title_name = char(title_name);
title(title_name, 'FontSize', 14); % Increased font size for the title
grid on;
xlim([min(start_epoch1, start_epoch2), max(num_epochs1, num_epochs2)]);
legend('Normal', 'Binaural', 'Location', 'best', 'FontSize', 12); % Increased font size for legend

% Adjust axes properties for better visibility in presentations
ax = gca; % Get current axes
ax.FontSize = 12; % Set axes font size for numbers on axes
ax.LineWidth = 1.5; % Make axes lines thicker

