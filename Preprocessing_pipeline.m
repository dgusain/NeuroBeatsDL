% % PPT: 1068, 1064,1067, 1047, 1043, 1037, 1024 - to run
% 
% run eeglab on console before running the script.
file_path = "D:/UB/SEM 02/CV & image processing/Ana/1006/Source bdf files/";
files = ["BBSAS-1006-CBLN-22.11.03-2_Artifact Rejection Info.Markers", "BBSAS-1006-CBLN-22.11.03-2.bdf", "BBSAS-1006-CBLN-22.11.03-2_OcularCorrection_EBR.Markers"];

markerFile = strcat(file_path, files(1));
markerFile = char(markerFile);
bdf_file = strcat(file_path, files(2));
bdf_file = char(bdf_file);
occmarkerFile = strcat(file_path, files(3));
occmarkerFile = char(occmarkerFile);
occ_correctedFile = 'D:/UB/SEM 02/CV & image processing/Ana/1006/';
occ_correctedSet = 'Occular_corrected_1006_2.set';
csv_file_output = 'D:/UB/SEM 02/CV & image processing/Ana/1006/preprocessed_1006_2.csv';

% Step 1: Read the Marker File
%markerFile = 'D:/UB/SEM 02/CV & image processing/Ana/1004/Source bdf files/BBSAS-1004-CGHW-22.10.25-1_Artifact Rejection Info.Markers';
fid = fopen(markerFile, 'r');
markers = textscan(fid, '%s %s %d %d %s', 'Delimiter', ',', 'HeaderLines', 1);
fclose(fid);

% Step 2: Load the BDF File into EEGLAB
EEG = pop_biosig(bdf_file);
%EEG = pop_biosig('D:/UB/SEM 02/CV & image processing/Ana/1004/Source bdf files/BBSAS-1004-CGHW-22.10.25-1.bdf');
EEG = eeg_checkset( EEG );

% Step 3: Re-referencing the channel data
% Define the indices of channels to be excluded during re-referencing
excludeChannels = {'VEO+', 'VEO-', 'HEOL', 'HEOR', 'M1', 'M2', 'B9', 'B10'}; % Add any other non-brain activity channels here

% Convert channel names to indices
excludeIndices = find(ismember({EEG.chanlocs.labels}, excludeChannels));

% Re-reference the EEG data using the mastoids (M1, M2) and exclude specified channels
mastoidIndices = find(ismember({EEG.chanlocs.labels}, {'M1', 'M2'})); % Find indices of M1 and M2
if length(mastoidIndices) == 2 % Check if both mastoid channels are present
    EEG = pop_reref( EEG, mastoidIndices, 'exclude', excludeIndices );
else
    error('Mastoid channels (M1, M2) not found or incomplete.');
end

% Step 4: Apply Marker Information for Artifact Rejection
% Assuming markers contains: Type, Description, Position, Length, Channel
types = markers{1};
positions = markers{3};
lengths = markers{4};

for i = 1:length(types)
    if strcmp(types{i}, 'Bad Interval')
        % Calculate the start and end points of the bad interval in EEG sample points
        % Sampling rate is given as 512Hz, which means each sample point is 1/512 second
        % Position given in the marker file is likely in milliseconds
        startSample = positions(i) * EEG.srate / 1000;
        endSample = startSample + lengths(i) * EEG.srate / 1000;
        
        % Here you could either create a new event type to mark bad intervals
        % or directly reject these segments. For simplicity, we'll mark them.
        EEG.event(end+1).type = 'Bad Interval';
        EEG.event(end).latency = startSample;
        EEG.event(end).duration = endSample - startSample;
    end
end

EEG = eeg_checkset( EEG );
% Save the artifact rejection dataset if necessary
%EEG = pop_saveset( EEG, 'filename','Rereferenced_Marked_1004_1.set','filepath','D:/UB/SEM 02/CV & image processing/Ana/1004/ICA dataset');

% Step 5: Applying Occular correction
%occmarkerFile = 'D:/UB/SEM 02/CV & image processing/Ana/1004/Source bdf files/BBSAS-1004-CGHW-22.10.25-1_OcularCorrection_EBR.Markers';
fid = fopen(occmarkerFile, 'r');
markers = textscan(fid, '%s %s %d %d %s', 'Delimiter', ',', 'HeaderLines', 1);
fclose(fid);
types = markers{1};
positions = markers{3};
lengths = markers{4};

for i = 1:length(types)
    if strcmp(types{i}, 'OcularCorrection')
        startSample = positions(i) * EEG.srate / 1000;
        endSample = startSample + lengths(i) * EEG.srate / 1000;
        EEG.event(end+1).type = 'OcularCorrection';
        EEG.event(end).latency = startSample;
        EEG.event(end).duration = endSample - startSample;
    end
end

EEG = eeg_checkset( EEG );
% Save the occular corrected dataset if necessary
EEG = pop_saveset( EEG, 'filename',occ_correctedSet,'filepath',occ_correctedFile);

% Step 6: Plotting the channel data
full_time_vector = (0:size(EEG.data, 2)-1) / EEG.srate;
% Define the time range for the last 1980 seconds
last_time_seconds = 1980;
start_time = max(full_time_vector) - last_time_seconds; % Start time in seconds
start_index = find(full_time_vector >= start_time, 1); % Find the corresponding index for the start time
% Adjust the time vector to include only the last 1980 seconds
time_vector = full_time_vector(start_index:end); % Time in seconds

figure; % Opens a new figure window
channel_names = {EEG.chanlocs.labels}; % Dynamically obtain channel names from EEG structure

scaling_factor = 100; % Scale factor for microvolts
threshold = -25 * scaling_factor; % Threshold set to -2500 microvolts
problematic_channels = [];
for ch = 1:EEG.nbchan
    channel_data_full = squeeze(EEG.data(ch,:,:)) / scaling_factor; % Adjusted for specific channel
    
    % Extract the portion of channel data for the last 1980 seconds
    channel_data = channel_data_full(start_index:end);
    
    if any(channel_data < threshold)
        problematic_channels = [problematic_channels, ch]; % Add channel index to problematic channels
        continue; % Skip plotting this channel
    end
    
    % Plot scaled channel data against the time vector
    plot(time_vector, channel_data);
    hold on; % Hold on to plot multiple lines in the same figure
end

% Finalize the plot
xlabel('Time (seconds)'); % Label for the x-axis
ylabel('Microvoltage (100 µV)'); % Label for the y-axis
title('Final EEG Data for the Last 1980 Seconds'); % Updated title for the plot
legend(channel_names); % Use actual channel names for the legend

hold off; % Release the plot

% Print the name and index of the problematic channel(s)
for i = 1:length(problematic_channels)
    fprintf('Problematic Channel: %s at index %d\n', channel_names{problematic_channels(i)}, problematic_channels(i));
end

% Step 7: Writing the preprocessed data to a .csv file
time_vector = (0:size(EEG.data, 2)-1) / EEG.srate; % Time in seconds
% Define the channel names for the legend
channel_names = {EEG.chanlocs.labels}; % Dynamically obtain channel names from EEG structure
channel_data_matrix = [];

for ch = 1:length(channel_names)
    channel_data = squeeze(EEG.data(ch,:,:));  
    %channel_data = squeeze(EEG.data(ch,1:50,:));
    channel_data_matrix = [channel_data_matrix; channel_data];  
end

channel_names{end+1} = 'Time';
channel_data_matrix = [channel_data_matrix; time_vector]';
%channel_data_matrix = [channel_data_matrix; time_vector(1:50)]';

% Write to CSV file
fileID = fopen(csv_file_output, 'w');
fprintf(fileID, '%s,', channel_names{1:end-1}); % Write headers
fprintf(fileID, '%s\n', channel_names{end}); % Write last header with newline
fclose(fileID);

dlmwrite(csv_file_output, channel_data_matrix, '-append', 'delimiter', ',');

