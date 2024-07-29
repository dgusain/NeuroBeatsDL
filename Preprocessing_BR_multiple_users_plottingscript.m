users = ["1004","1005","1006", "1066"];
s = size(users);
n = s(2);
session_time = 1980;
for j = 1:s(2)
        
        % Plotting calculation need to be added. 
        start_epoch = num_of_epochs-session_time;
        selected_BR_per_epoch1 = BR_per_user{j,1}(start_epoch(j,1):num_of_epochs(j,1));
        selected_BR_per_epoch2 = BR_per_user{j,2}(start_epoch(j,2):num_of_epochs(j,2));
        %selected_mean_1 = mean(selected_BR_per_epoch1);
        %selected_mean_2 = mean(selected_BR_per_epoch2);
        mean_BR_1 = mean(BR_per_user{j,1}(1:start_epoch(j,1))); % baseline mean
        mean_BR_2 = mean(BR_per_user{j,2}(1:start_epoch(j,2))); % baseline mean

        selected_BR_per_epoch1 = selected_BR_per_epoch1/mean_BR_1; % normalization by subtraction
        selected_BR_per_epoch2 = selected_BR_per_epoch2/mean_BR_2; % normalization by subtraction

        smoothed_BR1 = smooth(selected_BR_per_epoch1, 15); % smoothing by average window
        smoothed_BR2 = smooth(selected_BR_per_epoch2, 15); % smoothing by average window

        session_epochs1 = start_epoch(j,1):num_of_epochs(j,1);
        session_epochs2= start_epoch(j,2):num_of_epochs(j,2);
        start_x = min(start_epoch(j,1), start_epoch(j,2));
        end_x = max(num_of_epochs(j,1), num_of_epochs(j,2));
        
        figure; 
        hold on; 
        plot(session_epochs1, smoothed_BR1, '-r', 'LineWidth', 1);
        plot(session_epochs2, smoothed_BR2, '-b', 'LineWidth', 1);
        legend('Normal', 'Binaural','Location', 'best' );
        xlabel('Seconds');
        ylabel('Brain Rate (BR)');
        xlim([start_x, end_x]);
        title_name = strcat("Brain Rate (BR) for task period for PPT ", users(j));
        title_name = char(title_name);
        title(title_name);
        grid on;  
        hold off;
 end
 