%% MATLAB code to generate the signal model and estimate DOA with beamforming and MUSIC

%% Transmitted signal
clc; clear; close all;

Nsig        = 2;                                        % Number of signals
doa_true    = sort(randsample(180,Nsig));               % True angle of arrival of two sources
antenna_num = 16;                                       % Antenna numbers of the antenna array
time_ins    = 100;                                      % The number of time instants
s           = randn(Nsig,time_ins);  
x_clean     = signal_model(s, doa_true, antenna_num);   % Signal model of first source
SNR         = 5;

% adding AWGN
SNR_linear = 10^(SNR/10);
signal_power = mean(abs(x_clean(:)).^2);
noise_power = signal_power / SNR_linear;
noise = sqrt(noise_power/2) * (randn(size(x_clean)) + 1i * randn(size(x_clean)));
x_noisy = x_clean + noise;

% quantization
x_quantized = sign(real(x_noisy)) + 1i * sign(imag(x_noisy));

% grid for DoA estimation
% step = 1;
% theta_grid = -90:step:(90-step);
theta_grid = 0:180-1;
grid_size = length(theta_grid);

%% MUSIC algorithm with noisy data
% spatial correlation matrix
R_noisy = (x_noisy * x_noisy') / time_ins;

% eigendecomposition of correlation matrix
[V, D] = eig(R_noisy);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% noise subspace
En = V(:, Nsig+1:end);

% MUSIC spectrum calculation
P_MUSIC = zeros(1, grid_size);
for i = 1:grid_size
    a = exp(-1i * pi * (0:antenna_num-1)' * cosd(theta_grid(i)));
    P_MUSIC(i) = 1 / (a' * (En * En') * a);
end
P_MUSIC = abs(P_MUSIC) / max(abs(P_MUSIC));

% peaks in the MUSIC spectrum
[~, idx_peaks] = findpeaks(P_MUSIC, 'SortStr', 'descend', 'NPeaks', Nsig);
doa_est_MUSIC = sort(theta_grid(idx_peaks));


%% MUSIC algorithm with quantized data
% spatial correlation matrix
R_quantized = (x_quantized * x_quantized') / time_ins;

% eigendecomposition of correlation matrix
[V_q, D_q] = eig(R_quantized);
[~, idx_q] = sort(diag(D_q), 'descend');
V_q = V_q(:, idx_q);

% noise subspace
En_q = V_q(:, Nsig+1:end);

% MUSIC spectrum calculation
P_MUSIC_q = zeros(1, grid_size);
for i = 1:grid_size
    a_q = exp(-1i * pi * (0:antenna_num-1)' * cosd(theta_grid(i)));
    P_MUSIC_q(i) = 1 / (a_q' * (En_q * En_q') * a_q);
end
P_MUSIC_q = abs(P_MUSIC_q) / max(abs(P_MUSIC_q));

% peaks in the MUSIC spectrum
[~, idx_peaks_q] = findpeaks(P_MUSIC_q, 'SortStr', 'descend', 'NPeaks', Nsig);
doa_est_MUSIC_q = sort(theta_grid(idx_peaks_q));


%% Compressed Sensing based algorithm (similar to Complex Binary IHT)
% dictionary matrix (sensing matrix)
A = zeros(antenna_num, grid_size);
for i = 1:grid_size
    A(:, i) = exp(-1i * pi * (0:antenna_num-1)' * cosd(theta_grid(i)));
end

% IHT parameters
max_iter = 10;          % maximum number of iterations
K = Nsig;               % sparsity level (= number of signals)
mu = 1 / norm(A)^2;     % step size parameter
tol = 1e-6;             % convergence tolerance

% initialization
S_est = A' * x_quantized;

% implementation
for iter = 1:max_iter
    % equation 25 of paper - l1 case
    Y = sign(real(A*S_est)) + 1i * sign(imag(A*S_est)) - x_quantized;

    % gradient step
    S_temp = S_est - mu * A' * Y;

    % hard thresholding step (keeping K rows with largest l2 norm)
    row_norms = vecnorm(S_temp, 2, 2); % l2 norm of each row
    [~, idx] = findpeaks(row_norms, 'SortStr', 'descend', 'NPeaks', K);

    % set all but K largest rows to zero
    S_new = zeros(size(S_temp));
    S_new(idx, :) = S_temp(idx, :);

    % convergence check
    if norm(S_new - S_est, 'fro') < tol
        S_est = S_new;
        break;
    end

    S_est = S_new;
end

% normalize columns to unit L2 norm
S_est = S_est ./ vecnorm(S_est, 2, 1);

% DoA estimates (indices of nonzero rows)
row_norms = vecnorm(S_est, 2, 2);
[~, idx_cs] = findpeaks(row_norms, 'SortStr', 'descend', 'NPeaks', K);
doa_est_CS = sort(theta_grid(idx_cs(1:K)));


%% Performance comparison
disp('Performance Comparison:');
fprintf('True DoA angles: %s\n', mat2str(doa_true'));
fprintf('MUSIC estimation (noisy data): %s\n', mat2str(doa_est_MUSIC));
fprintf('MUSIC estimation (1-bit): %s\n', mat2str(doa_est_MUSIC_q));
fprintf('CBIHT-based estimation (1-bit): %s\n', mat2str(doa_est_CS));

% calculate, display errors
err_MUSIC = min(abs(doa_est_MUSIC - doa_true(1))) + min(abs(doa_est_MUSIC - doa_true(2)));
err_MUSIC_q = min(abs(doa_est_MUSIC_q - doa_true(1))) + min(abs(doa_est_MUSIC_q - doa_true(2)));
err_CS = min(abs(doa_est_CS - doa_true(1))) + min(abs(doa_est_CS - doa_true(2)));

fprintf('MUSIC error (noisy): %.2f degrees\n', err_MUSIC);
fprintf('MUSIC error (1-bit): %.2f degrees\n', err_MUSIC_q);
fprintf('CBIHT-based error (1-bit): %.2f degrees\n', err_CS);


%% Result plotting
figure;
subplot(3,1,1);
plot(theta_grid, 10*log10(P_MUSIC), 'LineWidth', 1.5);
hold on;
plot(doa_true, min(10*log10(P_MUSIC))*ones(size(doa_true)), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(doa_est_MUSIC, min(10*log10(P_MUSIC))*ones(size(doa_est_MUSIC)), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
title('MUSIC Spectrum (Noisy Data)');
xlabel('Angle (degrees)');
ylabel('Power (dB)');
legend('MUSIC Spectrum', 'True DoA', 'Estimated DoA');
grid on;

subplot(3,1,2);
plot(theta_grid, 10*log10(P_MUSIC_q), 'LineWidth', 1.5);
hold on;
plot(doa_true, min(10*log10(P_MUSIC_q))*ones(size(doa_true)), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(doa_est_MUSIC_q, min(10*log10(P_MUSIC_q))*ones(size(doa_est_MUSIC_q)), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
title('MUSIC Spectrum (1-bit Quantized Data)');
xlabel('Angle (degrees)');
ylabel('Power (dB)');
legend('MUSIC Spectrum', 'True DoA', 'Estimated DoA');
grid on;

subplot(3,1,3);
stem(theta_grid, abs(row_norms), 'LineWidth', 1.5);
hold on;
plot(doa_true, zeros(size(doa_true)), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(doa_est_CS, zeros(size(doa_est_CS)), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
title('CBIHT-Based Result (1-bit Quantized Data)');
xlabel('Angle (degrees)');
ylabel('Magnitude');
legend('CS Spectrum', 'True DoA', 'Estimated DoA');
grid on;


%% Monte Carlo simulation - different SNR values
SNR_range = -10:5:20;   % in dB
num_trials = 50;        % number of Monte Carlo trials

% initialize error metrics
err_MUSIC_snr = zeros(length(SNR_range), 1);
err_MUSIC_q_snr = zeros(length(SNR_range), 1);
err_CS_snr = zeros(length(SNR_range), 1);

% run simulation
for snr_idx = 1:length(SNR_range)
    current_snr = SNR_range(snr_idx);
    fprintf('Processing SNR = %d dB...\n', current_snr);

    err_MUSIC_trials = zeros(num_trials, 1);
    err_MUSIC_q_trials = zeros(num_trials, 1);
    err_CS_trials = zeros(num_trials, 1);

    for trial = 1:num_trials
        % generate random DoA angles
        doa_true_mc = sort(randsample(180,Nsig));

        % generate signal
        s_mc = randn(Nsig,time_ins);
        x_clean_mc = signal_model(s_mc, doa_true_mc, antenna_num);

        % add noise according to SNR
        SNR_linear_mc = 10^(current_snr/10);
        signal_power_mc = mean(abs(x_clean_mc(:)).^2);
        noise_power_mc = signal_power_mc / SNR_linear_mc;
        noise_mc = sqrt(noise_power_mc/2) * (randn(size(x_clean_mc)) + 1i*randn(size(x_clean_mc)));
        x_noisy_mc = x_clean_mc + noise_mc;

        % quantize
        x_quantized_mc = sign(real(x_noisy_mc)) + 1i * sign(imag(x_noisy_mc));

        % run MUSIC with noisy data
        R_noisy_mc = (x_noisy_mc * x_noisy_mc') / time_ins;
        [V_mc, D_mc] = eig(R_noisy_mc);
        [~, idx_mc] = sort(diag(D_mc), 'descend');
        V_mc = V_mc(:, idx_mc);
        En_mc = V_mc(:, Nsig+1:end);
        
        P_MUSIC_mc = zeros(1, length(theta_grid));
        for i = 1:length(theta_grid)
            a = exp(-1i * pi * (0:antenna_num-1)' * cosd(theta_grid(i)));
            P_MUSIC_mc(i) = 1 / (a' * (En_mc * En_mc') * a);
        end
        P_MUSIC_mc = abs(P_MUSIC_mc) / max(abs(P_MUSIC_mc));
        
        [~, idx_peaks_mc] = findpeaks(P_MUSIC_mc, 'SortStr', 'descend', 'NPeaks', Nsig);
        doa_est_MUSIC_mc = sort(theta_grid(idx_peaks_mc));

        % run MUSIC with quantized data
        R_quantized_mc = (x_quantized_mc * x_quantized_mc') / time_ins;
        [V_q_mc, D_q_mc] = eig(R_quantized_mc);
        [~, idx_q_mc] = sort(diag(D_q_mc), 'descend');
        V_q_mc = V_q_mc(:, idx_q_mc);
        En_q_mc = V_q_mc(:, Nsig+1:end);
        
        P_MUSIC_q_mc = zeros(1, length(theta_grid));
        for i = 1:length(theta_grid)
            a = exp(-1i * pi * (0:antenna_num-1)' * cosd(theta_grid(i)));
            P_MUSIC_q_mc(i) = 1 / (a' * (En_q_mc * En_q_mc') * a);
        end
        P_MUSIC_q_mc = abs(P_MUSIC_q_mc) / max(abs(P_MUSIC_q_mc));
        
        [~, idx_peaks_q_mc] = findpeaks(P_MUSIC_q_mc, 'SortStr', 'descend', 'NPeaks', Nsig);
        doa_est_MUSIC_q_mc = theta_grid(idx_peaks_q_mc);

        % run CBIHT-based implementation
        S_est_mc = A' * x_quantized_mc;
        for iter = 1:max_iter
            Y_mc = sign(real(A*S_est_mc)) + 1i * sign(imag(A*S_est_mc)) - x_quantized_mc;
            S_temp_mc = S_est_mc - mu * A' * Y_mc;
            row_norms_mc = vecnorm(S_temp_mc, 2, 2); % l2 norm of each row
            [~, idx] = findpeaks(row_norms_mc, 'SortStr', 'descend', 'NPeaks', K);
            S_new_mc = zeros(size(S_temp_mc));
            S_new_mc(idx, :) = S_temp_mc(idx, :);

            if norm(S_new_mc - S_est_mc, 'fro') < tol
                S_est_mc = S_new_mc;
                break;
            end
            S_est_mc = S_new_mc;
        end

        S_est_mc = S_est_mc ./ vecnorm(S_est_mc, 2, 1);
        row_norms_mc = vecnorm(S_est_mc, 2, 2);
        [~, idx_cs_mc] = findpeaks(row_norms_mc, 'SortStr', 'descend', 'NPeaks', K);
        doa_est_CS_mc = sort(theta_grid(idx_cs_mc(1:K)));

        % calculate errors
        err_MUSIC_trials(trial) = mean(min(abs(bsxfun(@minus, doa_est_MUSIC_mc', doa_true_mc))));
        err_MUSIC_q_trials(trial) = mean(min(abs(bsxfun(@minus, doa_est_MUSIC_q_mc', doa_true_mc))));
        err_CS_trials(trial) = mean(min(abs(bsxfun(@minus, doa_est_CS_mc', doa_true_mc))));
    end

    % average over trials
    err_MUSIC_snr(snr_idx) = mean(err_MUSIC_trials);
    err_MUSIC_q_snr(snr_idx) = mean(err_MUSIC_q_trials);
    err_CS_snr(snr_idx) = mean(err_CS_trials);
end

% plot SNR vs. error
figure;
semilogy(SNR_range, err_MUSIC_snr, 'bo-', 'LineWidth', 2);
hold on;
semilogy(SNR_range, err_MUSIC_q_snr, 'ro-', 'LineWidth', 2);
semilogy(SNR_range, err_CS_snr, 'go-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Average DoA Error (degrees)');
title('Performance Comparison: SNR vs. DoA Estimation Error');
legend('MUSIC (Noisy)', 'MUSIC (1-bit)', 'CBIHT-based (1-bit)');


%% Useful functions
% Signal model function
function x = signal_model(s, aoa_degree, antenna_num)
    aoa = aoa_degree * pi / 180; % to radian angle
    steering = zeros(antenna_num, size(s,1));
    for k = 1:antenna_num
        steering(k, :) = exp(-1i * pi * (k - 1) * cos(aoa));
    end
    x = steering*s;
end
