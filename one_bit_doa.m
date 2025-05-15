%% MATLAB code to generate the signal model and estimate DOA with beamforming and MUSIC

%% Transmitted signal
clear

Nsig        = 2;                                          % Number of signals
doa_true    = sort(randsample(0:180-1,Nsig));            % True angle of arrival of two sources
antenna_num = 16;                                         % Antenna numbers of the antenna array
time_ins    = 100;                                        % The number of time instants
s           = randn(Nsig,time_ins);  
x_clean     = signal_model(s, doa_true, antenna_num);     % Signal model of first source
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


%% Compressed Sensing based algorithm (Complex Binary IHT)
% dictionary matrix (sensing matrix)
A = zeros(antenna_num, grid_size);
for i = 1:grid_size
    A(:, i) = exp(-1i * pi * (0:antenna_num-1)' * cosd(theta_grid(i)));
end

% IHT parameters
max_iter = 10;          % maximum number of iterations
K = Nsig;               % sparsity level (= number of signals)
mu = 1 / norm(A)^2;    % step size parameter
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


% Performance comparison




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
