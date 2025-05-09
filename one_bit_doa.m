%% Matlab code to generate the signal model and estimate DOA with beamforming and MUSIC

%% Transmitted signal
clear

Nsig        = 2;                                          % Number of signals
doa_true    = sort(randsample(180,Nsig));                 % True angle of arrival of two sources
antenna_num = 16;                                         % Antenna numbers of the antenna array
time_ins    = 100;                                        % The number of time instants
s           = randn(Nsig,time_ins);  
x_clean     = signal_model(s, doa_true, antenna_num);     % Signal model of first source
SNR         = 5;                      
% The stage is yours.


% uncomment the following line and add AWGN
% x_noisy = your_own_code;

% quantization
% x_quantized = sign(real(x_noisy)) + 1i * sign(imag(x_noisy));


% MUSIC algorithm goes here


% Compressed sensing algorithm goes here


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


