function varargout = helper_functions(func_name, varargin)
    % HELPER_FUNCTIONS - Dispatcher function to call specific signal processing functions
    % Inputs:
    %   func_name - String specifying the function to execute
    %   varargin - Variable input arguments passed to the specified function
    % Outputs:
    %   varargout - Variable output arguments returned by the called function
    switch lower(func_name)
        case 'nuevoanalizadorspec'
            varargout{1} = nuevoanalizadorSpec(varargin{:}); % Call spectrum analyzer function
        case 'plot_time_phase'
            plot_time_phase(varargin{:}); % Call time and phase plotting function
        case 'constelaciones'
            constelaciones(varargin{:}); % Call constellation plotting function
        case 'plot_psd'
            plot_psd(varargin{:}); % Call PSD plotting function
        case 'apply_rician_channel'
            varargout{1} = apply_rician_channel(varargin{:}); % Call Rician channel application function
        case 'calculate_snr'
            [varargout{1}, varargout{2}] = calculate_snr(varargin{:}); % Call SNR calculation function
        case 'apply_snr'
            [varargout{1}, varargout{2}] = apply_snr(varargin{:}); % Call SNR application function
        case 'apply_rayleigh_channel'
            [varargout{1}, varargout{2}] = apply_rayleigh_channel(varargin{:}); % Call Rayleigh channel application function
        otherwise
            error('Funcion desconocida: %s', func_name); % Throw error for unknown function
    end
end

function sa = nuevoanalizadorSpec(fs, signal1, signal2, title)
    % NUEVOANALIZADORSPEC - Configures and runs a spectrum analyzer for two signals
    % Inputs:
    %   fs - Sampling frequency (Hz)
    %   signal1 - First input signal (e.g., original signal)
    %   signal2 - Second input signal (e.g., processed signal)
    %   title - Cell array with channel names for the plot
    % Outputs:
    %   sa - Spectrum analyzer object
    sa = spectrumAnalyzer(...
        'SampleRate', fs, ... % Set sampling frequency
        'SpectrumType', 'power-density', ... % Use power spectral density
        'SpectrumUnits', 'dBm', ... % Display in dBm
        'Method', 'welch', ... % Use Welch's method for PSD estimation
        'RBWSource', 'property', ... % Set resolution bandwidth manually
        'RBW', 50, ... % Resolution bandwidth in Hz
        'ChannelNames', title, ... % Set channel names for legend
        'ShowLegend', true); % Enable legend display
    reset(sa); % Reset spectrum analyzer state
    sa(signal1, signal2); % Analyze both signals
    release(sa); % Release system resources
end

function plot_time_phase(input, output, title_str, indices, fs, unwrap_angles)
    % PLOT_TIME_PHASE - Plots time-domain and phase of input and output signals
    % Inputs:
    %   input - Original input signal
    %   output - Processed output signal
    %   title_str - Title string for the processed signal plot
    %   indices - Sample indices to plot
    %   fs - Sampling frequency (Hz)
    %   unwrap_angles - Boolean to unwrap phase angles (true/false)
    t = indices / fs; % Convert sample indices to time (seconds)
    figure;
    subplot(2,2,1); % First subplot: Original signal I/Q components
    plot(t*1e3, real(input(indices)), 'b', 'DisplayName', 'I'); hold on;
    plot(t*1e3, imag(input(indices)), 'r', 'DisplayName', 'Q');
    title('Señal original'); xlabel('Tiempo (ms)'); ylabel('Amplitud');
    legend; grid on;
    
    subplot(2,2,3); % Second subplot: Processed signal I/Q components
    plot(t*1e3, real(output(indices)), 'b', 'DisplayName', 'I'); hold on;
    plot(t*1e3, imag(output(indices)), 'r', 'DisplayName', 'Q');
    title(['Señal con ', title_str]); xlabel('Tiempo (ms)'); ylabel('Amplitud');
    legend; grid on;
    
    subplot(2,2,2); % Third subplot: Original signal phase
    if unwrap_angles
        plot(t*1e3, unwrap(angle(input(indices))), 'DisplayName', 'Fase');
    else
        plot(t*1e3, 1*(angle(input(indices))), 'DisplayName', 'Fase');
    end
    title('Fase original'); xlabel('Tiempo (ms)'); ylabel('Fase (rad)');
    grid on;
    
    subplot(2,2,4); % Fourth subplot: Processed signal phase
    if unwrap_angles
        plot(t*1e3, unwrap(angle(output(indices))), 'DisplayName', 'Fase');
    else
        plot(t*1e3, 1*(angle(output(indices))), 'DisplayName', 'Fase');
    end
    title(['Fase con ', title_str]); xlabel('Tiempo (ms)'); ylabel('Fase (rad)');
    grid on;
end

function constelaciones(input, output, title_str, samples)
    % CONSTELACIONES - Plots constellation diagrams for input and output signals
    % Inputs:
    %   input - Original input signal
    %   output - Processed output signal
    %   title_str - Title string for the processed signal plot
    %   samples - Number of samples to plot
    input = input(1:samples); % Limit input signal to specified samples
    output = output(1:samples); % Limit output signal to specified samples
    figure;
    subplot(1, 2, 1); % First subplot: Original signal constellation
    scatter(real(input), imag(input), 'b.');
    title('Señal Original'); grid on;
    % axis([-0.05 0.05 -0.05 0.05]); % Commented: Fixed axis limits
    xlabel('I'); ylabel('Q');
    
    subplot(1, 2, 2); % Second subplot: Processed signal constellation
    scatter(real(output), imag(output), 'r.');
    title(['Señal con ', title_str]); grid on;
    % axis([-0.05 0.05 -0.05 0.05]); % Commented: Fixed axis limits
    xlabel('I'); ylabel('Q');
end

function plot_psd(fs, signal1, signal2, window, overlap, nfft, labels)
    % PLOT_PSD - Plots overlaid power spectral density (PSD) for two signals
    % Inputs:
    %   fs - Sampling frequency (Hz)
    %   signal1 - First signal (original)
    %   signal2 - Second signal (processed)
    %   window - Window size for pwelch
    %   overlap - Number of overlapping samples
    %   nfft - FFT size for PSD calculation
    %   labels - Cell array with legend labels {label1, label2}
    
    % Calculate PSD using Welch's method
    [pxx1, fr] = pwelch(signal1, window, overlap, nfft, fs, 'centered', 'psd');
    [pxx2, ~] = pwelch(signal2, window, overlap, nfft, fs, 'centered', 'psd');
    
    % Plot PSD
    figure;
    plot(fr/1e6, 10*log10(pxx1)+30, 'b', 'LineWidth', 2, 'DisplayName', labels{1});
    hold on;
    plot(fr/1e6, 10*log10(pxx2)+30, 'r', 'LineWidth', 2, 'DisplayName', labels{2});
    hold off;
    
    % Configure plot
    xlabel('Frecuencia (MHz)'); % Frequency axis in MHz
    ylabel('PSD (dBm/Hz)'); % PSD in dBm/Hz
    title('Densidad Espectral de Potencia'); % Plot title
    grid on;
    legend('show'); % Display legend
end

function rx_signal = apply_rician_channel(signal, fs, fc, v, TDL_D_nd, TDL_D_pow, DS_desired, K_desired, seed)
    % APPLY_RICIAN_CHANNEL - Applies a Rician fading channel to the input signal
    % Inputs:
    %   signal - Input signal (e.g., input_tx)
    %   fs - Sampling frequency (Hz, e.g., 2e6)
    %   fc - Carrier frequency (Hz)
    %   v - Receiver speed (km/h, e.g., 0.5)
    %   TDL_D_nd - Normalized delay vector (in units of delay spread)
    %   TDL_D_pow - Path gain vector (dB)
    %   DS_desired - Desired RMS delay spread (seconds, e.g., 1000e-9)
    %   K_desired - Desired K factor (dB, e.g., 6)
    %   seed - Random seed ('auto' or fixed number)
    % Outputs:
    %   rx_signal - Signal after Rician channel effects
    
    % Adjust K factor if different from reference (13.3 dB)
    kfact = 13.3;
    TDL_D_pow_scaled = TDL_D_pow; % Copy original path gains
    TDL_D_pow_scaled(2:end) = TDL_D_pow(2:end) - (K_desired - kfact); % Adjust non-LOS gains
    TDL_pow_l_scaled = 10.^(TDL_D_pow_scaled / 10); % Convert to linear scale
    
    % Calculate mean delay and RMS delay spread for normalization
    tau_m_scaled = sum(TDL_D_nd .* TDL_pow_l_scaled) / sum(TDL_pow_l_scaled);
    RMS_ds_scaled = sqrt(sum(TDL_pow_l_scaled .* (TDL_D_nd - tau_m_scaled).^2) / sum(TDL_pow_l_scaled));
    TDL_D_nd_renorm = TDL_D_nd ./ RMS_ds_scaled; % Normalize delays
    
    % Scale delays and use adjusted gains
    TDL_delays = TDL_D_nd_renorm * DS_desired; % Scale delays to desired RMS delay spread
    TDL_power = TDL_D_pow_scaled; % Use scaled path gains
    
    % Calculate Doppler shift
    c = physconst('LightSpeed'); % Speed of light (m/s)
    fD = v * fc / (c * 3.6); % Maximum Doppler shift (Hz)
    fpS = 0.7 * fD; % Doppler shift for LOS path
    K = 10.^(K_desired / 10); % Linear K factor
    
    % Configure Rician channel with unique seed (or specified by user)
    if strcmp(seed, 'auto')
        rng(randi(2^31 - 1));
    else
        rng(seed);
    end

    ricianChan = comm.RicianChannel( ...
        'SampleRate', fs, ... % Sampling frequency
        'KFactor', K, ... % Linear K factor
        'MaximumDopplerShift', fD, ... % Maximum Doppler shift
        'PathDelays', TDL_delays, ... % Scaled path delays
        'AveragePathGains', TDL_power, ... % Scaled path gains
        'NormalizePathGains', false, ... % Disable automatic gain normalization
        'DirectPathDopplerShift', fpS, ... % Doppler shift for LOS path
        'ChannelFiltering', true, ... % Enable channel filtering
        'PathGainsOutputPort', false, ... % Disable path gains output
        'FadingTechnique', "Filtered Gaussian noise", ... % Use filtered Gaussian noise
        'Visualization', 'Off'); % Disable visualization
    
    % Apply channel to signal
    tic;
    rx_signal = ricianChan(signal); % Pass signal through Rician channel
    fprintf('Señal pasada por el canal Rician en %.2f segundos\n', toc); % Log processing time
    
    % Clean up
    clear('ricianChan'); % Release channel object
end

function [rx_signal, gains_dB] = apply_rayleigh_channel(signal, fs, fc, v, norm_delays, gains_dB, DS_desired, seed)
    % APPLY_RAYLEIGH_CHANNEL - Applies a Rayleigh fading channel to the input signal
    % Inputs:
    %   signal - Input signal (e.g., input_tx)
    %   fs - Sampling frequency (Hz, e.g., 2e6)
    %   fc - Carrier frequency (Hz)
    %   v - Receiver speed (km/h, e.g., 0.5)
    %   norm_delays - Normalized delay vector (in units of delay spread)
    %   gains_dB - Path gain vector (dB)
    %   DS_desired - Desired RMS delay spread (seconds, e.g., 1000e-9)
    %   seed - Random seed ('auto' or fixed number)
    % Outputs:
    %   rx_signal - Signal after Rayleigh channel effects
    
    % Scale delays and use provided gains
    TDL_delays = norm_delays * DS_desired; % Scale delays to desired RMS delay spread
    TDL_power = gains_dB; % Use provided path gains
    
    % Calculate Doppler shift
    c = physconst('LightSpeed'); % Speed of light (m/s)
    fD = v * fc / (c * 3.6); % Maximum Doppler shift (Hz)
    
    % Configure Rayleigh channel with unique seed (or specified by user)
    if strcmp(seed, 'auto')
        rng(randi(2^31 - 1));
    else
        rng(seed);
    end

    rayChan = comm.RayleighChannel( ...
        'SampleRate', fs, ... % Sampling frequency
        'MaximumDopplerShift', fD, ... % Maximum Doppler shift
        'PathDelays', TDL_delays, ... % Scaled path delays
        'AveragePathGains', TDL_power, ... % Path gains
        'PathGainsOutputPort', true, ... % Disable path gains output
        'NormalizePathGains', false, ... % Disable automatic gain normalization
        'Visualization', 'Off'); % Disable visualization
    
    % Apply channel to signal
    tic;
    [rx_signal, gains_dB] = rayChan(signal); % Pass signal through Rayleigh channel
    fprintf('Señal pasada por el canal Rayleigh en %.2f segundos\n', toc); % Log processing time
    
    % Verificar potencia
    rx_power = mean(abs(rx_signal).^2);
    tx_power = mean(abs(signal).^2);
    disp(['Relación potencia rx/tx: ', num2str(rx_power / tx_power)]);

    % Clean up
    clear('rayChan'); % Release channel object
end

function [snr_windowed, snr_inst] = calculate_snr(signal, noise, mask, window_size)
    % CALCULATE_SNR - Calculates instantaneous and windowed SNR
    % Inputs:
    %   signal - Input signal (e.g., input_tx)
    %   noise - Noise signal
    %   mask - Indices for signal masking (e.g., find(abs(input_tx) >= 0.005))
    %   window_size - Size of the window for windowed SNR calculation
    % Outputs:
    %   snr_windowed - Windowed SNR (dB, vector)
    %   snr_inst - Instantaneous SNR (dB, vector)
    
    % Apply mask if provided
    if ~isempty(mask)
        signal = signal(mask); % Apply mask to signal
        noise = noise(mask); % Apply mask to noise
        fprintf('Máscara aplicada: %d muestras seleccionadas\n', length(mask)); % Log mask application
    else
        fprintf('Sin máscara: analizando toda la señal (%d muestras)\n', length(signal)); % Log full signal analysis
    end
    
    % Calculate instantaneous SNR
    pow_inst = abs(signal).^2; % Signal power
    pow_n_inst = abs(noise).^2; % Noise power
    snr_inst = 10*log10(pow_inst ./ pow_n_inst); % Instantaneous SNR in dB
    
    % Calculate windowed SNR
    num_samples = length(signal); % Number of samples
    num_windows = floor(num_samples / window_size); % Number of windows
    snr_windowed = zeros(num_windows, 1); % Initialize windowed SNR array
    
    for i = 1:num_windows
        idx_start = (i-1) * window_size + 1; % Window start index
        idx_end = i * window_size; % Window end index
        window_signal = signal(idx_start:idx_end); % Extract signal window
        window_noise = noise(idx_start:idx_end); % Extract noise window
        pow_signal = mean(abs(window_signal).^2); % Mean signal power
        pow_noise = mean(abs(window_noise).^2); % Mean noise power
        snr_windowed(i) = 10*log10(pow_signal / pow_noise); % Windowed SNR in dB
    end
    disp(['SNR inst min: ', num2str(min(snr_inst)), ', max: ', num2str(max(snr_inst))]);
    disp(['SNR inst min (windowed): ', num2str(min(snr_windowed)), ', max: ', num2str(max(snr_windowed)), ', mean: ' num2str(mean(snr_windowed))]);
end

function [rx_n, noise] = apply_snr(signal_tx, signal_rx, snr_db, mask, seed)
    % APPLY_SNR - Scales signal and adds AWGN to achieve desired SNR
    % Inputs:
    %   signal_tx - Transmitted signal (reference)
    %   signal_rx - Received signal
    %   snr_db - Desired SNR (dB)
    %   mask - Indices for signal masking
    %   seed - Random seed ('auto' or fixed number)
    % Outputs:
    %   rx_n - Signal with added AWGN
    %   noise - Generated noise vector
    
    % Set random number generator seed
    if strcmp(seed, 'auto')
        rng(randi(2^31 - 1));
    else
        rng(seed);
    end

    % Calculate scaling factor based on masked signal powers
    tx_power = mean(abs(signal_tx(mask)).^2); % Transmitted signal power
    rx_power = mean(abs(signal_rx(mask)).^2); % Received signal power
    a = sqrt(tx_power / rx_power); % Scaling factor
    disp(['Applying scaling factor: ', num2str(a)]); % Log scaling factor
    rx_scaled = a * signal_rx; % Scale received signal
    
    % Calculate noise power for desired SNR
    s_power = mean(abs(rx_scaled(mask)).^2); % Scaled signal power
    n_power = s_power / 10^(snr_db / 10); % Noise power for desired SNR
    noise = sqrt(n_power / 2) * (randn(1, length(signal_rx)) + 1j * randn(1, length(signal_rx))).'; % Generate complex AWGN
    rx_n = rx_scaled + noise; % Add noise to received signal
    disp(['tx_p:', num2str(tx_power), ', rx_p:', num2str(rx_power), ', s_p:', num2str(s_power), ', n_p:', num2str(n_power)])
end