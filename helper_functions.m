function varargout =  helper_functions(func_name, varargin)
    switch lower(func_name)
        case 'nuevoanalizadorspec'
            varargout{1} = nuevoanalizadorSpec(varargin{:});
        case 'plot_time_phase'
            plot_time_phase(varargin{:});
        case 'constelaciones'
            constelaciones(varargin{:});
        case 'plot_psd'
            plot_psd(varargin{:});
        case 'apply_rician_channel'
            varargout{1} = apply_rician_channel(varargin{:});
        case 'calculate_snr'
            [varargout{1}, varargout{2}] = calculate_snr(varargin{:});
        case 'apply_snr'
            [varargout{1}, varargout{2}] = apply_snr(varargin{:});
        case 'apply_rayleigh_channel'
            varargout{1} = apply_rayleigh_channel(varargin{:});
        otherwise
            error('Funcion desconocida: %s', func_name);
    end
end
function sa = nuevoanalizadorSpec(fs, signal1, signal2, title)
    sa = spectrumAnalyzer(...
        'SampleRate', fs, ...
        'SpectrumType', 'power-density', ...
        'SpectrumUnits', 'dBm', ...
        'Method', 'welch', ...
        'RBWSource', 'property', ...
        'RBW', 50, ...
        'ChannelNames', title,...
        'ShowLegend', true);
    reset(sa);
    sa(signal1, signal2);
    release(sa);
end

function plot_time_phase(input, output, title_str, indices, fs, unwrap_angles)
    t = indices/fs;
    figure;
    subplot(2,2,1);
    plot(t*1e3, real(input(indices)), 'b', 'DisplayName', 'I'); hold on;
    plot(t*1e3, imag(input(indices)), 'r', 'DisplayName', 'Q');
    title('Señal original'); xlabel('Tiempo (ms)'); ylabel('Amplitud');
    legend; grid on;
    
    subplot(2,2,3);
    plot(t*1e3, real(output(indices)), 'b', 'DisplayName', 'I'); hold on;
    plot(t*1e3, imag(output(indices)), 'r', 'DisplayName', 'Q');
    title(['Señal con ', title_str]); xlabel('Tiempo (ms)'); ylabel('Amplitud');
    legend; grid on;
    
    subplot(2,2,2);
    if unwrap_angles
        plot(t*1e3, unwrap(angle(input(indices))), 'DisplayName', 'Fase');
    else
        plot(t*1e3, 1*(angle(input(indices))), 'DisplayName', 'Fase');
    end
    title('Fase original'); xlabel('Tiempo (ms)'); ylabel('Fase (rad)');
    grid on;
    subplot(2,2,4);
    if unwrap_angles
        plot(t*1e3, unwrap(angle(output(indices))), 'DisplayName', 'Fase');
    else
        plot(t*1e3, 1*(angle(output(indices))), 'DisplayName', 'Fase');
    end
    title(['Fase con ', title_str]); xlabel('Tiempo (ms)'); ylabel('Fase (rad)');
    grid on;
end

function constelaciones(input, output, title_str, samples)
    input = input(1:samples);
    output = output(1:samples);
    figure;
    subplot(1, 2, 1);
    scatter(real(input), imag(input), 'b.');
    title('Señal Original'); grid on;
    % axis([-0.05 0.05 -0.05 0.05]);
    xlabel('I'); ylabel('Q');
    
    subplot(1, 2, 2);
    scatter(real(output), imag(output), 'r.');
    title(['Señal con ', title_str]); grid on;
    % axis([-0.05 0.05 -0.05 0.05]);
    xlabel('I'); ylabel('Q');
end

function plot_psd(fs, signal1, signal2, window, overlap, nfft, labels)
    % PLOT_PSD_COMPARISON - Grafica los espectros PSD superpuestos de dos señales
    % Inputs:
    %   fs - Frecuencia de muestreo (Hz)
    %   signal1 - Primera señal (original)
    %   signal2 - Segunda señal (afectada)
    %   window - Tamaño de la ventana para pwelch
    %   overlap - Número de muestras de solapamiento
    %   nfft - Tamaño de la FFT
    %   labels - Cell array con etiquetas para la leyenda {label1, label2}
    
    % Calcular PSD con pwelch
    [pxx1, fr] = pwelch(signal1, window, overlap, nfft, fs, 'centered', 'psd');
    [pxx2, ~] = pwelch(signal2, window, overlap, nfft, fs, 'centered', 'psd');
    
    % Graficar
    figure;
    plot(fr/1e6, 10*log10(pxx1)+30, 'b', 'LineWidth', 2, 'DisplayName', labels{1});
    hold on;
    plot(fr/1e6, 10*log10(pxx2)+30, 'r', 'LineWidth', 2, 'DisplayName', labels{2});
    hold off;
    
    % Configurar el gráfico
    xlabel('Frecuencia (MHz)');
    ylabel('PSD (dBm/Hz)');
    title('Densidad Espectral de Potencia');
    grid on;
    legend('show');
end
function rx_signal = apply_rician_channel(signal, fs, fc , v, TDL_D_nd, TDL_D_pow, DS_desired, K_desired)
    % APPLY_RICIAN_CHANNEL - Aplica un canal Rician a la señal de entrada
    % Inputs:
    %   signal - Señal de entrada (ej. input_tx)
    %   fs - Frecuencia de muestreo (Hz, ej. 2e6)
    %   fc -  Frecuencia portadora
    %   v - Velocidad (km/h, ej. 0.5)
    %   TDL_D_nd - Vector de retardos normalizados (en unidades de DS)
    %   TDL_D_pow - Vector de potencias relativas (dB)
    %   DS_desired - RMS Delay Spread deseado (segundos, ej. 1000e-9)
    %   K_desired - Factor K en dB (ej. 6)
    % Outputs:
    %   rx_signal - Señal afectada por el canal Rician
    
    % Modificación en torno al factor K (si es diferente de 13.3)
    kfact = 13.3;
    TDL_D_pow_scaled = TDL_D_pow; % Copiar potencias originales
    TDL_D_pow_scaled(2:end) = TDL_D_pow(2:end) - (K_desired - kfact);
    TDL_pow_l_scaled = 10.^(TDL_D_pow_scaled / 10);
    
    % Calcular tau_m y RMS delay spread para normalizar retardos
    tau_m_scaled = sum(TDL_D_nd .* TDL_pow_l_scaled) / sum(TDL_pow_l_scaled);
    RMS_ds_scaled = sqrt(sum(TDL_pow_l_scaled .* (TDL_D_nd - tau_m_scaled).^2) / sum(TDL_pow_l_scaled));
    TDL_D_nd_renorm = TDL_D_nd ./ RMS_ds_scaled;
    
    % Escalar retardos y usar potencias ajustadas
    TDL_delays = TDL_D_nd_renorm * DS_desired; % Escalar valores de retardo
    TDL_power = TDL_D_pow_scaled; % Potencia relativa de los caminos
    
    % Desplazamiento Doppler
    c = physconst('LightSpeed'); % Velocidad de la luz
    fD = v * fc / (c * 3.6); % Desplazamiento Doppler máximo
    fpS = 0.7 * fD; % Pico Doppler para el primer tap
    K = 10.^(K_desired / 10); % Factor K lineal
    
    % Configurar canal Rician con una semilla única
    seed_channel = randi(2^31 - 1);
    rng(seed_channel);
    ricianChan = comm.RicianChannel( ...
        'SampleRate', fs, ...
        'KFactor', K, ...
        'MaximumDopplerShift', fD, ...
        'PathDelays', TDL_delays, ...
        'AveragePathGains', TDL_power, ...
        'NormalizePathGains', false, ...
        'DirectPathDopplerShift', fpS, ...
        'ChannelFiltering', true, ...
        'PathGainsOutputPort', false, ...
        'FadingTechnique', "Filtered Gaussian noise", ...
        'Visualization', 'Off');
    
    % Pasar la señal por el canal
    tic;
    rx_signal = ricianChan(signal);
    fprintf('Señal pasada por el canal Rician en %.2f segundos\n', toc);
    
    % Limpiar objeto
    clear('ricianChan');
end

function rx_signal = apply_rayleigh_channel(signal, fs, fc , v, norm_delays, gains_dB, DS_desired)
    % APPLY_RICIAN_CHANNEL - Aplica un canal Rician a la señal de entrada
    % Inputs:
    %   signal - Señal de entrada (ej. input_tx)
    %   fs - Frecuencia de muestreo (Hz, ej. 2e6)
    %   fc -  Frecuencia portadora
    %   v - Velocidad (km/h, ej. 0.5)
    %   norm_delays - Vector de retardos normalizados (en unidades de DS)
    %   gains_dB - Vector de potencias relativas (dB)
    %   DS_desired - RMS Delay Spread deseado (segundos, ej. 1000e-9)
    %   K_desired - Factor K en dB (ej. 6)
    % Outputs:
    %   rx_signal - Señal afectada por el canal Rician
    
    % Escalar retardos y usar potencias ajustadas
    TDL_delays = norm_delays * DS_desired; % Escalar valores de retardo
    TDL_power = gains_dB; % Potencia relativa de los caminos
    
    % Desplazamiento Doppler
    c = physconst('LightSpeed'); % Velocidad de la luz
    fD = v * fc / (c * 3.6); % Desplazamiento Doppler máximo
    
    % Configurar canal Rician con una semilla única
    seed_channel = randi(2^31 - 1);
    rng(seed_channel);
    rayChan = comm.RayleighChannel( ...
        SampleRate=fs, ...
        MaximumDopplerShift=fD, ...
        PathDelays=TDL_delays, ...
        AveragePathGains=TDL_power, ...
        PathGainsOutputPort=false, ...
        NormalizePathGains=false, ...
        Visualization='Off');
    
    % Pasar la señal por el canal
    tic;
    rx_signal = rayChan(signal);
    fprintf('Señal pasada por el canal Rayleigh en %.2f segundos\n', toc);
    
    % Limpiar objeto
    clear('ricianChan');
end

function [snr_windowed, snr_inst] = calculate_snr(signal, noise, mask, window_size)
    % CALCULATE_SNR - Calcula SNR instantánea y por ventanas
    % Inputs:
    %   signal - Señal de entrada (ej. input_tx)
    %   noise - Ruido (ej. noise)
    %   mask - Índices de la máscara (ej. find(abs(input_tx) >= 0.005))
    %   window_size - Tamaño de la ventana para SNR por ventanas
    % Outputs:
    %   snr_windowed - SNR promedio por ventanas (dB, vector)
    %   snr_inst - SNR instantánea (dB, vector)
    
    % Aplicar máscara
    if ~isempty(mask)
        signal = signal(mask);
        noise = noise(mask);
        fprintf('Máscara aplicada: %d muestras seleccionadas\n', length(mask));
    else
        fprintf('Sin máscara: analizando toda la señal (%d muestras)\n', length(signal));
    end
    
    % SNR instantánea
    pow_inst = abs(signal).^2;
    pow_n_inst = abs(noise).^2;
    snr_inst = 10*log10(pow_inst./pow_n_inst);
    
    % SNR por ventanas
    num_samples = length(signal);
    num_windows = floor(num_samples / window_size);
    snr_windowed = zeros(num_windows, 1);
    
    for i = 1:num_windows
        idx_start = (i-1) * window_size + 1;
        idx_end = i * window_size;
        window_signal = signal(idx_start:idx_end);
        window_noise = noise(idx_start:idx_end);
        pow_signal = mean(abs(window_signal).^2);
        pow_noise = mean(abs(window_noise).^2);
        snr_windowed(i) = 10*log10(pow_signal / pow_noise);
    end

end

function [rx_n, noise] = apply_snr(signal_tx, signal_rx, snr_db, mask)
    tx_power = mean(abs(signal_tx(mask)).^2);
    rx_power = mean(abs(signal_rx(mask)).^2);
    a = sqrt(tx_power/rx_power);
    disp(["Aplying scaling factor: " num2str(a)]);
    rx_scaled = a * signal_rx(mask);
    s_power = mean(abs(rx_scaled).^2);
    n_power = s_power / 10^(snr_db/10);
    noise = sqrt(n_power/2)* (randn(1,length(signal_rx)) + 1j*randn(1,length(signal_rx))).';
    rx_n = signal_rx + noise;
end