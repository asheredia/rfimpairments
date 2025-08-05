%% ---------------------HARDWARE IMPAIRMENTS-----------------------
clear all; close all;

% Cargar señal
signal = load('LoRa_SF12_v1.mat');
input_tx = signal.LoRa_SF12_v1;
fs = 2e6;
mascara = find(abs(input_tx)>=0.005);
input = input_tx(mascara);

%% ----------Parametros de canal--------------
% generales
fc = 915e6;
DS_desired = 100e-9;
v_kmh = 10; 
channel_seed = 2025; % 'auto' o valor fijo
% Parámetros del canal Rician
TDL_D_nd = [0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, ...
            4.042, 7.937, 9.424, 9.708, 12.525];

TDL_D_pow = [-0.2, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, ...
             -27.8, -23.6, -24.8, -30, -27.7];
K_dB = 13.3;
% Parámetros del canal Rayleigh
TDL_C_nd = [0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, ...
            0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, ...
            2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, ...
            7.0427, 8.6523];

TDL_C_pow = [-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, ...
             -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, ...
             -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8];
%% Visualizar efectos en el dominio de frecuencia y tiempo
indices = 1:500000; % Estandarizar índices para graficar
t = indices/fs;

%% -----------SEÑAL SOBRE CANAL AWGN------------
noise_seed = 2025;
rng(noise_seed);
snr_db = 15;
s_power = mean(abs(input).^2);
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(input_tx)) + 1j*randn(1,length(input_tx)));
y = input_tx + noise.';
rx_awgn_act = y(mascara);
%% -----------SEÑAL SOBRE CANAL RICIAN---------
rx_Rician13 = helper_functions('apply_rician_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, K_dB, channel_seed);
% escalar señal y aplicar awgn
snr_db = 15;
[y13, w13] = helper_functions('apply_snr', input_tx, rx_Rician13, snr_db, mascara, noise_seed);

rx_Rician06 = helper_functions('apply_rician_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, 6, channel_seed);
% escalar señal y aplicar awgn
[y06, w06] = helper_functions('apply_snr', input_tx, rx_Rician06, snr_db, mascara, noise_seed);

rx_Rician20 = helper_functions('apply_rician_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, 20, channel_seed);
% escalar señal y aplicar awgn
[y20, w20] = helper_functions('apply_snr', input_tx, rx_Rician20, snr_db, mascara, noise_seed);
% rx_rician_act = y1(mascara);
%% -----------SEÑAL SOBRE CANAL RAYLEIGH---------
rx_Rayleigh = helper_functions('apply_rayleigh_channel', input_tx, fs,...
    fc, v_kmh, TDL_C_nd, TDL_C_pow, DS_desired, 2025);
% escalar señal y aplicar awgn (y[n] = x[n]*h[n] + w[n])
snr_db = 15;
[yRa, wRa] = helper_functions('apply_snr', input_tx, rx_Rayleigh, snr_db, mascara, noise_seed);
%rx_rayleigh_act = yRa(mascara);
%% --------VISUALIZAR RX----------
% awgn
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, y, {'Input signal', 'AWGN Channel signal'});
helper_functions('plot_time_phase',input,rx_awgn_act,['AWGN con SNR ' num2str(snr_db) ' dB'], 1:10000000, fs, true);
%% rician
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, y1, {'Input signal', 'Rician Channel signal'});
helper_functions('plot_time_phase',input, y20(mascara),['Rician ' num2str(K_dB) ' dB + AWGN'], 1:1000, fs, true);
%% rayleigh
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, yRa, {'Input signal', 'Rayleigh Channel signal'});
helper_functions('plot_time_phase',input,yRa(mascara),['Rayleigh DS = ' num2str(DS_desired*1e9) ' ns, AWGN'], 1:1000000, fs, true);
%% Guardar señal en bloques aleatorios
M = 4096;
vector = y06(mascara);
N = floor(length(vector)/M)*M;
bloques = reshape(vector(1:N), M, []).';
% seleccion aleatoria
rng(2025);
idx = randperm(size(bloques, 1), 5000);
IQ_data = bloques(idx,:);
%% señal piloto
sinewave = dsp.SineWave( ...
    Frequency=100000, ...
    SampleRate=fs, ...
    SamplesPerFrame=2e6, ...
    ComplexOutput=true);
x = sinewave();

%% ----------------------SNR INSTANTANEA-------------------------
% snr instantanea tras canal AWGN + Fading Channel
[snr_inst, ins] = helper_functions('calculate_snr', input_tx, noise.', mascara, 4096);
[snr_inst_y13, ~] = helper_functions('calculate_snr', (y13 - w13), w13, mascara, 4096);
[snr_inst_y06, ~] = helper_functions('calculate_snr', (y06 - w06), w06, mascara, 4096);
[snr_inst_y20, ~] = helper_functions('calculate_snr', (y20 - w20), w20, mascara, 4096);
[snr_inst_yRa, ~] = helper_functions('calculate_snr', (yRa - wRa), wRa, mascara, 4096);
%%
figure;
histogram(snr_inst, (snr_db -15):0.1:(snr_db+15), 'Normalization','percentage','EdgeColor','none');
xlim([(snr_db -16) (snr_db+16)])
title('Distribución de Probabilidad');
xlabel('SNR instantanea (dB)');
ylabel('Densidad');
xline(snr_db, 'r--');
hold on;
histogram(snr_inst_y20, (snr_db -15):0.1:(snr_db+15), 'Normalization','percentage','EdgeColor','none');
histogram(snr_inst_y13, (snr_db -15):0.1:(snr_db+15), 'Normalization','percentage','EdgeColor','none');
histogram(snr_inst_y06, (snr_db -15):0.1:(snr_db+15), 'Normalization','percentage','EdgeColor','none');
% hold on;
% histogram(snr_inst_y2, (snr_db -15):0.1:(snr_db+15), 'Normalization','pdf','EdgeColor','none');
legend('Sólo AWGN', 'SNR media' ,'Rician K 20dB + AWGN', 'Rician K 13.3dB + AWGN', 'Rician K 6dB + AWGN');

%% -------------VERSION 2 DE HISTOGRAMAS - Rician ----------
% Para distribucion y comparacion con Rician de diferentes K
figure;
% Compute and plot kernel density estimates
[x1, f1] = ksdensity(snr_inst, linspace(snr_db - 17, snr_db + 17, 1000));
[x2, f2] = ksdensity(snr_inst_y20, linspace(snr_db - 17, snr_db + 17, 1000));
[x3, f3] = ksdensity(snr_inst_y13, linspace(snr_db - 17, snr_db + 17, 1000));
[x4, f4] = ksdensity(snr_inst_y06, linspace(snr_db - 17, snr_db + 17, 1000));

% Plot smooth curves with distinct colors
plot(f1, x1, 'k', 'LineWidth', 2); % Blue
hold on;
plot(f2, x2, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); % Forest Green
plot(f3, x3, 'Color', '#D95319', 'LineWidth', 2); % Magenta
plot(f4, x4, 'Color', '#7E2F8E', 'LineWidth', 2); % Dark Red
% xline(snr_db, 'r--', 'LineWidth', 1.5);

% Set plot properties
xlim([snr_db - 18, snr_db + 18]);
title('Distribución de SNR instantánea');
xlabel('SNR instantánea (dB)');
ylabel('Densidad');
legend('Sólo AWGN', 'Rician K 20dB + AWGN', 'Rician K 13.3dB + AWGN', 'Rician K 6dB + AWGN', 'Location', 'best');
grid on;
hold off;

%% -------------VERSION 2 DE HISTOGRAMAS - Rayleigh ----------
% Para distribucion y comparacion con Rayleigh
figure;
% Compute and plot kernel density estimates
[xa, fa] = ksdensity(snr_inst, linspace(snr_db - 17, snr_db + 17, 1000));
[xb, fb] = ksdensity(snr_inst_yRa, linspace(snr_db - 17, snr_db + 17, 1000));

% Plot smooth curves with distinct colors
plot(fa, xa, 'k', 'LineWidth', 2); % Blue
hold on;
plot(fb, xb, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); % Forest Green
% xline(snr_db, 'r--', 'LineWidth', 1.5);

% Set plot properties
xlim([snr_db - 17.5, snr_db + 17.5]);
title('Distribución de SNR instantánea');
xlabel('SNR instantánea (dB)');
ylabel('Densidad');
legend('Sólo AWGN', 'Rayleigh + AWGN', 'Location', 'best');
grid on;
hold off;


%% -----------RMS Delay Spread Calculation-------------

% Gain paths into a linear scale
TDL_pow_l = 10.^(TDL_D_pow / 10);
TDL_delays = TDL_D_nd * DS_desired;
fD = (1/3.6) * v_kmh * fc / physconst('LightSpeed');
% Mean delay
tau_m = sum(TDL_delays .* TDL_pow_l) / sum(TDL_pow_l);

% RMS Delay Spread (DS_rms)
DS_rms = sqrt(sum(TDL_pow_l .* (TDL_delays - tau_m).^2) / sum(TDL_pow_l));

% Display results
disp(['Retardo Medio: ', num2str(tau_m), ' s']);
disp(['RMS Delay Spread: ', num2str(DS_rms), ' s']);

% ------Coherent bandwidth and time calculation-----

t_coh = 1./(5.6*fD); 
B_coh = 1./(2*pi*DS_rms);

disp(['Tiempo de coherencia: ', num2str(t_coh), ' s']);
disp(['Ancho de banda de coherencia: ', num2str(B_coh), ' Hz']);

%% comportamiento del canal
mean_mag = mean(abs(ganancias));
std_mag = std(abs(ganancias));
%%
figure;
for m = 1:floor(size(ganancias, 2)/3)
    subplot(1, floor(size(ganancias, 2)/3), m);
    histogram(abs(ganancias(:, m)), 50);
    title(['Distribución de magnitud - Trayectoria ', num2str(m)]);
    xlabel('Magnitud');
    ylabel('Frecuencia');
end

%%
[acf, lags] = xcorr(abs(ganancias(:, 1)), 'normalized');
coherence_time = lags(find(acf < 0.5, 1)) / fs; % Tiempo donde la autocorrelación cae por debajo de 0.5
disp(['Tiempo de coherencia aproximado (segundos): ', num2str(coherence_time)]);
%%
max_delay = max(TDL_D_nd * DS_desired);
coherence_bw = 1 / max_delay;
disp(['Ancho de banda de coherencia aproximado (Hz): ', num2str(coherence_bw)]);

%% PASAR TODAS LAS SEÑALES POR CANAL
%% -------------Parallel Processing with Parfor-------------
% Process files in parallel using parfor for acceleration
% parpool("Processes", 3); % Commented: Initialize parallel pool with 20 workers
% Initialize cell arrays to store results
num_files = size(file_list, 1); % Number of files to process
senial_cell = cell(num_files, 1); % Cell array for processed signals
clean_signals = cell(num_files, 1); % Cell array for clean (masked) signals
filename_cell = cell(num_files, 1); % Cell array for output filenames
output_dir = '/media/wicomtec/Datos2/DATASET UPC-LPWAN-1/RAW/muestras'; % Output directory
noise_seed = 2025;
% Process files in parallel
snr_db = 15; % SNR in dB for AWGN application
parfor i = 1:num_files
    % Load signal from .mat file
    mat_file = file_list{i, 1}; % Retrieve .mat file name
    var_name = file_list{i, 2}; % Retrieve variable name inside .mat file
    signal_data = load(mat_file, var_name); % Load signal data
    input_tx = signal_data.(var_name); % Extract signal

    % Display processing information
    disp(['Procesando archivo: ', mat_file, ' con ', num2str(length(input_tx)), ' muestras']);

    % Apply amplitude mask to filter samples
    mascara = abs(input_tx) >= 0.005;

    % disp(['Muestras de señal útil: ', num2str(length(mascara))])

    % Commented: Apply Rician channel model (disabled)
    rx_rician = helper_functions('apply_rician_channel', input_tx, fs, ...
        fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, 6, channel_seed);

    % Scale signal and apply AWGN
    [y1, ~] = helper_functions('apply_snr', input_tx, rx_rician, snr_db, mascara, noise_seed);
    y1_act = y1(mascara);

    % Limit signal to 5M samples
    senial = y1_act(1:10000000);

    % Generate output filename
    save_filename = fullfile(output_dir, strrep(mat_file, '.mat', '_hw_rician.mat'));

    % Store results in cell arrays
    cl = input_tx(mascara); % Apply mask to clean signal
    clean_signals{i} = cl(1:10000000); % Store first 10M samples of clean signal
    senial_cell{i} = senial; % Store processed signal
    filename_cell{i} = save_filename; % Store output filename

    % Clear variables to free memory
    % clear signal_data input_tx rx_rician y1 rx_chan_cfo rx_chan_cfo_act senial;
end

%% Save Files in a Sequential Loop
% Save processed signals to disk using a standard for loop
for i = 1:num_files
    senial = senial_cell{i}; % Retrieve processed signal
    save_filename = filename_cell{i}; % Retrieve output filename
    tic % Start timing
    save(save_filename, 'senial', '-v7.3'); % Save signal in MATLAB v7.3 format
    toc % Display elapsed time
    disp(['Guardado: ', save_filename]); % Log save status
end
% Commented: Clear cell arrays to free memory
% clear senial_cell filename_cell;