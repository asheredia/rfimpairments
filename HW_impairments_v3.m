clear all;
% List of files to be processed
file_list = {
    'WiSun_mode_1a_v1.mat', 'WiSun_mode_1a_v1';
    'WiSun_mode_1b_v1.mat', 'WiSun_mode_1b_v1';
    'WiSun_mode_2a_v1.mat', 'WiSun_mode_2a_v1';
    'WiSun_mode_2b_v1.mat', 'WiSun_mode_2b_v1';
    'WiSun_mode_3a_v1.mat', 'WiSun_mode_3a_v1';
    'WiSun_mode_4a_v1.mat', 'WiSun_mode_4a_v1';
    'WiSun_mode_4b_v1.mat', 'WiSun_mode_4b_v1';
    'WiSun_mode_5_v1.mat', 'WiSun_mode_5_v1'
    'MR_FSK_mode_1_v1.mat', 'MR_FSK_mode_1_v1';
    'MR_FSK_mode_2_v1.mat', 'MR_FSK_mode_2_v1';
    'MR_FSK_mode_3_v1.mat', 'MR_FSK_mode_3_v1';
    'MR_OFDM1_MCS0_v1.mat', 'MR_OFDM1_MCS0_v1';
    'MR_OFDM1_MCS2_v1.mat', 'MR_OFDM1_MCS2_v1';
    'MR_OFDM2_MCS0_v1.mat', 'MR_OFDM2_MCS0_v1';
    'MR_OFDM2_MCS2_v1.mat', 'MR_OFDM2_MCS2_v1';
    'MR_OFDM2_MCS5_v1.mat', 'MR_OFDM2_MCS5_v1';
    'MR_OFDM3_MCS1_v1.mat', 'MR_OFDM3_MCS1_v1';
    'MR_OFDM3_MCS2_v1.mat', 'MR_OFDM3_MCS2_v1';
    'MR_OFDM3_MCS5_v1.mat', 'MR_OFDM3_MCS5_v1';
    'MR_OFDM4_MCS2_v1.mat', 'MR_OFDM4_MCS2_v1';
    'MR_OFDM4_MCS5_v1.mat', 'MR_OFDM4_MCS5_v1';
    'MR_OQPSK_RM0_v1.mat', 'MR_OQPSK_RM0_v1';
    'MR_OQPSK_RM1_v1.mat', 'MR_OQPSK_RM1_v1';
    'MR_OQPSK_RM2_v1.mat', 'MR_OQPSK_RM2_v1';
    'MR_OQPSK_RM3_v1.mat', 'MR_OQPSK_RM3_v1';
    'HaLow_OFDM_MCS0_v1.mat', 'HaLow_OFDM_MCS0_v1';
    'HaLow_OFDM_MCS1_v1.mat', 'HaLow_OFDM_MCS1_v1';
    'HaLow_OFDM_MCS3_v1.mat', 'HaLow_OFDM_MCS3_v1';
    'HaLow_OFDM_MCS5_v1.mat', 'HaLow_OFDM_MCS5_v1';
    'HaLow_OFDM_MCS10_v1.mat', 'HaLow_OFDM_MCS10_v1';
    'LoRa_SF07_v1.mat', 'LoRa_SF07_v1';    
    'LoRa_SF08_v1.mat', 'LoRa_SF08_v1';   
    'LoRa_SF09_v1.mat', 'LoRa_SF09_v1';
    'LoRa_SF10_v1.mat', 'LoRa_SF10_v1';
    'LoRa_SF11_v1.mat', 'LoRa_SF11_v1';
    'LoRa_SF12_v1.mat', 'LoRa_SF12_v1';
    'LoRa_SF7B_v1.mat', 'LoRa_SF7B_v1';
    'LoRa_SF8C_v1.mat', 'LoRa_SF8C_v1';
    'SigFox_v1.mat', 'SigFox_v1'   
};
% Sample rate (same as the one set on SDR to capture signal data)
fs = 2e6;

%% -----------------HW Impairments - Simulation Parameters--------------------
% Define parameters for hardware impairments simulation
ppm = 0.5; % Carrier Frequency Offset (CFO) in parts per million (ppm)
fc = 915e6; % Carrier frequency in Hz
offset = fc * ppm * 1e-6; % Calculated CFO in Hz based on ppm and carrier frequency
ampImb = 1; % Amplitude imbalance in dB for IQ imbalance simulation
phImb = 1; % Phase mismatch in degrees for IQ imbalance simulation
phNzLevel = [-85 -118 -125 -145]; % Phase noise levels in dBc/Hz at specified frequency offsets
pnHzFreqOff = [1e3 9.5e3 19.5e3 195e3]; % Frequency offsets in Hz where phase noise is applied
% phNzLevel = -90; % Commented: Scalar phase noise level in dBc/Hz (alternative option)
% pnHzFreqOff = 1e3; % Commented: Scalar frequency offset in Hz (alternative option)

%% ----------Channel Simulation Parameters--------------
% Configure parameters for channel simulation (Rayleigh or Rician)
DS_desired = 100e-9; % Desired delay spread in seconds for the channel
v_kmh = 10; % Speed of the receiver in km/h for Doppler effect

% ---Rician Channel Parameters (Power Delay Profile - PDP)---
% Normalized delays for Rician channel model (dimensionless)
TDL_D_nd = [0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, ...
            4.042, 7.937, 9.424, 9.708, 12.525];
% Path gains in dB for Rician channel model
TDL_D_pow = [-0.2, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, ...
             -27.8, -23.6, -24.8, -30, -27.7];
% K factor in dB for Rician channel (ratio of direct to scattered power)
K_dB = 13.3;

% ---Rayleigh Channel Parameters (Power Delay Profile - PDP)---
% Normalized delays for Rayleigh channel model (dimensionless)
TDL_C_nd = [0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, ...
            0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, ...
            2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, ...
            7.0427, 8.6523];
% Path gains in dB for Rayleigh channel model
TDL_C_pow = [-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, ...
             -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, ...
             -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8];
%% -------------Files Processing (One at a Time)-------------
% Process files sequentially without parallel computing
snr_db = 10; % Signal-to-Noise Ratio (SNR) in dB to be applied
for i = 1:size(file_list, 1)
    mat_file = file_list{i, 1}; % Retrieve .mat file name from file list
    var_name = file_list{i, 2}; % Retrieve variable name inside .mat file

    % Load complex signal data from the .mat file
    signal_data = load(mat_file, var_name);
    input_tx = signal_data.(var_name);
    clear("signal_data"); % Clear temporary variable to free memory
    disp(['Samples in file: ', num2str(length(input_tx))]) % Display number of samples
    mascara = find(abs(input_tx) >= 0.005); % Create mask for samples with amplitude >= 0.005
    input = input_tx(mascara); % Apply mask to input signal
    tic % Start timing for performance measurement

    % Commented: Apply Rician channel model (disabled)
    % rx_Rician = helper_functions('apply_rician_channel', input_tx, fs,...
    %     fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, K_dB);

    % Scale signal and apply Additive White Gaussian Noise (AWGN)
    [y1, w1] = helper_functions('apply_snr', input_tx, input_tx, snr_db, mascara);
    % rx_rician_act = y1(mascara); % Commented: Apply mask to Rician channel output

    % Apply Carrier Frequency Offset (CFO) to the signal
    rx_chan_cfo = frequencyOffset(y1, fs, offset);
    rx_chan_cfo_act = rx_chan_cfo(mascara); % Apply mask to CFO-affected signal

    % Commented: Apply IQ imbalance to the signal (disabled)
    % rx_chan_iqi = iqimbal(y1, ampImb, phImb);
    % rx_chan_iqi_act = rx_chan_iqi(mascara);

    % Commented: Apply phase noise to the signal (disabled)
    % pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
    % rx_chan_phn = pnoise(y1);
    % rx_chan_phn_act = rx_chan_phn(mascara);

    % Define output directory for saving processed signals
    output_dir = '/media/wicomtec/Datos2/DATASET UPC-LPWAN-1/RAW/muestras';
    senial = rx_chan_cfo_act(1:5000000); % Limit output signal to 5M samples
    save_filename = fullfile(output_dir, strrep(mat_file, '.mat', '_hw_cfo.mat')); % Generate output filename
    save(save_filename, 'senial', '-v7.3'); % Save processed signal in MATLAB v7.3 format
    toc % Display elapsed time
    disp(['Procesado: ', mat_file, ' -> Guardado como: ', save_filename]); % Log processing status

    % Clear variables to free memory
    clear("signal_data", "rx_signal", "rx_pkt_scaled", 'noise', 'input_tx', 'rayChan');
    clear("HaLow*", "MR_*", "WiSun*", "LoRa*");
end

%% -------------Parallel Processing with Parfor-------------
% Process files in parallel using parfor for acceleration
parpool("Processes", 8); % Commented: Initialize parallel pool with 20 workers
% Initialize cell arrays to store results
num_files = size(file_list, 1); % Number of files to process
senial_cell = cell(num_files, 1); % Cell array for processed signals
clean_signals = cell(num_files, 1); % Cell array for clean (masked) signals
filename_cell = cell(num_files, 1); % Cell array for output filenames
output_dir = '/media/wicomtec/Datos2/DATASET UPC-LPWAN-1/RAW/muestras'; % Output directory
noise_seed = 2025;
% Process files in parallel
snr_db = 20; % SNR in dB for AWGN application
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
    % rx_rician = helper_functions('apply_rician_channel', input_tx, fs, ...
    %     fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, K_dB);

    % Scale signal and apply AWGN
    [y1, ~] = helper_functions('apply_snr', input_tx, input_tx, snr_db, mascara, noise_seed);

    % Commented: Apply CFO to the signal (disabled)
    % rx_chan_cfo = frequencyOffset(y1, fs, offset);
    % rx_chan_cfo_act = rx_chan_cfo(mascara);

    % Commented: Apply phase noise to the signal (disabled)
    pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
    rx_chan_phn = pnoise(y1);
    rx_chan_phn_act = rx_chan_phn(mascara);

    % Apply IQ imbalance to the signal
    % rx_chan_iqi = iqimbal(y1, ampImb, phImb);
    % rx_chan_iqi_act = rx_chan_iqi(mascara); % Apply mask to IQ-imbalanced signal

    % Limit signal to 5M samples
    senial = rx_chan_phn_act(1:10000000);

    % ONLY USE WITH PHASE NOISE:
    % ruido_fase = unwrap(angle(rx_chan_phn_act)) - unwrap(angle(input_tx(mascara)));
    % rms_phnz = rms(ruido_fase)*180/pi();
    % % orf = ruido_fase(mascara);
    % disp(['Phase noise mean power for: ' var_name ' -> ' num2str(var(ruido_fase)) ' rad^2'])

    % Generate output filename
    save_filename = fullfile(output_dir, strrep(mat_file, '.mat', '_hw_phn.mat'));

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

%% Calculate AWGN for a clean signal at a given index (to compare)
rng(noise_seed);
snr_db = 20;
s_power = var(clean_signals{idx});
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(clean_signals{idx})) + 1j*randn(1,length(clean_signals{idx})));
y = clean_signals{idx} + noise.';
%% Plotting
idx = 36;
% Plot time and phase comparison for a specific signal pair
helper_functions('plot_time_phase', y, senial_cell{idx}, ...
    ['Canal AWGN ' num2str(snr_db) ' dB'], 5000:5500, fs, false); % Plot samples 5000 to 6000
%% PSD Plots
window = 1024;
overlap = window/2;
nfft=1024;
helper_functions('plot_psd',fs, y, senial_cell{idx}, window, overlap, nfft, {'Signal with AWGN', 'Impaired + AWGN signal'});
%% Matlab spectrogram with SpectrumAnalyzer
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    y, senial_cell{36}, {'Signal with AWGN', 'Impaired + AWGN signal'});
%% Constellations (useful for IQ imbalance)
helper_functions('constelaciones',y, senial_cell{idx}, 'AWGN + IQI', 100000);