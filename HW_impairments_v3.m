clear all;
file_list = {
    'HaLow_OFDM_MCS0_v1.mat', 'HaLow_OFDM_MCS0_v1';
    'HaLow_OFDM_MCS1_v1.mat', 'HaLow_OFDM_MCS1_v1';
    'HaLow_OFDM_MCS3_v1.mat', 'HaLow_OFDM_MCS3_v1';
    'HaLow_OFDM_MCS5_v1.mat', 'HaLow_OFDM_MCS5_v1';
    'HaLow_OFDM_MCS10_v1.mat', 'HaLow_OFDM_MCS10_v1';
    'LoRa_SF07_v1.mat', 'LoRa_SF07_v1';
    'LoRa_SF7B_v1.mat', 'LoRa_SF7B_v1';
    'LoRa_SF08_v1.mat', 'LoRa_SF08_v1';
    'LoRa_SF8C_v1.mat', 'LoRa_SF8C_v1';
    'LoRa_SF09_v1.mat', 'LoRa_SF09_v1';
    'LoRa_SF10_v1.mat', 'LoRa_SF10_v1';
    'LoRa_SF11_v1.mat', 'LoRa_SF11_v1';
    'LoRa_SF12_v1.mat', 'LoRa_SF12_v1';
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
    'SigFox_v1.mat', 'SigFox_v1';
    'WiSun_mode_1a_v1.mat', 'WiSun_mode_1a_v1';
    'WiSun_mode_1b_v1.mat', 'WiSun_mode_1b_v1';
    'WiSun_mode_2a_v1.mat', 'WiSun_mode_2a_v1';
    'WiSun_mode_2b_v1.mat', 'WiSun_mode_2b_v1';
    'WiSun_mode_3a_v1.mat', 'WiSun_mode_3a_v1';
    'WiSun_mode_4a_v1.mat', 'WiSun_mode_4a_v1';
    'WiSun_mode_4b_v1.mat', 'WiSun_mode_4b_v1';
    'WiSun_mode_5_v1.mat', 'WiSun_mode_5_v1'
};
fs = 2e6;

%% -----------------Parametros de simulacion--------------------
ppm = 5; % PPM para CFO
fc = 915e6; % Frecuencia portadora
offset = fc * ppm * 1e-6; % CFO en Hz
ampImb = 1.761; % Desbalance de Amplitud (dB)
phImb = 50; % Desbalance de fase (º)
phNzLevel = [-85 -118 -125 -145]; % Nivel de ruido de fase (dBc/Hz)
pnHzFreqOff = [1e3 9.5e3 19.5e3 195e3]; % Offset de frecuencia para ruido de fase (Hz)
% phNzLevel = -90; % Nivel de ruido de fase (dBc/Hz)
% pnHzFreqOff = 1e3; % Offset de frecuencia para ruido de fase (Hz)

%% ----------Parametros de canal--------------
% generales
DS_desired = 100e-9;
v_kmh = 10;
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
%% ------------- PROCESAR ARCHIVOS -------------
snr_db = 10;
for i = 1:size(file_list, 1)
    mat_file = file_list{i, 1};   % Nombre del archivo .mat
    var_name = file_list{i, 2};   % Nombre de la variable dentro del archivo .mat
    
    % Cargar la señal desde el archivo
    signal_data = load(mat_file, var_name);
    input_tx = signal_data.(var_name);
    clear("signal_data");
    disp(['muestras del archivo: ', num2str(length(input_tx))])
    mascara = find(abs(input_tx)>=0.005);
    input = input_tx(mascara);
    tic
    % rx_Rician = helper_functions('apply_rician_channel', input_tx, fs,...
    %     fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, K_dB);
    % escalar señal y aplicar awgn
    [y1, w1] = helper_functions('apply_snr', input_tx, input_tx, snr_db, mascara);
    % rx_rician_act = y1(mascara);
    % %------ Aplicar CFO a señal y1-------
    rx_chan_cfo = frequencyOffset(y1, fs, offset);
    rx_chan_cfo_act = rx_chan_cfo(mascara);
    % -------aplicar IQ imbalance a y1-------
    %rx_chan_iqi = iqimbal(y1, ampImb, phImb);
    %rx_chan_iqi_act = rx_chan_iqi(mascara);
    % -------Aplicar Phase Noise a y1--------
    % pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
    % rx_chan_phn = pnoise(y1);
    % rx_chan_phn_act = rx_chan_phn(mascara);

    output_dir = '/media/wicomtec/Datos2/DATASET UPC-LPWAN-1/RAW/muestras';
    senial = rx_chan_cfo_act(1:5000000);
    save_filename = fullfile(output_dir, strrep(mat_file, '.mat', '_hw_cfo.mat'));
    save(save_filename, 'senial', '-v7.3'); 
    toc
    disp(['Procesado: ', mat_file, ' -> Guardado como: ', save_filename]);
    % Liberar memoria
    clear("signal_data","rx_signal", "rx_pkt_scaled", 'noise', 'input_tx', 'rayChan');
    clear("HaLow*", "MR_*", "WiSun*", "LoRa*");
end

%% PROCESAMIENTO CON BUCLE PARFOR - ACELERADO
%% ------------- PROCESAR ARCHIVOS CON PARFOR -------------
% parpool("Processes",20);
% Inicializar celdas para almacenar resultados
num_files = size(file_list, 1);
senial_cell = cell(num_files, 1);
clean_signals = cell(num_files, 1);
filename_cell = cell(num_files, 1);
output_dir = '/media/wicomtec/Datos2/DATASET UPC-LPWAN-1/RAW/muestras';

% Procesar archivos en paralelo
snr_db = 10;
parfor i = 1:num_files
    % Cargar la señal desde el archivo
    mat_file = file_list{i, 1};   % Nombre del archivo .mat
    var_name = file_list{i, 2};   % Nombre de la variable dentro del archivo .mat
    signal_data = load(mat_file, var_name);
    input_tx = signal_data.(var_name);
    
    % Mostrar información
    disp(['Procesando archivo: ', mat_file, ' con ', num2str(length(input_tx)), ' muestras']);
    
    % Aplicar máscara
    mascara = abs(input_tx) >= 0.005;
    
    % Aplicar canal Rician
    % rx_rician = helper_functions('apply_rician_channel', input_tx, fs, ...
    %     fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, K_dB);
    
    % Escalar señal y aplicar AWGN (recolocar despues del canal)
    [y1, ~] = helper_functions('apply_snr', input_tx, input_tx, snr_db, mascara);
    
    
    % Aplicar CFO
    % rx_chan_cfo = frequencyOffset(y1, fs, offset);
    % rx_chan_cfo_act = rx_chan_cfo(mascara);

    
    % -------Aplicar Phase Noise a y1--------
    % pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
    % rx_chan_phn = pnoise(y1);
    % rx_chan_phn_act = rx_chan_phn(mascara);

    % -------aplicar IQ imbalance a y1-------
    rx_chan_iqi = iqimbal(y1, ampImb, phImb);
    rx_chan_iqi_act = rx_chan_iqi(mascara);
    
    % Limitar la señal a 5M muestras
    senial = rx_chan_iqi_act(1:5000000);
    
    % Generar nombre del archivo de salida
    save_filename = fullfile(output_dir, strrep(mat_file, '.mat', '_hw_iqi.mat'));
    
    % Almacenar resultados en celdas
    cl = input_tx(mascara);
    clean_signals{i} = cl(1:5000000);
    senial_cell{i} = senial;
    filename_cell{i} = save_filename;
    
    % Liberar memoria
    %clear signal_data input_tx rx_rician y1 rx_chan_cfo rx_chan_cfo_act senial;
end

%% Guardar archivos en un bucle for normal
for i = 1:num_files
    senial = senial_cell{i};
    save_filename = filename_cell{i};
    tic
    save(save_filename, 'senial', '-v7.3');
    toc
    disp(['Guardado: ', save_filename]);
end

% Liberar memoria de las celdas
%clear senial_cell filename_cell;
%% graficar
helper_functions('plot_time_phase',clean_signals{14},senial_cell{14},['Canal AWGN ' num2str(snr_db) ' dB'], 5000:6000, fs, true);