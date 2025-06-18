%% ---------------------HARDWARE IMPAIRMENTS-----------------------
% clear all; close all;

% Cargar señal
signal = load('LoRa_SF12_v1.mat');
input_tx = signal.LoRa_SF12_v1;
fs = 2e6;
mascara = find(abs(input_tx)>=0.005);
input = input_tx(mascara);

%% -----------------Parametros de simulacion--------------------
ppm = 10; % PPM para CFO
fc = 915e6; % Frecuencia portadora
offset = fc * ppm * 1e-6; % CFO en Hz
ampImb = 3; % Desbalance de Amplitud (dB)
phImb = 10; % Desbalance de fase (º)
phNzLevel = -75; % Nivel de ruido de fase (dBc/Hz)
pnHzFreqOff = 10e3; % Offset de frecuencia para ruido de fase (Hz)

%% ----------Parametros de canal--------------
% generales
DS_desired = 300e-9;
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

%% >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
%        Aplicar hardware impairments en TX
%  >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
tx_cfo = frequencyOffset(input_tx, fs, offset); % CFO (poco aplicable en TX)
tx_iqi = iqimbal(input_tx, ampImb, phImb); % IQ imbalance
pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
tx_pn = pnoise(input_tx); % Phase noise
%% ------------------CFO---------------------
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, tx_cfo, {'Input signal', 'Frequency offset signal'});
tx_cfo_act = tx_cfo(mascara);
helper_functions('plot_time_phase', ...
    input, tx_cfo_act ,['CFO ' num2str(offset/1e3) ' kHz'], indices, fs);
%% ------------------IQ Imbalance-----------------
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, tx_iqi, {'Input signal', 'IQ imbalanced signal'});
Ia = num2str(ampImb);
Ip = num2str(phImb);
tx_iqi_act = tx_iqi(mascara);
helper_functions('plot_time_phase',...
    input, tx_iqi_act, ['IQI: A=' Ia 'dB, P=' Ip 'º'], indices, fs, true);
%% -------------------Phase Noise-------------------
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, tx_pn, {'Input signal', 'Phase Noise signal'});
ph = num2str(phNzLevel);
froff = num2str(pnHzFreqOff/1e3);
tx_pn_act = tx_pn(mascara);
helper_functions('plot_time_phase',...
    input, tx_pn_act, ['Phase noise: ' ph ' dBc/Hz at ' froff ' kHz'], indices, fs, true);

%% >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
%        Aplicar hardware impairments en RX
%  >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<
%% -----------SEÑAL SOBRE CANAL AWGN------------
noise_seed = 2025;
rng(noise_seed);
snr_db = 15;
s_power = var(input);
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(input_tx)) + 1j*randn(1,length(input_tx)));
y = input_tx + noise.';
rx_awgn_act = y(mascara);
%% -----------SEÑAL SOBRE CANAL RICIAN---------
rx_Rician = helper_functions('apply_rician_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, K_dB, channel_seed);
% escalar señal y aplicar awgn
snr_db = 15;
[y1, w1] = helper_functions('apply_snr', input_tx, rx_Rician, snr_db, mascara, noise_seed);

rx_Rician6 = helper_functions('apply_rician_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, 6, channel_seed);
% escalar señal y aplicar awgn
[y1_6, w1_6] = helper_functions('apply_snr', input_tx, rx_Rician6, snr_db, mascara, noise_seed);

rx_Rician20 = helper_functions('apply_rician_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, 20, channel_seed);
% escalar señal y aplicar awgn
[y1_20, w1_20] = helper_functions('apply_snr', input_tx, rx_Rician20, snr_db, mascara, noise_seed);
% rx_rician_act = y1(mascara);
%% -----------SEÑAL SOBRE CANAL RAYLEIGH---------
rx_Rayleigh = helper_functions('apply_rayleigh_channel', input_tx, fs,...
    fc, v_kmh, TDL_D_nd, TDL_D_pow, DS_desired, 2025);
% escalar señal y aplicar awgn (y[n] = x[n]*h[n] + w[n])
snr_db = 15;
[y2, w2] = helper_functions('apply_snr', input_tx, rx_Rayleigh, snr_db, mascara, noise_seed);
rx_rayleigh_act = y2(mascara);
%% --------VISUALIZAR RX----------
% awgn
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, y, {'Input signal', 'AWGN Channel signal'});
helper_functions('plot_time_phase',input,rx_awgn_act,['AWGN con SNR ' num2str(snr_db) ' dB'], indices, fs, true);
%% rician
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, y1, {'Input signal', 'Rician Channel signal'});
helper_functions('plot_time_phase',input,rx_rician_act,['Rician ' num2str(K_dB) ' dB + AWGN'], 1:2000, fs, true);
%% rayleigh
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, y2, {'Input signal', 'Rayleigh Channel signal'});
helper_functions('plot_time_phase',input,rx_rayleigh_act,['Rayleigh DS = ' num2str(DS_desired*1e9) ' ns, AWGN'], 1:1000000, fs, true);
%% --------CANAL + CFO----------
rx_chan_cfo = frequencyOffset(y1, fs, offset);
rx_chan_cfo_act = rx_chan_cfo(mascara);
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, rx_chan_cfo, {'Input signal', 'Rician Channel + Frequency offset signal'});
helper_functions('plot_time_phase',...
    input, rx_chan_cfo_act ,['Rician + CFO ' num2str(offset/1e3) ' kHz'], 1:2000, fs, false);
%% primero channel, luego CFO, luego AWGN
rx_chan_cfo2 = frequencyOffset(rx_Rician, fs, offset);
[y12, w12] = helper_functions('apply_snr', input_tx, rx_chan_cfo2, snr_db, mascara);
rx_chan_cfo_act2 = y12(mascara);
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, rx_chan_cfo, {'Input signal', 'Rician Channel + Frequency offset signal'});
helper_functions('plot_time_phase',...
    input, rx_chan_cfo_act2 ,['Rician + CFO ' num2str(offset/1e3) ' kHz'], 1:2000, fs, true);
%% --------CANAL + Phase Noise----------
pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
rx_chan_phn = pnoise(y1);
ph = num2str(phNzLevel);
froff = num2str(pnHzFreqOff/1e3);
rx_chan_phn_act = rx_chan_phn(mascara);
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, rx_chan_phn, {'Input signal', 'Rician + Phase Noise signal'});
helper_functions('plot_time_phase',input, rx_chan_phn_act , ...
    ['Rician + Phase noise: ' ph ' dBc/Hz at ' froff ' kHz'], 1:2000, fs, true);
%% primero channel, luego phase noise, luego AWGN
pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
rx_chan_phn2 = pnoise(rx_Rician);
[y13, w13] = helper_functions('apply_snr', input_tx, rx_chan_phn2, snr_db, mascara);
rx_chan_phn_act2 = y13(mascara);
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, rx_chan_phn2, {'Input signal', 'Rician + Phase Noise signal'});
helper_functions('plot_time_phase',input, rx_chan_phn_act2 , ...
    ['Rician + Phase noise: ' ph ' dBc/Hz at ' froff ' kHz'], 1:2000, fs, true);
%% --------CANAL + IQ IMBALANCE----------
rx_chan_iqi = iqimbal(y1, ampImb, phImb);
Ia = num2str(ampImb);
Ip = num2str(phImb);
rx_chan_iqi_act = rx_chan_iqi(mascara);
clear sa;
sa = helper_functions('nuevoanalizadorSpec',fs, ...
    input_tx, rx_chan_iqi, {'Input signal', 'Rician + IQI signal'});
helper_functions('plot_time_phase',input, rx_chan_iqi_act, ...
    ['Rician + IQI: A=' Ia 'dB, P=' Ip 'º'], 1:2000, fs, false);
%% Opcional: Calcular y mostrar PSD con pwelch
[pxx, fr] = pwelch(input_tx, 1024, 512, 1024, fs, 'centered', 'psd');
figure;
plot(fr/1e6, 10*log10(pxx)+30, 'LineWidth', 2);
xlabel('Frecuencia (MHz)'); ylabel('PSD (dBm/Hz)');
title('Densidad espectral de potencia (Señal original)'); grid on;

%% --------------Diagramas de constelacion (en TX)-----------
helper_functions('constelaciones',input, rx_awgn_act, 'AWGN', 50);
helper_functions('constelaciones',input, tx_cfo_act, 'CFO', 1000);
helper_functions('constelaciones',input, tx_iqi_act, 'IQ Imbalance Matlab', 1000);
helper_functions('constelaciones',input, tx_pn_act, 'Phase Noise', 1000);
% efectos en RX (no es tan evidente debido al paso por el canal)
helper_functions('constelaciones',input, rx_chan_cfo_act, 'Channel + CFO', 100);
helper_functions('constelaciones',input, rx_chan_iqi_act, 'Channel + IQI', 100);
helper_functions('constelaciones',input, rx_chan_phn_act, 'Channel + PhN', 100);
%% señal piloto
sinewave = dsp.SineWave( ...
    Frequency=100000, ...
    SampleRate=fs, ...
    SamplesPerFrame=2e6, ...
    ComplexOutput=true);
x = sinewave();
%%
phNzLevel = -75; % Phase noise levels in dBc/Hz at specified frequency offsets
pnHzFreqOff = 10e3; % Frequency offsets in Hz where phase noise is applied
pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
distorted = pnoise(x);
%% aplicando iq imbalance a una señal de banda ancha con portadora
tx = frequencyOffset(input_tx, fs, 200e3);
rx = iqimbal(tx, 3, 10);
%%
helper_functions('plot_time_phase',...
    input, frequencyOffset(input, fs, 5*915), 'Phase noise:', 5000:5500, fs, false);
%%
function constelacionesr(input, output1, output2, output3, output4, title_str, samples)
    % CONSTELACIONES - Plots a single constellation diagram with input and three output signals overlaid
    % Inputs:
    %   input - Original input signal
    %   output1 - First processed output signal
    %   output2 - Second processed output signal
    %   output3 - Third processed output signal
    %   title_str - Title string for the plot
    %   samples - Number of samples to plot
    input = input(1:samples); % Limit input signal to specified samples
    output1 = output1(1:samples); % Limit first output signal to specified samples
    output2 = output2(1:samples); % Limit second output signal to specified samples
    output3 = output3(1:samples); % Limit third output signal to specified samples
    output4 = output4(1:samples);
    
    figure;
    hold on; % Enable overlaying plots
    plot(real(output1), imag(output1),'Color', '#A2142F', 'DisplayName', ['Salida 1 con ', title_str], 'LineWidth',3);
    plot(real(output2), imag(output2), 'Color', '#77AC30', 'DisplayName', ['Salida 2 con ', title_str], 'LineWidth',3);
    plot(real(output3), imag(output3), 'Color', '#7E2F8E', 'DisplayName', ['Salida 3 con ', title_str], 'LineWidth',3);
    plot(real(output4), imag(output4), 'Color', '#4DBEEE', 'DisplayName', ['Salida 4 con ', title_str], 'LineWidth',3);
    plot(real(input), imag(input), 'Color', '#0072BD', 'DisplayName', 'Señal Original', 'LineWidth',4);
    
    hold off;
    
    title(['Constelación: ', title_str]);
    xlabel('I'); ylabel('Q');
    grid on;
    legend('show'); % Display legend to distinguish signals
    % axis([-0.05 0.05 -0.05 0.05]); % Commented: Fixed axis limits
end
constelacionesr(x, iqimbal(x,1,1), iqimbal(x,1,5), iqimbal(x,3,10), iqimbal(x,5,15), 'IQI', 10000);
%% --------------------Comparacion de PSDs-----------------------
window = 1024;
overlap = window/2;
nfft=1024;
helper_functions('plot_psd',fs, x, iqimbal(x, 1, 1), window, 0, nfft, {'Original', 'Distorted'});
%% ESTIMACION DE IQ IMBALANCE MATLAB
hicomp = comm.IQImbalanceCompensator('CoefficientOutputPort',true);
[compSig, coef] = step(hicomp, rx_chan_iqi);
[aest, pest] = iqcoef2imbal(coef(end));
disp(['Ganancia de amplitud estimada: ' num2str(aest) ' dB']);
disp(['Desfase estimado: ' num2str(pest) ' grados']);
%% phase noise meaurements (experimental)
PNtarget = [-101 -105 -106 -109 -132];
FreqOff = [1e3 5e3 10e3 100e3 1e6];
f1 = fr(fr>=0);
p1 = pxx(fr>=0);
p1 = 10*log10(p1)+30;
PNMeasure = phaseNoiseMeasure(f1,p1,25e3,FreqOff,'on','Phase Noise', PNtarget);
%% ----------------metricas hw impairments-------------------
ruido_fase = unwrap(angle(tx_pn_act)) - unwrap(angle(input));
rms_phnz = rms(ruido_fase)*180/pi();
% orf = ruido_fase(mascara);
disp(['Potencia media del ruido de fase: ' num2str(var(ruido_fase)) ' rad^2'])
disp(['Ruido de fase RMS en grados: ' num2str(rms_phnz)])
disp(['Desviación estándar: ' num2str(std(ruido_fase))])
figure;
histogram(ruido_fase, 100, 'Normalization','pdf')
xlabel('Ruido de fase (rad)');
ylabel('Densidad de probabilidad');
title('Histograma del ruido de fase');
%% ------------------evm----------------------
err = tx_pn_act - input;
% evm = std(err)/std(input) * 100;
evm = sqrt(mean(abs(err).^2)/mean(abs(input).^2)) * 100;
fprintf('EVM: %.2f%%\n', evm);

%% ----------------------SNR INSTANTANEA-------------------------
% snr instantanea tras canal AWGN + Fading Channel
[snr_inst, ins] = helper_functions('calculate_snr', input_tx, noise.', mascara, 128);
[snr_inst_y13, ~] = helper_functions('calculate_snr', rx_Rician, w1, mascara, 128);
[snr_inst_y6, ~] = helper_functions('calculate_snr', rx_Rician6, w1_6, mascara, 128);
[snr_inst_y20, ~] = helper_functions('calculate_snr', rx_Rician20, w1_20, mascara, 128);
[snr_inst_y2, ~] = helper_functions('calculate_snr', rx_Rayleigh, w2, mascara, 128);
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
histogram(snr_inst_y6, (snr_db -15):0.1:(snr_db+15), 'Normalization','percentage','EdgeColor','none');
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
[x4, f4] = ksdensity(snr_inst_y6, linspace(snr_db - 17, snr_db + 17, 1000));

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
[xb, fb] = ksdensity(snr_inst_y2, linspace(snr_db - 17, snr_db + 17, 1000));

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

%% evolucion de las snrs ventaneadas
figure;
plot((1:length(ins)) * 2048, ins);
xlabel('Sample Index (Window Center)');
ylabel('Windowed SNR (dB)');
title(['Windowed SNR (Target SNR = ' num2str(snr_db) ' dB)']);


%% diferencias de fase instantanea

% Extraer componentes I y Q
I = real(vara);
Q = imag(vara);

% Calcular fase instantánea de cada componente
phase_I = unwrap(atan2(imag(hilbert(I)), I));
phase_Q = unwrap(atan2(imag(hilbert(Q)), Q));

% Diferencia de fase (debería ser ±90°)
phase_diff = phase_Q - phase_I;
phase_diff_deg = phase_diff * 180/pi;

% Desviación de la ortogonalidad ideal (90°)
ortogonality_error = abs(mod(abs(phase_diff_deg), 180) - 90);
figure;
plot(1:length(phase_diff_deg),phase_diff_deg)
%% producto escalar normalizado
% Normalizar las componentes
I_norm = I / sqrt(sum(I.^2));
Q_norm = Q / sqrt(sum(Q.^2));

% Producto escalar (debería ser 0 para ortogonalidad perfecta)
dot_product = I_norm .* Q_norm;

% Ortogonalidad instantánea
ortogonality_instant = abs(dot_product);

% Ángulo entre vectores
angle_between = acos(abs(dot_product)) * 180/pi;

figure;
plot(1:length(angle_between),angle_between)
%% evaluacion de varianzas, magnitud de error y error de phase
v_in = var(input);
v_rx_cfo = var(out_cfo);
v_rx_iqi = var(tx_iqi_act);
v_rx_phn = var(out_rx_pn);
v_rx_awgn_cfo = var(rx_chan_cfo);
v_rx_awgn_iqi = var(rx_chan_iqi_act);
v_rx_awgn_phn = var(rx_chan_phn_act);
errpower_cfo = mean(abs(out_cfo - input).^2);
errpower_iqi = mean(abs(tx_iqi_act - input).^2);
errpower_phn = mean(abs(out_rx_pn - input).^2);
errphase_cfo = mean(abs(unwrap(angle(out_cfo) - angle(input))));
errphase_iqi = mean(abs(unwrap(angle(tx_iqi_act) - angle(input))));
errphase_phn = mean(abs(unwrap(angle(out_rx_pn) - angle(input))));






%% iq imbalance asimetrico
gR = 1; % dB
phiR = 8; % degree
gR = 10^(gR/10);
phiR = phiR*pi()/180;
K1 = (1+gR*exp(-1i*phiR))/2;
K2 = (1-gR*exp(1i*phiR))/2;
rx_iqi_asim = K1.*input_tx + K2.*conj(input_tx);
helper_functions('constelaciones',input, rx_iqi_asim(mascara), 'IQ Imbalance asim', 100);
%% iq imbalance simetrico
gR = 1; % dB
phiR = 8; % degree
gR = 10^(gR/10);
phiR = phiR*pi()/180;
alphaR = cos(phiR/2) + 1i*gR*sin(phiR/2);
betaR = gR*cos(phiR/2) - 1i*sin(phiR/2);
rx_iqi_sim = alphaR.*input_tx + betaR.*conj(input_tx);
helper_functions('constelaciones',input, rx_iqi_sim(mascara), 'IQ Imbalance sim', 100);

%% iq imbalance simetrico (version 2)
gR = 1; % dB
phiR = 8; % degree
gR = 10^(gR/10);
phiR = phiR*pi()/180;
alphaR = cos(phiR) - 1i*gR*sin(phiR);
betaR = gR*cos(phiR) + 1i*sin(phiR);
rx_iqi_sim2 = alphaR.*input_tx - betaR.*conj(input_tx);
helper_functions('constelaciones',input, rx_iqi_sim2(mascara), 'IQ Imbalance sim2', 100);
%% calculo de la IRR segun el algoritmo IQI de matlab
% Parámetros
gR = 1; % dB
phiR = 5; % grados
gR = 10^(gR/10); % Escala lineal: 1.1220
phiR = phiR * pi / 180; % Radianes: 0.0873

% Calcular términos
sqrt_gR = sqrt(gR); % 1.0593
cos_phi = cos(phiR/2); % 0.9991
sin_phi = sin(phiR/2); % 0.0436

% Calcular alphaR y betaR
alphaR = ((gR + 1) / (2 * sqrt_gR)) * cos_phi + 1j * ((1 - gR) / (2 * sqrt_gR)) * sin_phi;
betaR = ((gR - 1) / (2 * sqrt_gR)) * cos_phi - 1j * ((gR + 1) / (2 * sqrt_gR)) * sin_phi;

% Mostrar resultados
fprintf('alphaR = %.4f + j %.4f\n', real(alphaR), imag(alphaR)); % 1.0005 - j 0.0025
fprintf('betaR = %.4f + j %.4f\n', real(betaR), imag(betaR)); % 0.0575 - j 0.0437

% Calcular IRR
IRR = 10 * log10(abs(alphaR)^2 / abs(betaR)^2);
fprintf('IRR = %.2f dB\n', IRR); % 22.83 dB



%% ANALISIS POR SEPARADO A UNA SEÑAL CON CONSTELACION
f_s = 1000000;
M = 16;
snrdB = 30;
refConst = qammod(0:M-1,M,UnitAveragePower=true);
data = randi([0 M-1], 1000, 1);
modSig = qammod(data,M,UnitAveragePower=true);

%%
% recibida = iqimbal(modSig, 3, 10);
% recibida = frequencyOffset(modSig, f_s, 5);
p_noise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', f_s);
% recibida = pnoise(modSig);
% recibida = awgn(iqimbal(modSig, 3, 10), 30, 'measured');
recibida = awgn(pnoise(modSig), 30, 'measured');
constDiagram = comm.ConstellationDiagram(...
    ReferenceConstellation=refConst);
constDiagram(recibida);
release(constDiagram);
%%
rChan = comm.RayleighChannel(...
    SampleRate=50000,...
    MaximumDopplerShift=4,...
    PathDelays=[0 2e-8], ...
    AveragePathGains=[0 -9]);
cd = comm.ConstellationDiagram;
M = 4;
for n = 1:125
    tx = randi([0 M-1], 500, 1);
    pskSig = pskmod(tx,M,pi/4);
    fadedSig = rChan(pskSig);
    update(cd,fadedSig);
end
% helper_functions('plot_psd',f_s, modSig, recibida, window, overlap, nfft, {'Original', 'Recibida'});

%% -----------------GUARDAR ARCHIVO------------------
output_dir = '/media/wicomtec/Datos2/DATASET UPC-LPWAN-1/RAW/muestras';
senial = rx_chan_cfo_act(1:5000000);
save_filename = fullfile(output_dir, strrep('WiSun_mode_3a_v1.mat', '.mat', '_hw_cfo.mat'));
save(save_filename, 'senial', '-v7.3');

