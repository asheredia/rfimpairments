%% ---------------------HARDWARE IMPAIRMENTS-----------------------
clear all; close all;

% Cargar señal
signal = load('LoRa_SF12_v1.mat');
input_tx = signal.LoRa_SF12_v1;
fs = 2e6;
mascara = find(abs(input_tx)>=0.005);
input = input_tx(mascara);

%% -----------------Parametros de simulacion--------------------
ppm = 20; % PPM para CFO
fc = 915e6; % Frecuencia portadora
offset = fc * ppm * 1e-6; % CFO en Hz
ampImb = 3; % Desbalance de Amplitud (dB)
phImb = 10; % Desbalance de fase (º)
phNzLevel = -75; % Nivel de ruido de fase (dBc/Hz)
pnHzFreqOff = 10e3; % Offset de frecuencia para ruido de fase (Hz)

%% Aplicar hardware impairments (a TODA la grabacion)
rx_cfo = frequencyOffset(input_tx, fs, offset); % CFO
rx_iqi = iqimbal(input_tx, ampImb, phImb); % IQ imbalance
pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
rx_pn = pnoise(input_tx); % Phase noise

%% Visualizar efectos en el dominio de frecuencia y tiempo
indices = 1:500; % Estandarizar índices para graficar
t = indices/fs;
%% --------------------AWGN funcion----------------------
rng(123);
snr_db = 15;
rx_awgn = awgn(input,snr_db,"measured");
ruido = rx_awgn - input;
disp(['SNR: ' num2str(10*log10(var(input)/var(ruido))) ' dB'])
%% -------------------AWGN manual------------------
rng(123);
snr_db = 15;
s_power = var(input);
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(input)) + 1j*randn(1,length(input)));
rx_awgn_manual = input + noise.';
ruido = rx_awgn_manual - input;
disp(['SNR: ' num2str(10*log10(var(input)/var(ruido))) ' dB'])
%% AGWN grafica
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'Signal over AWGN channel'};
sascope(input, rx_awgn_manual);
release(sascope);
helper_functions('plot_time_phase',t, input, rx_awgn ,['SNR ' num2str(snr_db) ' dB'], indices);
%% ------------------CFO---------------------
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'Frequency offset signal'};
sascope(input_tx, rx_cfo);
release(sascope);
out_cfo = rx_cfo(mascara);
helper_functions('plot_time_phase',t, input, out_cfo ,['CFO ' num2str(offset/1e3) ' kHz'], indices);
%% ------------------IQ Imbalance-----------------
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'IQ imbalanced signal'};
sascope(input_tx, rx_iqi);
release(sascope);
Ia = num2str(ampImb);
Ip = num2str(phImb);
out_iqi = rx_iqi(mascara);
helper_functions('plot_time_phase',t, input, out_iqi, ['IQI: A=' Ia 'dB, P=' Ip 'º'], indices);
%% -------------------Phase Noise-------------------
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'Signal with Phase noise'};
sascope(input_tx, rx_pn);
release(sascope);
ph = num2str(phNzLevel);
froff = num2str(pnHzFreqOff/1e3);
out_rx_pn = rx_pn(mascara);
helper_functions('plot_time_phase',t, input, out_rx_pn, ['Phase noise: ' ph ' dBc/Hz at ' froff ' kHz'], indices);

%% --------AWGN + CFO----------
snr_db = 15;
s_power = var(input);
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(input_tx)) + 1j*randn(1,length(input_tx)));
y = input_tx + noise.';
% y_n = awgn(input, snr_db, 'measured');
% y(mascara) = y_n;
rx_awgn_cfo = frequencyOffset(y, fs, offset);
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'AWGN + Frequency offset signal'};
sascope(input_tx, rx_awgn_cfo);
release(sascope);
out_awgn_cfo = rx_awgn_cfo(mascara);
helper_functions('plot_time_phase',t, input, out_awgn_cfo ,['AWGN + CFO ' num2str(offset/1e3) ' kHz'], indices);

%% --------AWGN + Phase Noise----------
rng(123);
snr_db = 15;
s_power = var(input);
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(input_tx)) + 1j*randn(1,length(input_tx)));
y = input_tx + noise.';
pnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
rx_awgn_phn = pnoise(y);
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'AWGN + Phase Noise signal'};
sascope(input_tx, rx_awgn_phn);
release(sascope);
ph = num2str(phNzLevel);
froff = num2str(pnHzFreqOff/1e3);
out_awgn_phn = rx_awgn_phn(mascara);
helper_functions('plot_time_phase',t, input, out_awgn_phn ,['AWGN + Phase noise: ' ph ' dBc/Hz at ' froff ' kHz'], indices);

%% --------AWGN + IQ IMBALANCE----------
rng(123);
snr_db = 15;
s_power = var(input);
n_power = s_power / 10^(snr_db/10);
noise = sqrt(n_power/2)* (randn(1,length(input_tx)) + 1j*randn(1,length(input_tx)));
y = input_tx + noise.';
rx_awgn_iqi = iqimbal(y, ampImb, phImb);
clear sascope;
sascope = helper_functions('nuevoanalizadorSpec',fs);
sascope.ChannelNames = {'Input signal', 'IQ imbalanced signal'};
sascope(input_tx, rx_awgn_iqi);
release(sascope);
Ia = num2str(ampImb);
Ip = num2str(phImb);
out_awgn_iqi = rx_awgn_iqi(mascara);
helper_functions('plot_time_phase',t, input, out_awgn_iqi, ['AWGN + IQI: A=' Ia 'dB, P=' Ip 'º'], indices);
%% Opcional: Calcular y mostrar PSD con pwelch
[pxx, fr] = pwelch(input_tx, 1024, 512, 1024, fs, 'centered', 'psd');
figure;
plot(fr/1e6, 10*log10(pxx)+30, 'LineWidth', 2);
xlabel('Frecuencia (MHz)'); ylabel('PSD (dBm/Hz)');
title('Densidad espectral de potencia (Señal original)'); grid on;

%% Diagramas de constelacion (efectos aislados)
helper_functions('constelaciones',input, rx_awgn, 'AWGN', 50);
helper_functions('constelaciones',input, out_cfo, 'CFO', 100);
helper_functions('constelaciones',input, out_iqi, 'IQ Imbalance Matlab', 100);
helper_functions('constelaciones',input, out_rx_pn, 'Phase Noise', 100);
% efectos combinados (no es tan evidente debido al AWGN)
helper_functions('constelaciones',input, out_awgn_cfo, 'AWGN + CFO', 100);
helper_functions('constelaciones',input, out_awgn_iqi, 'AWGN + IQI', 100);
helper_functions('constelaciones',input, out_awgn_phn, 'AWGN + PhN', 100);
%% señal piloto
sinewave = dsp.SineWave( ...
    Frequency=10000, ...
    SampleRate=fs/10, ...
    SamplesPerFrame=2e6, ...
    ComplexOutput=true);
x = sinewave();
%% aplicando iq imbalance a una señal de banda ancha con portadora
tx = frequencyOffset(input_tx, fs, 200e3);
rx = iqimbal(tx, 3, 10);
%% Comparacion de PSDs
window = 1024;
overlap = window/2;
nfft=1024;
helper_functions('plot_psd',fs, input, out_awgn_phn, window, overlap, nfft, {'Original', 'awgn+phn'});
%% estimacion de iq imbalance
hicomp = comm.IQImbalanceCompensator('CoefficientOutputPort',true);
[compSig, coef] = step(hicomp, input_tx);
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
%% metricas hw impairments
ruido_fase = unwrap(angle(out_rx_pn)) - unwrap(angle(input));
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
%% evm
err = out_awgn_phn - input;
% evm = std(err)/std(input) * 100;
evm = sqrt(mean(abs(err).^2)/mean(abs(input).^2)) * 100;
fprintf('EVM: %.2f%%\n', evm);


%% otras pruebas


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