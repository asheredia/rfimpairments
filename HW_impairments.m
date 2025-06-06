%% HARDWARE IMPAIRMENTS
% cargar archivo de señal en formato MAT
clear all
signal = load('LoRa_SF12_v1.mat');
input_tx = signal.LoRa_SF12_v1;
fs = 2e6; % 2 Msps
mascara = find(abs(input_tx)>=0.005);
%% FFT
L = length(input_tx);
% N = 1024;
% win = hann(N);
% hop = N;
% n_b = floor((length(input_tx) - N)/ hop);
% S = zeros(N, 1, 'single');
% for k = 1:n_b-1
%     idx = (k-1)*hop + (1:N);
%     x_win = input_tx(idx) .* win.';
%     X = fftshift(fft(x_win, N));
%     S = S + abs(X).^2;
% end
% S = S / n_b;
% S_dB = 10*log10(S);
frecuencias = fs/L*(-L/2:L/2-1);
x = input_tx;
X = fftshift(fft(x)); % fft centrada
plot(frecuencias/1e6, 20*log10(abs(X)), LineWidth=2)
%% Welch
[pxx, fr] = pwelch(input_tx,1024,512,1024,fs,"centered","psd");
plot(fr, 10*log10(pxx)+30);
xlabel("Frecuencia (MHz)");
ylabel("PSD (dBm/Hz)");
title("Densidad espectral de potencia");
grid on;
%% ---------------CARRIER FRECUENCY OFFSET-------------------
fc=915e6;
% offset = 100e3; % frecuencia en Hz
ppm = 20; % Si tenemos el argumento en partes por millon
offset = fc*ppm*1e-6;
disp(['offset de ', num2str(offset), ' Hz'])
rx_cfo = frequencyOffset(input_tx, fs, offset);
sascope = spectrumAnalyzer(SampleRate=fs, FrequencySpan="full", ...
    SpectrumType="power-density",...
    SpectrumUnits="dBm",...
    Method="welch",...
    FrequencyResolutionMethod="rbw",...
    RBWSource="property",RBW=50,...
    ShowLegend=true, ChannelNames=["Input signal", "Frequency offset signal"]);
%% frequency domain signals
sascope(input_tx, rx_cfo);
release(sascope);
%% time domain signal
indices = 1:500;
t = indices/fs;
input = input_tx(mascara);
output = rx_cfo(mascara);
figure;
subplot(2,2,1);
plot(t*1e3, real(input(indices)));
hold on;
plot(t*1e3, imag(input(indices)));
title("Señal original");
xlabel("Tiempo (ms)")
subplot(2,2,3);
plot(t*1e3, real(output(indices)));
hold on;
plot(t*1e3, imag(output(indices)));
title("Señal con CFO");
xlabel("Tiempo (ms)")
% ver las fases de la señal
% figure;
subplot(2,2,2);
plot(t*1e3, angle(input(indices)));
title("Señal original - fase");
xlabel("Tiempo (ms)")
subplot(2,2,4);
plot(t*1e3, angle(output(indices)));
title("Señal con CFO - fase");
xlabel("Tiempo (ms)")
%% ---------------IQ IMBALANCE---------------
ampImb = 5;
phImb = 10;
rx_iqi = iqimbal(input_tx, ampImb, phImb);
spectrum = spectrumAnalyzer(...
    SampleRate=fs,...
    SpectrumUnits="dBm",...
    Method='welch',...
    RBWSource="Property", ...
    RBW=50, ...
    ShowLegend=true, ChannelNames=["Input signal", "IQ imbalanced signal"]);
%% frequency domain signals
spectrum(input_tx, rx_iqi);
release(spectrum);
%% time domain signal
indices = 10001:10500;
t = indices/fs;
mascara = find(abs(input_tx)>=0.005);
input = input_tx(mascara);
output = rx_iqi(mascara);
figure;
subplot(2,2,1);
plot(t*1e3, real(input(indices)));
hold on;
plot(t*1e3, imag(input(indices)));
title("Señal original");
xlabel("Tiempo (ms)")
subplot(2,2,3);
plot(t*1e3, real(output(indices)));
hold on;
plot(t*1e3, imag(output(indices)));
title("Señal con IQ imbalance");
xlabel("Tiempo (ms)")
% ver las fases de la señal
% figure;
subplot(2,2,2);
plot(t*1e3, angle(input(indices)));
title("Señal original - fase");
xlabel("Tiempo (ms)")
subplot(2,2,4);
plot(t*1e3, angle(output(indices)));
title("Señal con IQ imbalance - fase");
xlabel("Tiempo (ms)")
%% -------------------PHASE NOISE----------------------
phNzLevel = -90; % in dBc/Hz
pnHzFreqOff = 1e3; % in Hz
pnoise = comm.PhaseNoise(...
    "Level",phNzLevel, ...
    "FrequencyOffset",pnHzFreqOff,...
    "SampleRate",fs);
Y = pnoise(input_tx);
% Y = pnoise(x);
y = Y(mascara);
%% frequency domain signals
spectrum5 = spectrumAnalyzer(...
    SampleRate=fs,...
    SpectrumUnits="dBm",...
    SpectrumType="power-density",...
    Method='welch',...
    RBWSource="Property", ...
    RBW=50,...
    ShowLegend=true, ChannelNames=["Input signal", "Signal with Phase noise"]);
spectrum5(input_tx, Y);
release(spectrum5);
%% graficas en el dominio del tiempo
power_x = 10*log10(mean(abs(input_tx).^2)) + 30;
power_y = 10*log10(mean(abs(Y).^2)) + 30;
indices = 1:500;
t = indices/fs;
input = input_tx(mascara);
output = Y(mascara);
figure;
subplot(2,2,1);
plot(t*1e3, real(input(indices)));
hold on;
plot(t*1e3, imag(input(indices)));
title("Señal original");
xlabel("Tiempo (ms)")
subplot(2,2,3);
plot(t*1e3, real(output(indices)));
hold on;
plot(t*1e3, imag(output(indices)));
title("Señal con Phase noise");
xlabel("Tiempo (ms)")
% ver las fases de la señal
% figure;
subplot(2,2,2);
plot(t*1e3, angle(input(indices)));
title("Señal original - fase");
xlabel("Tiempo (ms)")
subplot(2,2,4);
plot(t*1e6, angle(output(indices)));
title("Señal con Phase noise - fase");
xlabel("Tiempo (ms)")

%% rms phase noise
ph_err = unwrap(angle(Y) - angle(input_tx));
rms_ph_nz_deg = rms(ph_err)*180/pi();
disp(['The computed RMS phase noise is (degrees): ',...
    num2str(rms_ph_nz_deg)]);
%% espectros con RBW ajustado
fc = 0;
freqSpan = 2e6;
sascopeRBW100 = spectrumAnalyzer( ...
    SampleRate=fs, ...
    Method="welch", ...
    FrequencySpan="Span and center frequency", ...
    CenterFrequency=fc, ...
    Span=freqSpan, ...
    RBWSource="Property", ...
    RBW=100, ...
    SpectrumType="Power density", ...
    SpectralAverages=10, ...
    SpectrumUnits="dBm", ...
    YLimits=[-150 10], ...
    Title="Resolution Bandwidth 100 Hz", ...
    ChannelNames={'signal','signal with phase noise'}, ...
    Position=[79 147 605 374]);
sascopeRBW1k = spectrumAnalyzer( ...
    SampleRate=fs, ...
    Method="welch", ...
    FrequencySpan="Span and center frequency", ...
    CenterFrequency=fc, ...
    Span=freqSpan, ...
    RBWSource="Property", ...
    RBW=1000, ...
    SpectrumType="Power density", ...
    SpectralAverages=10, ...
    SpectrumUnits="dBm", ...
    YLimits=[-150 10], ...
    Title="Resolution Bandwidth 1 kHz", ...
    ChannelNames={'signal','signal with phase noise'}, ...
    Position=[685 146 605 376]);
%%
sascopeRBW1k(input_tx, Y);

%% comporbacion
% Parámetros
Level = -65; % dBc/Hz
FrequencyOffset = 50e3; % Hz
SampleRate = 2e6; % Hz

% Calcular ratio = fs / FrequencyOffset
ratio = SampleRate / FrequencyOffset;

% Tabla para determinar el número de coeficientes (según MATLAB)
ratioVec = [10 50 100 500 1e3 5e3 1e4 5e4 1e5 5e5 1e6 5e6 1e7 5e7];
nCoeffVec = 2.^[7 7 7 7 7 10 10 11 12 15 16 18 19 19];
[~, idx] = min(abs(ratio - ratioVec));
nCoeff = nCoeffVec(idx);
fprintf('Número de coeficientes (nCoeff): %d\n', nCoeff);

% Calcular el numerador (num)
num = sqrt(2 * pi * FrequencyOffset * 10^(Level/10));
fprintf('Numerador (num): %.6f\n', num);

% Calcular el denominador (den)
den = [1 cumprod(((2:nCoeff)-2.5)./((2:nCoeff)-1))];
fprintf('Primeros 7 coeficientes de den: [');
fprintf('%.6f ', den(1:7));
fprintf(']\n');

% Para el filtro IIR, el denominador real es [1, -den(2:end)]
den_filter = [1, -den(2:end)];
fprintf('Primeros 5 coeficientes del denominador del filtro: [');
fprintf('%.6f ', den_filter(1:5));
fprintf(']\n');

%% varianza
fs = 2e6;
phNzLevel = -65;
pnHzFreqOff = 50e3;
duration = 120.0;

% Generar señal constante para extraer phi_k
t = 0:1/fs:120-1/fs;
signal_in = ones(size(t), 'like', 1j).';
%
phnoise = comm.PhaseNoise('Level', phNzLevel, 'FrequencyOffset', pnHzFreqOff, 'SampleRate', fs);
signal_out = phnoise(signal_in);

% Extraer phi_k como la fase de signal_out (ya que signal_in tiene fase 0)
phi_k = angle(signal_out);
variance_phi_k = var(phi_k);
disp(['Varianza de phi_k: ', num2str(variance_phi_k), ' rad^2']);

%%
% medir correlacion cruzada
% [cross_corr, lags] = xcorr(input_tx, rxSig, 10, 'normalized');
% grafica de magnitud y fase
% t = 0:1/fs:0.1;
% figure;
% subplot(3,1,1);
% plot(t, abs(input_tx(1:length(t))));
% title("Señal original - Magnitud");
% subplot(3,1,2);
% plot(t, abs(input_tx(1:length(t))));
% title("Señal IQ imb - Magnitud");
% subplot(3,1,3);
% plot(lags./fs, abs(cross_corr));
% title("Correlación cruzada Tx a Rx");
% figure;
% subplot(2,1,1);
% plot(t, angle(input_tx(1:length(t))));
% title("Señal original - fase")
% subplot(2,1,2);
% plot(t, angle(input_tx(1:length(t))));
% title("Señal IQ imb - fase")