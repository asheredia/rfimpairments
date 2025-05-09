function varargout =  helper_functions(func_name, varargin)
    switch lower(func_name)
        case 'nuevoanalizadorspec'
            varargout{1} = nuevoanalizadorSpec(varargin{1});
        case 'plot_time_phase'
            plot_time_phase(varargin{:});
        case 'constelaciones'
            constelaciones(varargin{:});
        case 'plot_psd'
            plot_psd(varargin{:});
        otherwise
            error('Funcion desconocida: %s', func_name);
    end
end
function sascope = nuevoanalizadorSpec(fs)
    sascope = spectrumAnalyzer(...
        'SampleRate', fs, ...
        'SpectrumType', 'power-density', ...
        'SpectrumUnits', 'dBm', ...
        'Method', 'welch', ...
        'RBWSource', 'property', ...
        'RBW', 50, ...
        'ShowLegend', true);
end

function plot_time_phase(t, input, output, title_str, indices)
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
    plot(t*1e3, angle(input(indices)), 'DisplayName', 'Fase');
    title('Fase original'); xlabel('Tiempo (ms)'); ylabel('Fase (rad)');
    grid on;
    
    subplot(2,2,4);
    plot(t*1e3, angle(output(indices)), 'DisplayName', 'Fase');
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