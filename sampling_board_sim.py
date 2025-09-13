# sampling_board_sim.py
"""
Sampling board simulator GUI
Stages:
 - FAA (anti-aliasing filter) : lowpass
 - Sample & Hold (natural or instantaneous sampling)
 - Llave Analógica (analog switch) : gating the sampled signal with a pulse train
 - FR (reconstruction filter) : lowpass

Author: ChatGPT (adapt and extend as needed)
"""

import sys
import numpy as np
from scipy import signal
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# DSP helper functions
# ---------------------------

def make_time_vector(duration_s, fs_plot):
    t = np.arange(0, duration_s, 1.0/fs_plot)
    return t

def input_signal(t, kind='sine', f0=50.0):
    if kind == 'sine':
        return np.sin(2*np.pi*f0*t)
    elif kind == 'sum_sines':
        return 0.6*np.sin(2*np.pi*f0*t) + 0.4*np.sin(2*np.pi*(2*f0)*t*0.5)
    elif kind == 'square':
        return signal.square(2*np.pi*f0*t)
    elif kind == 'noise':
        rng = np.random.RandomState(0)
        return rng.normal(scale=0.5, size=t.shape)
    else:
        return np.zeros_like(t)
'''
def butter_lowpass_filter(x, fs, cutoff, order=5):
    # normalized cutoff: Wn = cutoff/(fs/2)
    if cutoff >= fs/2:
        # no filtering if cutoff at/above Nyquist
        return x
    b, a = signal.butter(order, cutoff/(fs/2), btype='low', analog=False)
    y = signal.filtfilt(b, a, x)
    return y
'''
def cauer_lowpass_filter(x, fs, cutoff, order=6, rp=1, rs=40):
    """
    Filtro pasa-bajos Cauer (Elliptic).
    
    Parámetros:
        x : array_like
            Señal de entrada
        fs : float
            Frecuencia de muestreo [Hz]
        cutoff : float
            Frecuencia de corte [Hz]
        order : int
            Orden del filtro
        rp : float
            Rizado máximo en banda de paso [dB]
        rs : float
            Atenuación mínima en banda de stop [dB]
    """
    if cutoff >= fs/2:
        # No se filtra si cutoff >= Nyquist
        return x
    
    # Normalizar frecuencia de corte
    Wn = cutoff / (fs/2)
    
    # Coeficientes del filtro elíptico
    b, a = signal.ellip(order, rp, rs, Wn, btype='low', analog=False)
    
    # Filtro sin desfase
    y = signal.filtfilt(b, a, x)
    return y


def sample_and_hold_track_hold(x, t, fs_sample, duty=0.5):
    """
    Track-and-Hold behavior:
    - During clock high: output follows the input
    - During clock low: output holds the last sampled value

    Parameters:
        x : np.ndarray
            Input signal
        t : np.ndarray
            Time vector
        fs_sample : float
            Sampling frequency
        duty : float
            Clock duty cycle (0-1). Default 0.5 = half period high, half low

    Returns:
        y : np.ndarray
            Output signal after sample-and-hold
        sample_times : np.ndarray
            Times when sampling occurred
        sample_values : np.ndarray
            Values sampled at those times
    """
    Ts = 1.0 / fs_sample
    clk = ((t % Ts) < duty * Ts).astype(float)  # clock waveform (1=high, 0=low)

    y = np.zeros_like(x)
    last_val = 0.0
    sample_times = []
    sample_values = []

    for i in range(len(t)):
        if clk[i] == 1:  # track mode
            last_val = x[i]
            y[i] = x[i]
            # register sample at rising edge
            if i > 0 and clk[i-1] == 0:
                sample_times.append(t[i])
                sample_values.append(last_val)
        else:  # hold mode
            y[i] = last_val

    return y, np.array(sample_times), np.array(sample_values)

def analog_switch(sampled_signal, t, fs_sample, on_fraction=0.5):
    # Simulate an analog switch that is closed for a fraction of the sampling period
    Ts = 1.0/fs_sample
    on_width = on_fraction * Ts
    gate = np.zeros_like(t)
    times = np.arange(0, t[-1]+Ts/2, Ts)
    for st in times:
        gate += (np.abs(t - st) <= (on_width/2)).astype(float)
    return sampled_signal * (gate > 0)

def compute_spectrum(x, fs_plot):
    N = len(x)
    # Use rfft for real input
    X = np.fft.rfft(x * np.hanning(N))
    freqs = np.fft.rfftfreq(N, d=1.0/fs_plot)
    mag = np.abs(X) / N
    return freqs, mag

# ---------------------------
# PyQt5 GUI App
# ---------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

class SamplingSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sampling Board Simulator")
        self.setGeometry(100, 100, 1200, 700)

        # Default parameters
        self.duration = 0.1            # seconds to simulate
        self.fs_plot = 200000         # high-resolution time grid for plotting (Hz)
        self.input_kind = 'sine'
        self.input_f = 1000.0         # input fundamental (Hz)
        self.fs_sample = 8000.0       # sampling frequency (Hz)
        self.aa_cutoff = 3500.0       # FAA cutoff (Hz)
        self.fr_cutoff = 3500.0       # Reconstruction filter cutoff (Hz)
        self.stage_bypass = {
            'FAA': False,
            'S&H': False,
            'Llave': False,
            'FR': False
        }
        self.sample_mode = 'natural'  # 'natural' or 'instantaneous'
        self.llave_on_frac = 0.9

        # Build UI
        self._build_ui()

        # initial plot
        self.update_processing_and_plot()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout()
        central.setLayout(layout)

        # Left pane: controls
        ctrl = QtWidgets.QFrame()
        ctrl.setFixedWidth(320)
        cl = QtWidgets.QVBoxLayout()
        ctrl.setLayout(cl)

        # Input selection
        cl.addWidget(QtWidgets.QLabel("Input signal"))
        self.combo_input = QtWidgets.QComboBox()
        self.combo_input.addItems(['sine', 'sum_sines', 'square', 'noise'])
        self.combo_input.currentTextChanged.connect(self.on_params_changed)
        cl.addWidget(self.combo_input)

        cl.addWidget(QtWidgets.QLabel("Input frequency (Hz)"))
        self.spin_f = QtWidgets.QDoubleSpinBox()
        self.spin_f.setRange(0.1, 100000.0)
        self.spin_f.setValue(self.input_f)
        self.spin_f.setSingleStep(10)
        self.spin_f.valueChanged.connect(self.on_params_changed)
        cl.addWidget(self.spin_f)

        cl.addWidget(QtWidgets.QLabel("Sampling frequency (Hz)"))
        self.spin_fs = QtWidgets.QDoubleSpinBox()
        self.spin_fs.setRange(10.0, 100000.0)
        self.spin_fs.setValue(self.fs_sample)
        self.spin_fs.setSingleStep(100.0)
        self.spin_fs.valueChanged.connect(self.on_params_changed)
        cl.addWidget(self.spin_fs)

        cl.addWidget(QtWidgets.QLabel("Duration (s)"))
        self.spin_dur = QtWidgets.QDoubleSpinBox()
        self.spin_dur.setRange(0.01, 10.0)
        self.spin_dur.setValue(self.duration)
        self.spin_dur.setSingleStep(0.01)
        self.spin_dur.valueChanged.connect(self.on_params_changed)
        cl.addWidget(self.spin_dur)

        cl.addWidget(QtWidgets.QLabel("FAA cutoff (Hz)"))
        self.spin_aa = QtWidgets.QDoubleSpinBox()
        self.spin_aa.setRange(1.0, 100000.0)
        self.spin_aa.setValue(self.aa_cutoff)
        self.spin_aa.valueChanged.connect(self.on_params_changed)
        cl.addWidget(self.spin_aa)

        cl.addWidget(QtWidgets.QLabel("FR cutoff (Hz)"))
        self.spin_fr = QtWidgets.QDoubleSpinBox()
        self.spin_fr.setRange(1.0, 100000.0)
        self.spin_fr.setValue(self.fr_cutoff)
        self.spin_fr.valueChanged.connect(self.on_params_changed)
        cl.addWidget(self.spin_fr)

        cl.addWidget(QtWidgets.QLabel("Sample mode"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(['natural', 'instantaneous'])
        self.combo_mode.currentTextChanged.connect(self.on_params_changed)
        cl.addWidget(self.combo_mode)

        self.chk_ffa = QtWidgets.QCheckBox("Bypass FAA")
        self.chk_ffa.stateChanged.connect(self.on_bypass_changed)
        cl.addWidget(self.chk_ffa)
        self.chk_sh = QtWidgets.QCheckBox("Bypass Sample & Hold")
        self.chk_sh.stateChanged.connect(self.on_bypass_changed)
        cl.addWidget(self.chk_sh)
        self.chk_llave = QtWidgets.QCheckBox("Bypass Llave (analog switch)")
        self.chk_llave.stateChanged.connect(self.on_bypass_changed)
        cl.addWidget(self.chk_llave)
        self.chk_fr = QtWidgets.QCheckBox("Bypass FR (reconstruction)")
        self.chk_fr.stateChanged.connect(self.on_bypass_changed)
        cl.addWidget(self.chk_fr)

        cl.addWidget(QtWidgets.QLabel("Llave ON fraction (0-1)"))
        self.spin_llave = QtWidgets.QDoubleSpinBox()
        self.spin_llave.setRange(0.0, 1.0)
        self.spin_llave.setSingleStep(0.05)
        self.spin_llave.setValue(self.llave_on_frac)
        self.spin_llave.valueChanged.connect(self.on_params_changed)
        cl.addWidget(self.spin_llave)

        btn_update = QtWidgets.QPushButton("Update & Plot")
        btn_update.clicked.connect(self.update_processing_and_plot)
        cl.addWidget(btn_update)

        cl.addStretch()
        layout.addWidget(ctrl)

        # Right pane: plots
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout()
        right.setLayout(rl)

        # Node selection
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel("Node to view:"))
        self.node_combo = QtWidgets.QComboBox()
        self.node_combo.addItems(['Input', 'After FAA', 'After S&H', 'After Llave', 'After FR'])
        self.node_combo.currentTextChanged.connect(self.update_plots_from_cache)
        hl.addWidget(self.node_combo)
        rl.addLayout(hl)

        # Time plot canvas
        self.time_canvas = MplCanvas(self, width=8, height=3)
        rl.addWidget(self.time_canvas)

        # Frequency plot canvas
        self.freq_canvas = MplCanvas(self, width=8, height=3)
        rl.addWidget(self.freq_canvas)

        layout.addWidget(right)

    def on_params_changed(self, _=None):
        # update parameter fields
        self.input_kind = self.combo_input.currentText()
        self.input_f = float(self.spin_f.value())
        self.fs_sample = float(self.spin_fs.value())
        self.duration = float(self.spin_dur.value())
        self.aa_cutoff = float(self.spin_aa.value())
        self.fr_cutoff = float(self.spin_fr.value())
        self.sample_mode = self.combo_mode.currentText()
        self.llave_on_frac = float(self.spin_llave.value())

    def on_bypass_changed(self, _=None):
        self.stage_bypass['FAA'] = self.chk_ffa.isChecked()
        self.stage_bypass['S&H'] = self.chk_sh.isChecked()
        self.stage_bypass['Llave'] = self.chk_llave.isChecked()
        self.stage_bypass['FR'] = self.chk_fr.isChecked()

    def update_processing_and_plot(self):
        # read latest params
        self.on_params_changed()
        self.on_bypass_changed()

        # high-res time axis for plotting and processing
        t = make_time_vector(self.duration, self.fs_plot)

        # Input
        x_in = input_signal(t, kind=self.input_kind, f0=self.input_f)

        # FAA (anti-aliasing) - lowpass before sampling (unless bypass)
        if self.stage_bypass['FAA']:
            x_aa = x_in.copy()
        else:
            x_aa = cauer_lowpass_filter(x_in, self.fs_plot, self.aa_cutoff, order=6)

        # Sampling: S&H
        if self.stage_bypass['S&H']:
            # no sampling (pass-through)
            sampled_signal = x_aa.copy()
            sample_times = np.array([])
            sample_values = np.array([])
        if self.stage_bypass['S&H']:
            sampled_signal = x_aa.copy()
            sample_times, sample_values = np.array([]), np.array([])
        else:
            sampled_signal, sample_times, sample_values = sample_and_hold_track_hold(
            x_aa, t, self.fs_sample, duty=0.5  # adjust duty cycle if needed
    )


        # Llave (analog switch)
        if self.stage_bypass['Llave']:
            after_llave = sampled_signal.copy()
        else:
            after_llave = analog_switch(sampled_signal, t, self.fs_sample, on_fraction=self.llave_on_frac)

        # FR (reconstruction)
        if self.stage_bypass['FR']:
            x_recon = after_llave.copy()
        else:
            # lowpass to smooth pulses; reconstruct approximate analog
            # choose smoothing cutoff (user sets fr_cutoff)
            x_recon = cauer_lowpass_filter(after_llave, self.fs_plot, self.fr_cutoff, order=6)

        # Cache nodes
        self.cache = {
            't': t,
            'Input': x_in,
            'After FAA': x_aa,
            'After S&H': sampled_signal,
            'After Llave': after_llave,
            'After FR': x_recon,
            'sample_times': sample_times,
            'sample_values': sample_values,
            'fs_plot': self.fs_plot
        }

        # Update plots
        self.update_plots_from_cache()

    def update_plots_from_cache(self, _=None):
        node = self.node_combo.currentText()
        t = self.cache['t']
        x = self.cache[node]
        fs_plot = self.cache['fs_plot']

        # Time plot
        ax = self.time_canvas.axes
        ax.clear()
        ax.plot(t, x, linewidth=1)
        # overlay sample points if appropriate
        if node in ['After S&H', 'After Llave', 'After FR', 'After FAA', 'Input']:
            sample_times = self.cache['sample_times']
            sample_values = self.cache['sample_values']
            if sample_times.size > 0:
                # only mark those sample points that lie within t range
                mask = (sample_times >= t[0]) & (sample_times <= t[-1])
                ax.plot(sample_times[mask], sample_values[mask], 'o', markersize=6, label='samples')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{node} — Time domain')
        ax.grid(True)
        self.time_canvas.draw()

        # Frequency plot
        axf = self.freq_canvas.axes
        axf.clear()
        freqs, mag = compute_spectrum(x, fs_plot)
        axf.semilogy(freqs, mag + 1e-12)  # small offset for log scale
        axf.set_xlim(0, min(fs_plot/2, max(5000, self.fs_sample*3)))
        axf.set_xlabel('Frequency (Hz)')
        axf.set_ylabel('Magnitude')
        axf.set_title(f'{node} — Spectrum (FFT)')
        axf.grid(True, which='both', ls='--')
        self.freq_canvas.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    sim = SamplingSimulator()
    sim.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()