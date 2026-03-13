import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ====================
# 1. KONFIGURASI & MODEL FISIKA
# ====================

@dataclass
class TankConfig:
    """Konfigurasi parameter sistem tangki air asrama"""
    radius: float = 1.0              # meter
    max_height: float = 4.0          # meter
    area_outlet: float = 0.02        # m² (ukuran pipa keluar)
    g: float = 9.81                  # m/s²
    initial_h: float = 0.5           # meter (ketinggian awal)
    v_in: float = 1.5                # m/s (kecepatan air masuk)
    area_inlet: float = 0.05         # m² (luas penampang inlet)
    simulation_time: float = 60.0    # menit
    time_step: float = 1.0           # detik
    
    tank_area: float = field(init=False)
    max_volume: float = field(init=False)
    
    def __post_init__(self):
        self.tank_area = np.pi * (self.radius ** 2)
        self.max_volume = self.tank_area * self.max_height

class TankSimulator:
    def __init__(self, config: TankConfig):
        self.config = config
        self.results = None

    def tank_dynamics(self, t, h):
        """ODE: dh/dt = (Qin - Qout) / Area_Tangki"""
        # Proteksi: Ketinggian tidak bisa negatif
        h = max(0, h[0])
        
        # Aliran masuk (Pompa) - Berhenti jika penuh (float switch logic)
        q_in = self.config.v_in * self.config.area_inlet if h < self.config.max_height * 0.98 else 0
        
        # Aliran keluar (Hukum Torricelli)
        v_out = np.sqrt(2 * self.config.g * h)
        q_out = self.config.area_outlet * v_out
        
        dh_dt = (q_in - q_out) / self.config.tank_area
        return [dh_dt]

    def run_simulation(self):
        t_span = (0, self.config.simulation_time * 60)
        t_eval = np.linspace(0, t_span[1], 500)
        
        sol = solve_ivp(
            self.tank_dynamics, 
            t_span, 
            [self.config.initial_h], 
            t_eval=t_eval,
            method='RK45'
        )
        
        self.time_history = sol.t / 60  # Ke menit
        self.height_history = sol.y[0]
        self.volume_history = self.height_history * self.config.tank_area
        
        # Hitung Metrik
        full_idx = np.where(self.height_history >= self.config.max_height * 0.95)[0]
        time_to_full = self.time_history[full_idx[0]] if len(full_idx) > 0 else None
        
        return {
            'time': self.time_history,
            'height': self.height_history,
            'volume': self.volume_history,
            'max_h': np.max(self.height_history),
            'final_v': self.volume_history[-1],
            'time_to_full': time_to_full
        }

# ====================
# 2. ANTARMUKA STREAMLIT
# ====================

def main():
    st.set_page_config(page_title="Simulasi Tangki Air Asrama", layout="wide")
    
    st.title("💧 Simulasi Dinamika Fluida: Tangki Air Asrama")
    st.markdown("""
    Aplikasi ini memodelkan perubahan ketinggian air dalam tangki berdasarkan **Hukum Torricelli** dan laju pengisian pompa. Gunakan panel di kiri untuk mengubah skenario.
    """)

    # --- SIDEBAR / INPUT ---
    st.sidebar.header("⚙️ Parameter Sistem")
    
    with st.sidebar.expander("Dimensi Tangki", expanded=True):
        r = st.slider("Radius Tangki (m)", 0.5, 5.0, 1.0, 0.1)
        h_max = st.slider("Tinggi Maksimum (m)", 2.0, 10.0, 4.0, 0.5)
        h0 = st.slider("Ketinggian Awal (m)", 0.0, h_max, 0.5, 0.1)

    with st.sidebar.expander("Sistem Pompa & Pipa", expanded=True):
        v_in = st.slider("Kecepatan Pompa (m/s)", 0.0, 5.0, 1.5, 0.1)
        a_out = st.slider("Luas Pipa Outlet (m²)", 0.005, 0.1, 0.02, 0.005)

    sim_time = st.sidebar.number_input("Durasi Simulasi (menit)", 10, 300, 60)

    # --- EKSEKUSI SIMULASI ---
    config = TankConfig(
        radius=r, max_height=h_max, initial_h=h0, 
        v_in=v_in, area_outlet=a_out, simulation_time=float(sim_time)
    )
    
    simulator = TankSimulator(config)
    res = simulator.run_simulation()

    # --- DISPLAY METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kapasitas Maks", f"{config.max_volume:.2f} m³")
    col2.metric("Tinggi Maks Tercapai", f"{res['max_h']:.2f} m")
    col3.metric("Volume Akhir", f"{res['final_v']:.2f} m³")
    
    status_full = f"{res['time_to_full']:.1f} Menit" if res['time_to_full'] else "Tidak Penuh"
    col4.metric("Waktu Sampai Penuh", status_full)

    # --- VISUALISASI ---
    tab1, tab2 = st.tabs(["📈 Grafik Dinamis", "📋 Tabel Data"])

    with tab1:
        # Create Plotly Figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Garis Ketinggian
        fig.add_trace(
            go.Scatter(x=res['time'], y=res['height'], name="Tinggi Air (m)", 
                       line=dict(color='royalblue', width=4)),
            secondary_y=False,
        )
        
        # Garis Volume
        fig.add_trace(
            go.Scatter(x=res['time'], y=res['volume'], name="Volume Air (m³)", 
                       line=dict(color='lightblue', dash='dash')),
            secondary_y=True,
        )

        # Annotasi Limit
        fig.add_hline(y=config.max_height, line_dash="dot", line_color="red", 
                      annotation_text="Batas Meluap")

        fig.update_layout(
            title="Profil Ketinggian dan Volume Air terhadap Waktu",
            xaxis_title="Waktu (menit)",
            hovermode="x unified",
            height=600
        )
        fig.update_yaxes(title_text="Ketinggian (meter)", secondary_y=False)
        fig.update_yaxes(title_text="Volume (m³)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df = pd.DataFrame({
            'Waktu (menit)': res['time'],
            'Ketinggian (m)': res['height'],
            'Volume (m³)': res['volume']
        })
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Simulasi (CSV)", csv, "hasil_tangki.csv", "text/csv")

if __name__ == "__main__":
    main()