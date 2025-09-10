import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.optimize import least_squares, differential_evolution
import io
import time

# Set page config
st.set_page_config(
    page_title="Geophysical Self-Potential Inversion",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4a90e2 0%, #7bb3f0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .status-panel {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .tab-header {
        background: #e9ecef;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border-left: 4px solid #007bff;
    }
    .method-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
    }
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.25rem;
        text-align: center;
    }
    .citation-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Forward Models (keeping all existing models exactly as they were)
def forward_inclined_sheet_5param(params, x_inp):
    """Forward model for inclined sheet with 5 parameters (from least squares code)"""
    x0, alpha, h, K, a = params
    
    # Ensure physical constraints
    if h <= 0 or a <= 0:
        return np.full_like(x_inp, 1e10)
        
    # Calculate sheet endpoints in rotated coordinate system
    b = a * np.cos(alpha)  # horizontal half-projection
    d = a * np.sin(alpha)  # vertical half-projection
    
    sp_calc = np.zeros_like(x_inp)
    
    for i, x in enumerate(x_inp):
        # Distance to upper end of sheet
        r1_sq = (x - x0 + b)**2 + (h - d)**2
        
        # Distance to lower end of sheet  
        r2_sq = (x - x0 - b)**2 + (h + d)**2
        
        # Avoid numerical issues
        if r1_sq <= 0 or r2_sq <= 0:
            sp_calc[i] = 0
        else:
            ratio = r1_sq / r2_sq
            ratio = np.clip(ratio, 1e-10, 1e10)
            sp_calc[i] = K * np.log(ratio)
            
    return sp_calc

def forward_sphere_cylinder(par, x_inp, q=1.5):
    """Forward model for sphere/cylinder (q=1.5 sphere, 1.0 horizontal cylinder, 0.5 vertical cylinder)"""
    var_x0, var_alpha, var_h, var_k = par[0], par[1], par[2], par[3]
    var_sp = []
    for i in x_inp:
        var_up = (i - var_x0) * np.cos(var_alpha) - var_h * np.sin(var_alpha)
        var_down = ((i - var_x0)**2 + var_h**2) ** (q)
        var = var_k * (var_up / var_down)
        var_sp.append(var)
    return np.array(var_sp)

# Enhanced Least Squares Implementation
class LSQInverter:
    def __init__(self, position, data, model_type):
        self.position = np.array(position)
        self.data = np.array(data)
        self.model_type = model_type
        
    def forward_model(self, params):
        if self.model_type == "Inclined Sheet (5-param)":
            return forward_inclined_sheet_5param(params, self.position)
        elif self.model_type == "Sphere":
            return forward_sphere_cylinder(params, self.position, q=1.5)
        elif self.model_type == "Horizontal Cylinder":
            return forward_sphere_cylinder(params, self.position, q=1.0)
        elif self.model_type == "Vertical Cylinder":
            return forward_sphere_cylinder(params, self.position, q=0.5)
    
    def objective_function(self, params):
        synthetic = self.forward_model(params)
        return synthetic - self.data
    
    def estimate_initial_parameters(self):
        """Enhanced parameter estimation from least squares code"""
        if self.model_type == "Inclined Sheet (5-param)":
            # x0: position of extremum
            if abs(np.min(self.data)) > abs(np.max(self.data)):
                x0_est = self.position[np.argmin(self.data)]
                K_est = np.min(self.data) * 2
            else:
                x0_est = self.position[np.argmax(self.data)]
                K_est = np.max(self.data) * 2
                
            # Rough depth estimate from half-width
            anomaly_center = x0_est
            half_amplitude = K_est / 2
            
            # Find half-width points
            distances = np.abs(self.data - half_amplitude)
            half_width_idx = np.argsort(distances)[:2]
            half_width = np.abs(np.diff(self.position[half_width_idx]))[0] / 2
            
            h_est = max(half_width, 5)  # Minimum 5m depth
            alpha_est = np.pi/4  # 45 degrees
            a_est = half_width * 1.5  # Estimate half-length
            
            return [x0_est, alpha_est, h_est, K_est, a_est]
        else:
            # 4-parameter models
            if abs(np.min(self.data)) > abs(np.max(self.data)):
                x0_est = self.position[np.argmin(self.data)]
                K_est = np.min(self.data) * 2
            else:
                x0_est = self.position[np.argmax(self.data)]
                K_est = np.max(self.data) * 2
                
            pos_range = self.position.max() - self.position.min()
            h_est = pos_range * 0.2
            alpha_est = np.pi/4  # 45 degrees
            
            return [x0_est, alpha_est, h_est, K_est]
    
    def invert(self, initial_params=None, method='local'):
        if initial_params is None:
            initial_params = self.estimate_initial_parameters()
            
        pos_range = self.position.max() - self.position.min()
        
        if self.model_type == "Inclined Sheet (5-param)":
            lower_bounds = [
                self.position.min() - pos_range*0.2,  # x0
                0,                                     # alpha (0 to 180 degrees)
                0.1,                                   # h (positive depth)
                -1000,                                 # K (can be negative)
                1                                      # a (positive half-length)
            ]
            upper_bounds = [
                self.position.max() + pos_range*0.2,  # x0
                np.pi,                                 # alpha
                200,                                   # h
                1000,                                  # K
                pos_range                              # a
            ]
        else:
            lower_bounds = [
                self.position.min() - pos_range*0.2,  # x0
                0,                                     # alpha
                0.1,                                   # h
                -1000                                  # K
            ]
            upper_bounds = [
                self.position.max() + pos_range*0.2,  # x0
                np.pi,                                 # alpha
                200,                                   # h
                1000                                   # K
            ]
        
        if method == 'local':
            try:
                result = least_squares(
                    self.objective_function,
                    initial_params,
                    bounds=(lower_bounds, upper_bounds),
                    max_nfev=1000
                )
                return result.x, result.success, result
            except:
                return initial_params, False, None
        
        elif method == 'global':
            bounds = list(zip(lower_bounds, upper_bounds))
            
            def global_objective(params):
                residuals = self.objective_function(params)
                return np.sum(residuals**2)
            
            try:
                result = differential_evolution(
                    global_objective, 
                    bounds, 
                    seed=42,
                    maxiter=1000,
                    popsize=15
                )
                return result.x, result.success, result
            except:
                return initial_params, False, None

# Enhanced PINN Implementation (keeping existing implementation)
class SPPINN(nn.Module):
    def __init__(self, layers=[1, 50, 50, 50, 1]):
        super(SPPINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

class PINNInverter:
    def __init__(self, position, data, model_type):
        self.position = position
        self.data = data
        self.model_type = model_type
        self.device = torch.device('cpu')  # Use CPU for Streamlit
        
        # Find minimum SP position for x0 constraint (from original PINN code)
        self.min_idx = np.argmin(data)
        self.x0_target = position[self.min_idx]  # Target x0 position
    
    def forward_model_numpy(self, params, x_inp):
        """NumPy version of PINN forward model - exactly matching the PyTorch version"""
        if self.model_type == "Inclined Sheet (5-param)":
            x0, alpha, h, K, a = params[0], params[1], params[2], params[3], params[4]
            
            b = a * np.cos(alpha)
            d = a * np.sin(alpha)
            
            r1_sq = ((x_inp - x0) + b)**2 + (h - d)**2
            r2_sq = ((x_inp - x0) - b)**2 + (h + d)**2
            
            ratio = np.clip(r1_sq / r2_sq, 1e-10, None)
            return K * np.log(ratio)
        else:
            # Exactly match the PyTorch PINN forward model
            x0, alpha, h, K = params[0], params[1], params[2], params[3]
            
            # Alpha is stored in degrees, convert to radians (exactly like PyTorch version)
            alpha_rad = alpha * np.pi / 180.0
            
            numerator = (x_inp - x0) * np.cos(alpha_rad) - h * np.sin(alpha_rad)
            denominator = ((x_inp - x0)**2 + h**2)**(3/2)  # Original uses **(3/2)
            
            return K * numerator / denominator
        
    def forward_model_torch(self, x, params):
        """Analytical SP model matching the original working PINN code"""
        if self.model_type == "Inclined Sheet (5-param)":
            x0, alpha, h, K, a = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
            
            b = a * torch.cos(alpha)
            d = a * torch.sin(alpha)
            
            r1_sq = ((x - x0) + b)**2 + (h - d)**2
            r2_sq = ((x - x0) - b)**2 + (h + d)**2
            
            ratio = torch.clamp(r1_sq / r2_sq, min=1e-10)
            return K * torch.log(ratio)
        else:
            # Use the original working PINN forward model - exactly matching the original code
            x0, alpha, h, K = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
            
            # Alpha is stored in degrees, convert to radians (exactly like original)
            alpha_rad = alpha * torch.pi / 180.0
            
            numerator = (x - x0) * torch.cos(alpha_rad) - h * torch.sin(alpha_rad)
            denominator = ((x - x0)**2 + h**2)**(3/2)  # Original uses **(3/2)
            
            return K * numerator / denominator
    
    def train(self, epochs=1000, lambda_physics=1.0, lambda_x0=10.0):
        """Enhanced training exactly matching original PINN code"""
        # Create model and move to device
        self.model = SPPINN()
        self.model.to(self.device)
        
        # Convert data to tensors
        self.x_data = torch.tensor(self.position.reshape(-1, 1), dtype=torch.float32).to(self.device)
        self.sp_data = torch.tensor(self.data.reshape(-1, 1), dtype=torch.float32).to(self.device)
        
        # Physics collocation points
        x_min, x_max = self.position.min(), self.position.max()
        x_range = x_max - x_min
        x_physics = np.linspace(x_min - 0.5*x_range, x_max + 0.5*x_range, 200)
        self.x_physics = torch.tensor(x_physics.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(self.device)
        
        # Learnable source parameters - exactly like original (alpha in degrees)
        self.source_params = nn.Parameter(torch.tensor([
            [self.x0_target],  # x0 - initialize at min SP position  
            [45.0],            # alpha - dip angle in DEGREES (like original)
            [10.0],            # h - depth
            [50.0]             # K - strength
        ], dtype=torch.float32)).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.source_params], 
            lr=0.001
        )
        
        # Training loop - exactly matching original
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Data loss
            sp_pred_data = self.model(self.x_data)
            data_loss = torch.mean((sp_pred_data - self.sp_data)**2)
            
            # Physics loss
            sp_pred_physics = self.model(self.x_physics)
            sp_analytical = self.sp_forward_model(self.x_physics, self.source_params.T)
            physics_loss = torch.mean((sp_pred_physics - sp_analytical.unsqueeze(1))**2)
            
            # x0 constraint loss - keep x0 close to minimum SP position
            x0_current = self.source_params[0, 0]
            x0_target_tensor = torch.tensor(self.x0_target, dtype=torch.float32).to(self.device)
            x0_constraint_loss = (x0_current - x0_target_tensor)**2
            
            total_loss = data_loss + lambda_physics * physics_loss + lambda_x0 * x0_constraint_loss
            total_loss.backward()
            self.optimizer.step()
            
            # CRITICAL FIX: Apply constraints EVERY epoch to prevent parameter drift
            with torch.no_grad():
                # Constrain alpha to 0-180 degrees (MUST be enforced every step)
                self.source_params[1, 0].data = torch.clamp(self.source_params[1, 0].data, 0.0, 180.0)
                
                # Constrain h to be positive depth
                self.source_params[2, 0].data = torch.clamp(self.source_params[2, 0].data, 0.1, 200.0)
                
                # Constrain x0 within reasonable bounds
                pos_range = self.position.max() - self.position.min()
                x0_min = self.position.min() - pos_range*0.5
                x0_max = self.position.max() + pos_range*0.5
                self.source_params[0, 0].data = torch.clamp(self.source_params[0, 0].data, x0_min, x0_max)
        
        # Return learned parameters
        with torch.no_grad():
            final_params = self.source_params.cpu().numpy().flatten()
            
        return final_params, True
    
    def sp_forward_model(self, x, params):
        """Analytical SP model for dipping sheet - exactly like original"""
        x0, alpha, h, K = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        alpha_rad = alpha * torch.pi / 180.0
        
        numerator = (x - x0) * torch.cos(alpha_rad) - h * torch.sin(alpha_rad)
        denominator = ((x - x0)**2 + h**2)**(3/2)
        
        return K * numerator / denominator
    
    def predict(self, x_test):
        """Make predictions using trained neural network - exactly like original"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32).to(self.device)
            sp_pred = self.model(x_tensor)
            return sp_pred.cpu().numpy().flatten()

# Statistics calculation
def calculate_statistics(observed, predicted):
    """Calculate fit statistics"""
    residuals = predicted - observed
    rms_error = np.sqrt(np.mean(residuals**2))
    r_squared = 1 - np.sum(residuals**2) / np.sum((observed - np.mean(observed))**2)
    max_residual = np.max(np.abs(residuals))
    
    return {
        'rms_error': rms_error,
        'r_squared': r_squared,
        'max_residual': max_residual
    }

# Header and citation
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Geophysical Self-Potential Inversion Platform</h1>
        <p style="margin-bottom: 0;">Quantitative Self-Potential Analysis with Multi-Method Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="citation-box">
        <h4>📖 Citation</h4>
        <p><strong>Copyright:</strong> Peter Adetokunbo</p>
        <p><strong>Cite as:</strong> Peter Adetokunbo*, Oluseun Adetola Sanuade, Michael Ayuk Ayuk, Ayodeji Adekunle Eluyemi, 
         and Farag Mewafy (2025). Physics-Informed Neural Networks for 
        Simultaneous Multi-Anomaly Inversion of Self-Potential Data. <em>In Press</em></p>
    </div>
    """, unsafe_allow_html=True)

# Status Panel
def render_status_panel():
    data_status = "✅ Loaded" if st.session_state.get('data_loaded', False) else "❌ Not Loaded"
    windows_count = len(st.session_state.get('windows', []))
    model_status = st.session_state.get('selected_model', 'Not Selected')
    results_count = len(st.session_state.get('results', {}))
    
    st.markdown("""
    <div class="status-panel">
        <h4>📊 Project Status</h4>
        <div class="status-item"><span><strong>Data:</strong></span><span>{}</span></div>
        <div class="status-item"><span><strong>Windows:</strong></span><span>{} defined</span></div>
        <div class="status-item"><span><strong>Model:</strong></span><span>{}</span></div>
        <div class="status-item"><span><strong>Results:</strong></span><span>{} available</span></div>
    </div>
    """.format(data_status, windows_count, model_status, results_count), unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'windows' not in st.session_state:
        st.session_state.windows = []
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'Not Selected'
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'Data Management'

# Tab 1: Data Management
def render_data_management_tab():
    st.markdown('<div class="tab-header"><h3>📊 Data Management</h3><p>Upload and manage your geophysical datasets</p></div>', unsafe_allow_html=True)
    
    # Top row: Data upload (left) and Dataset overview (right)
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your geophysical data",
            type=['csv', 'txt'],
            help="File should contain two columns: position and measurement values"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                    if data.shape[1] >= 2:
                        position = data.iloc[:, 0].values
                        measurements = data.iloc[:, 1].values
                    else:
                        st.error("CSV file must have at least 2 columns")
                        return
                else:  # txt file
                    data = np.loadtxt(uploaded_file)
                    if data.shape[1] >= 2:
                        position = data[:, 0]
                        measurements = data[:, 1]
                    else:
                        st.error("TXT file must have at least 2 columns")
                        return
                        
                st.session_state.position = position
                st.session_state.measurements = measurements
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
    
    with col2:
        if st.session_state.get('data_loaded', False):
            st.subheader("📈 Dataset Overview")
            st.metric(
                label="Data Points",
                value=f"{len(st.session_state.position)}"
            )
            st.metric(
                label="Distance Range (m)", 
                value=f"{st.session_state.position.min():.1f} to {st.session_state.position.max():.1f}"
            )
            st.metric(
                label="SP Field Range (mV)",
                value=f"{st.session_state.measurements.min():.1f} to {st.session_state.measurements.max():.1f}"
            )
    
    # Middle row: Large plot visualization (full width)
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("📊 Data Visualization")
        
        # Create interactive plot with larger size
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.position, 
            y=st.session_state.measurements,
            mode='lines+markers',
            name='Self-Potential Data',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6, color='#2E86AB')
        ))
        
        # Add window overlays
        colors = ['rgba(255,165,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(255,0,0,0.3)', 'rgba(128,0,128,0.3)']
        for i, window in enumerate(st.session_state.windows):
            color = colors[i % len(colors)]
            fig.add_vrect(
                x0=window['start'], 
                x1=window['end'],
                fillcolor=color,
                layer="below",
                line_width=0,
            )
            fig.add_annotation(
                x=(window['start'] + window['end'])/2,
                y=st.session_state.measurements.max(),
                text=window['name'],
                showarrow=False,
                bgcolor=color.replace('0.3', '0.8'),
                bordercolor='white',
                borderwidth=1
            )
        
        fig.update_layout(
            title="Self-Potential Data Profile with Analysis Windows",
            xaxis_title="Distance (m)",
            yaxis_title="Self-Potential Field (mV)",
            height=600,  # Larger height
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Upload your data file to see visualization")
    
    # Bottom row: Window Management (full width)
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("🎯 Anomaly Window Definition")
        
        col1, col2, col3, col4 = st.columns([3, 3, 3, 3])
        
        with col1:
            pos_min = float(st.session_state.position.min())
            pos_max = float(st.session_state.position.max())
            window_start = st.number_input("Window Start (m)", min_value=pos_min, max_value=pos_max, value=pos_min, step=1.0)
        
        with col2:
            window_end = st.number_input("Window End (m)", min_value=pos_min, max_value=pos_max, value=pos_max, step=1.0)
        
        with col3:
            window_name = st.text_input("Window Name", value=f"Window_{len(st.session_state.windows)+1}")
        
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("➕ Add Window", type="primary"):
                if window_start < window_end:
                    mask = (st.session_state.position >= window_start) & (st.session_state.position <= window_end)
                    if np.sum(mask) > 3:
                        window_data = {
                            'name': window_name,
                            'start': window_start,
                            'end': window_end,
                            'position': st.session_state.position[mask],
                            'data': st.session_state.measurements[mask]
                        }
                        st.session_state.windows.append(window_data)
                        st.success(f"✅ Added {window_name}")
                        st.rerun()
                    else:
                        st.error("Window must contain at least 4 data points")
                else:
                    st.error("Window start must be less than window end")
            
            if st.button("🗑️ Clear All"):
                st.session_state.windows = []
                st.session_state.results = {}
                st.rerun()
    
    
    # Current Windows Summary
    if st.session_state.windows:
        st.markdown("### 📋 Defined Analysis Windows")
        windows_data = []
        for i, window in enumerate(st.session_state.windows):
            windows_data.append({
                'Window': window['name'],
                'Start (m)': f"{window['start']:.1f}",
                'End (m)': f"{window['end']:.1f}",
                'Points': len(window['position']),
                'Range (mV)': f"{window['data'].min():.1f} to {window['data'].max():.1f}",
                'Action': f"remove_{i}"
            })
        
        windows_df = pd.DataFrame(windows_data)
        
        # Display table and remove buttons
        for i, window in enumerate(st.session_state.windows):
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 1])
            with col1:
                st.write(f"**{window['name']}**")
            with col2:
                st.write(f"{window['start']:.1f} - {window['end']:.1f} m")
            with col3:
                st.write(f"{len(window['position'])} points")
            with col4:
                st.write(f"{window['data'].min():.1f} to {window['data'].max():.1f} mV")
            with col5:
                if st.button("❌", key=f"remove_{i}", help=f"Remove {window['name']}"):
                    st.session_state.windows.pop(i)
                    if window['name'] in st.session_state.results:
                        del st.session_state.results[window['name']]
                    st.rerun()

# Tab 2: Model Configuration
def render_model_configuration_tab():
    st.markdown('<div class="tab-header"><h3>🔧 Model Configuration</h3><p>Configure forward models and parameter bounds</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 7])
    
    with col1:
        st.subheader("📐 Forward Model Selection")
        
        model_types = [
            "Inclined Sheet (5-param)",
            "Sphere", 
            "Horizontal Cylinder",
            "Vertical Cylinder"
        ]
        
        selected_model = st.selectbox("Select Forward Model", model_types, key="model_selector")
        st.session_state.selected_model = selected_model
        
        # Model-specific parameter bounds
        st.markdown("### ⚙️ Parameter Constraints")
        
        if selected_model == "Inclined Sheet (5-param)":
            st.markdown("**5-Parameter Model:**")
            st.text_input("x₀ bounds (m)", value="auto ± 20%", disabled=True)
            st.text_input("α bounds (deg)", value="0 to 180", disabled=True)
            st.text_input("h bounds (m)", value="0.1 to 200", disabled=True)
            st.text_input("K bounds", value="-1000 to 1000", disabled=True)
            st.text_input("a bounds (m)", value="1 to range", disabled=True)
        else:
            st.markdown("**4-Parameter Model:**")
            st.text_input("x₀ bounds (m)", value="auto ± 20%", disabled=True)
            st.text_input("α bounds (deg)", value="0 to 180", disabled=True)
            st.text_input("h bounds (m)", value="0.1 to 200", disabled=True)
            st.text_input("K bounds", value="-1000 to 1000", disabled=True)
    
    with col2:
        st.subheader("📖 Model Theory & Equations")
        
        if selected_model == "Inclined Sheet (5-param)":
            st.markdown("""
            #### Inclined Sheet Model
            
            **Description:** Models a thin, inclined sheet-like conductor (e.g., ore vein, fault zone)
            
            **Parameters:**
            - **x₀**: Horizontal position of sheet center (m)
            - **α**: Dip angle from horizontal (degrees)
            - **h**: Depth to sheet center (m)  
            - **K**: Intensity coefficient (related to current density)
            - **a**: Half-length of sheet (m)
            
            **Forward Model:**
            ```
            SP(x) = K * ln(r₁²/r₂²)
            
            where:
            r₁² = (x - x₀ + b)² + (h - d)²
            r₂² = (x - x₀ - b)² + (h + d)²
            b = a * cos(α)
            d = a * sin(α)
            ```
            
            **Applications:**
            - Mineral exploration (ore veins)
            - Groundwater investigations
            - Geological fault mapping
            """)
            
        elif selected_model == "Sphere":
            st.markdown("""
            #### Spherical Model (q = 1.5)
            
            **Description:** Models a spherical conductor or polarized body
            
            **Parameters:**
            - **x₀**: Horizontal position of sphere center (m)
            - **α**: Polarization angle (degrees)
            - **h**: Depth to sphere center (m)
            - **K**: Intensity coefficient
            
            **Forward Model:**
            ```
            SP(x) = K * [(x-x₀)cos(α) - h*sin(α)] / [(x-x₀)² + h²]^1.5
            ```
            
            **Applications:**
            - Ore bodies (massive sulfides)
            - Groundwater wells
            - Subsurface cavities
            """)
            
        elif selected_model == "Horizontal Cylinder":
            st.markdown("""
            #### Horizontal Cylinder Model (q = 1.0)
            
            **Description:** Models a horizontal cylindrical conductor
            
            **Parameters:**
            - **x₀**: Horizontal position of cylinder axis (m)
            - **α**: Polarization angle (degrees)
            - **h**: Depth to cylinder axis (m)
            - **K**: Intensity coefficient
            
            **Forward Model:**
            ```
            SP(x) = K * [(x-x₀)cos(α) - h*sin(α)] / [(x-x₀)² + h²]^1.0
            ```
            
            **Applications:**
            - Underground pipes/utilities
            - Horizontal ore bodies
            - Groundwater flow channels
            """)
            
        else:  # Vertical Cylinder
            st.markdown("""
            #### Vertical Cylinder Model (q = 0.5)
            
            **Description:** Models a vertical cylindrical conductor
            
            **Parameters:**
            - **x₀**: Horizontal position of cylinder axis (m)
            - **α**: Polarization angle (degrees)
            - **h**: Depth to cylinder top (m)
            - **K**: Intensity coefficient
            
            **Forward Model:**
            ```
            SP(x) = K * [(x-x₀)cos(α) - h*sin(α)] / [(x-x₀)² + h²]^0.5
            ```
            
            **Applications:**
            - Vertical wells/boreholes
            - Vertical ore shoots
            - Subsurface chimneys
            """)
        
        # Model comparison
        st.markdown("### 📊 Model Comparison")
        comparison_data = {
            'Model': ['Inclined Sheet', 'Sphere', 'Horizontal Cylinder', 'Vertical Cylinder'],
            'Parameters': ['5 (x₀,α,h,K,a)', '4 (x₀,α,h,K)', '4 (x₀,α,h,K)', '4 (x₀,α,h,K)'],
            'Geometry': ['2D Sheet', '3D Sphere', '2D Horizontal', '2D Vertical'],
            'Applications': ['Veins, Faults', 'Ore Bodies', 'Pipes, Flows', 'Wells, Shoots']
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

# Tab 3: Inversion Setup
def render_inversion_setup_tab():
    st.markdown('<div class="tab-header"><h3>🚀 Inversion Setup & Execution</h3><p>Configure optimization methods and run multi-method inversions</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or not st.session_state.windows:
        st.warning("⚠️ Please load data and define analysis windows first")
        return
    
    col1, col2 = st.columns([4, 6])
    
    with col1:
        st.subheader("🎯 Analysis Configuration")
        
        # Window selection
        window_names = [w['name'] for w in st.session_state.windows]
        selected_window_name = st.selectbox("Analysis Window", window_names)
        selected_model = st.session_state.get('selected_model', 'Sphere')
        
        st.markdown(f"**Selected Model:** {selected_model}")
        
        # Method selection with cards
        st.markdown("### 🔬 Optimization Methods")
        
        method_descriptions = {
            'Least Squares': {
                'desc': 'Fast local optimization using gradient-based methods',
                'pros': 'Fast convergence, computationally efficient',
                'cons': 'May get trapped in local minima'
            },
            'Global Optimization': {
                'desc': 'Robust global search using differential evolution',
                'pros': 'Finds global minimum, handles complex landscapes',
                'cons': 'Slower convergence, more computational cost'
            },
            'Physics-Informed NN': {
                'desc': 'AI-based optimization with physics constraints',
                'pros': 'Incorporates physical laws, handles noise well',
                'cons': 'Requires training time, may need tuning'
            }
        }
        
        run_lsq = st.checkbox("📈 Least Squares", value=True)
        if run_lsq:
            with st.expander("ℹ️ Least Squares Details"):
                st.write(f"**Description:** {method_descriptions['Least Squares']['desc']}")
                st.write(f"**Advantages:** {method_descriptions['Least Squares']['pros']}")
                st.write(f"**Limitations:** {method_descriptions['Least Squares']['cons']}")
        
        run_global = st.checkbox("🌐 Global Optimization", value=True)
        if run_global:
            with st.expander("ℹ️ Global Optimization Details"):
                st.write(f"**Description:** {method_descriptions['Global Optimization']['desc']}")
                st.write(f"**Advantages:** {method_descriptions['Global Optimization']['pros']}")
                st.write(f"**Limitations:** {method_descriptions['Global Optimization']['cons']}")
        
        run_pinn = st.checkbox("🧠 Physics-Informed NN", value=True)
        if run_pinn:
            with st.expander("ℹ️ PINN Details"):
                st.write(f"**Description:** {method_descriptions['Physics-Informed NN']['desc']}")
                st.write(f"**Advantages:** {method_descriptions['Physics-Informed NN']['pros']}")
                st.write(f"**Limitations:** {method_descriptions['Physics-Informed NN']['cons']}")
    
    with col2:
        st.subheader("⚙️ Advanced Parameters")
        
        # Create tabs for different parameter categories
        param_tab1, param_tab2, param_tab3 = st.tabs(["🧠 PINN", "🔍 Optimization", "📊 General"])
        
        with param_tab1:
            st.write("**Neural Network Configuration:**")
            pinn_epochs = st.number_input("Training Epochs", min_value=100, max_value=10000, value=3000, step=100)
            col2a, col2b = st.columns(2)
            with col2a:
                lambda_physics = st.number_input("Physics Weight (λ)", min_value=0.001, max_value=10.0, value=0.01, step=0.001, format="%.3f")
            with col2b:
                lambda_x0 = st.number_input("Constraint Weight", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
        
        with param_tab2:
            st.write("**Optimization Configuration:**")
            col2a, col2b = st.columns(2)
            with col2a:
                lsq_max_iter = st.number_input("LSQ Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
                global_max_iter = st.number_input("Global Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
            with col2b:
                global_popsize = st.number_input("Global Population Size", min_value=5, max_value=50, value=15, step=5)
                tolerance = st.number_input("Convergence Tolerance", min_value=1e-8, max_value=1e-4, value=1e-6, format="%.1e")
        
        with param_tab3:
            st.write("**Analysis Options:**")
            save_intermediate = st.checkbox("Save Intermediate Results", value=False)
            verbose_output = st.checkbox("Verbose Output", value=True)
            parallel_methods = st.checkbox("Parallel Execution", value=False, help="Run methods simultaneously (experimental)")
        
        # Run button
        st.markdown("---")
        if st.button("🚀 Execute Multi-Method Inversion", type="primary", use_container_width=True):
            if not (run_lsq or run_global or run_pinn):
                st.error("❌ Please select at least one inversion method")
            else:
                execute_inversion(selected_window_name, selected_model, run_lsq, run_global, run_pinn,
                                pinn_epochs, lambda_physics, lambda_x0, lsq_max_iter, global_max_iter, global_popsize)

def execute_inversion(selected_window_name, selected_model, run_lsq, run_global, run_pinn,
                     pinn_epochs, lambda_physics, lambda_x0, lsq_max_iter, global_max_iter, global_popsize):
    """Execute the multi-method inversion with professional progress tracking"""
    
    # Get selected window data
    selected_window = next(w for w in st.session_state.windows if w['name'] == selected_window_name)
    
    # Get custom bounds if available
    custom_bounds = None
    if 'parameter_bounds' in st.session_state and selected_model in st.session_state.parameter_bounds:
        custom_bounds = st.session_state.parameter_bounds[selected_model]
    
    results = {}
    
    # Create progress container
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔄 Inversion Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        method_status = st.empty()
        
        total_methods = sum([run_lsq, run_global, run_pinn])
        current_method = 0
        
        # Display current bounds being used
        if custom_bounds:
            with st.expander("📋 Using Custom Parameter Bounds"):
                bounds_text = f"""
                **{selected_model} Bounds:**
                - x₀: {custom_bounds['x0'][0]:.1f} to {custom_bounds['x0'][1]:.1f} m
                - α: {custom_bounds['alpha'][0]:.1f} to {custom_bounds['alpha'][1]:.1f} deg
                - h: {custom_bounds['h'][0]:.1f} to {custom_bounds['h'][1]:.1f} m
                - K: {custom_bounds['k'][0]:.1f} to {custom_bounds['k'][1]:.1f}
                """
                if selected_model == "Inclined Sheet (5-param)":
                    bounds_text += f"- a: {custom_bounds['a'][0]:.1f} to {custom_bounds['a'][1]:.1f} m"
                st.code(bounds_text)
        
        # Least Squares
        if run_lsq:
            with method_status.container():
                st.info("🔄 Executing Least Squares Optimization...")
            
            try:
                start_time = time.time()
                inverter = LSQInverter(selected_window['position'], selected_window['data'], selected_model)
                invert_result = inverter.invert(method='local', custom_bounds=custom_bounds)
                if invert_result is not None and len(invert_result) == 3:
                    params, success, result_obj = invert_result
                    synthetic = inverter.forward_model(params)
                    stats = calculate_statistics(selected_window['data'], synthetic)
                    execution_time = time.time() - start_time
                else:
                    raise Exception("Inversion failed to return valid results")
                
                results['Least Squares'] = {
                    'params': params,
                    'synthetic': synthetic,
                    'stats': stats,
                    'success': success,
                    'execution_time': execution_time
                }
                
                with method_status.container():
                    st.success(f"✅ Least Squares completed in {execution_time:.2f}s")
                    
            except Exception as e:
                with method_status.container():
                    st.error(f"❌ Least Squares failed: {str(e)}")
            
            current_method += 1
            progress_bar.progress(current_method / total_methods)
        
        # Global Optimization
        if run_global:
            with method_status.container():
                st.info("🔄 Executing Global Optimization...")
            
            try:
                start_time = time.time()
                inverter = LSQInverter(selected_window['position'], selected_window['data'], selected_model)
                invert_result = inverter.invert(method='global', custom_bounds=custom_bounds)
                if invert_result is not None and len(invert_result) == 3:
                    params, success, result_obj = invert_result
                    synthetic = inverter.forward_model(params)
                    stats = calculate_statistics(selected_window['data'], synthetic)
                    execution_time = time.time() - start_time
                else:
                    raise Exception("Inversion failed to return valid results")
                
                results['Global Optimization'] = {
                    'params': params,
                    'synthetic': synthetic,
                    'stats': stats,
                    'success': success,
                    'execution_time': execution_time
                }
                
                with method_status.container():
                    st.success(f"✅ Global Optimization completed in {execution_time:.2f}s")
                    
            except Exception as e:
                with method_status.container():
                    st.error(f"❌ Global Optimization failed: {str(e)}")
            
            current_method += 1
            progress_bar.progress(current_method / total_methods)
        
        # PINN
        if run_pinn:
            with method_status.container():
                st.info("🔄 Training Physics-Informed Neural Network...")
            
            try:
                start_time = time.time()
                pinn_inverter = PINNInverter(selected_window['position'], selected_window['data'], selected_model)
                params, success = pinn_inverter.train(epochs=pinn_epochs, lambda_physics=lambda_physics, lambda_x0=lambda_x0)
                
                # Use neural network prediction
                synthetic = pinn_inverter.predict(selected_window['position'])
                stats = calculate_statistics(selected_window['data'], synthetic)
                execution_time = time.time() - start_time
                
                results['PINN'] = {
                    'params': params,
                    'synthetic': synthetic,
                    'stats': stats,
                    'success': success,
                    'execution_time': execution_time
                }
                
                with method_status.container():
                    st.success(f"✅ PINN completed in {execution_time:.2f}s")
                    
            except Exception as e:
                with method_status.container():
                    st.error(f"❌ PINN failed: {str(e)}")
            
            current_method += 1
            progress_bar.progress(current_method / total_methods)
        
        progress_bar.progress(1.0)
        with status_text.container():
            st.success("🎉 Multi-method inversion completed successfully!")
        
        # Store results
        st.session_state.results[selected_window_name] = {
            'model_type': selected_model,
            'results': results,
            'window_data': selected_window,
            'custom_bounds': custom_bounds  # Store the bounds used
        }
        
        # Clear progress after delay
        time.sleep(2)
        progress_container.empty()

# Tab 4: Results & Analysis
def render_results_analysis_tab():
    st.markdown('<div class="tab-header"><h3>📊 Results Analysis & Export</h3><p>Analyze inversion results and generate publication-quality outputs</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.results:
        st.info("📋 No results available. Please run inversions first.")
        return
    
    # Results selection
    result_windows = list(st.session_state.results.keys())
    selected_result_window = st.selectbox("📁 Select Results", result_windows)
    
    window_results = st.session_state.results[selected_result_window]
    results = window_results['results']
    model_type = window_results['model_type']
    selected_window = window_results['window_data']
    
    # Solution selector
    st.subheader("🎯 Solution Selection")
    selection_method = st.radio(
        "Selection Method:",
        ["Auto (Best RMS)", "Manual Selection", "Compare All"],
        horizontal=True
    )
    
    selected_method = None
    selected_result = None
    
    if selection_method == "Auto (Best RMS)":
        if results:
            best_method = min(results.keys(), 
                            key=lambda x: results[x]['stats']['rms_error'] if results[x]['success'] else float('inf'))
            selected_method = best_method
            selected_result = results[best_method]
            st.success(f"🏆 Auto-selected: **{best_method}** (RMS: {selected_result['stats']['rms_error']:.4f})")
    
    elif selection_method == "Manual Selection":
        if results:
            available_methods = [method for method, result in results.items() if result['success']]
            if available_methods:
                selected_method = st.selectbox("Choose method:", available_methods)
                selected_result = results[selected_method]
                st.info(f"✅ Selected: **{selected_method}**")
    
    # Results visualization
    render_results_plots(results, selected_window, model_type, selected_result_window)
    
    # Parameter tables and export
    if selected_method and selected_result:
        render_parameter_analysis(results, model_type, selected_method, selected_result)
    
    # Export tools
    render_export_tools(results, selected_window, model_type, selected_result_window, selected_method, selected_result)

def render_results_plots(results, selected_window, model_type, window_name):
    """Render comprehensive results visualization"""
    
    # Create main comparison plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Fit Comparison', 'Residual Analysis', 'Method Statistics', 'Execution Time'),
        specs=[[{"colspan": 2}, None],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.12
    )
    
    # Plot observed data
    fig.add_trace(
        go.Scatter(
            x=selected_window['position'],
            y=selected_window['data'],
            mode='markers',
            name='Observed Data',
            marker=dict(color='black', size=8, symbol='circle'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Method colors and styles
    colors = {'Least Squares': '#2E86AB', 'Global Optimization': '#A23B72', 'PINN': '#F18F01'}
    styles = {'Least Squares': 'dash', 'Global Optimization': 'dot', 'PINN': 'solid'}
    
    # Plot results for each method
    method_names = []
    rms_errors = []
    execution_times = []
    
    for method, result in results.items():
        if result['success']:
            color = colors.get(method, '#666666')
            style = styles.get(method, 'solid')
            
            # Synthetic data
            fig.add_trace(
                go.Scatter(
                    x=selected_window['position'],
                    y=result['synthetic'],
                    mode='lines',
                    name=f'{method}',
                    line=dict(color=color, width=3, dash=style),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Collect statistics
            method_names.append(method)
            rms_errors.append(result['stats']['rms_error'])
            execution_times.append(result['execution_time'])
    
    # Add statistics bars
    if method_names:
        # RMS Error comparison
        fig.add_trace(
            go.Bar(
                x=method_names,
                y=rms_errors,
                name='RMS Error',
                marker=dict(color=[colors[m] for m in method_names]),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Execution time comparison
        fig.add_trace(
            go.Bar(
                x=method_names,
                y=execution_times,
                name='Execution Time',
                marker=dict(color=[colors[m] for m in method_names]),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=700,
        title=f"Multi-Method Inversion Results: {window_name} ({model_type})",
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Self-Potential Field (mV)", row=1, col=1)
    fig.update_yaxes(title_text="RMS Error", row=2, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def render_parameter_analysis(results, model_type, selected_method, selected_result):
    """Render detailed parameter analysis"""
    
    st.subheader(f"📋 Parameter Analysis: {selected_method}")
    
    # Parameter display - all models now have 4 optimized parameters
    params = selected_result['params'].copy()
    param_names = ['x₀ (m)', 'α (deg)', 'h (m)', 'K']
    
    # Show fixed 'a' separately for inclined sheet
    if model_type == "Inclined Sheet (5-param)":
        fixed_a = st.session_state.get('fixed_a', 10.0)
        st.info(f"📏 Fixed sheet half-length: a = {fixed_a:.2f} m (optimized)")
    
    # Convert angle to degrees for display
    if selected_method != 'PINN':
        params[1] = np.degrees(params[1])
    
    # Display parameters in metric cards
    param_cols = st.columns(len(params))
    for i, (name, value) in enumerate(zip(param_names, params)):
        with param_cols[i]:
            st.markdown(f'<div class="metric-card"><h4>{value:.2f}</h4><p>{name}</p></div>', 
                       unsafe_allow_html=True)
    
    # Statistics display
    st.markdown("### 📊 Fit Statistics")
    stats_cols = st.columns(3)
    with stats_cols[0]:
        st.markdown(f'<div class="metric-card"><h4>{selected_result["stats"]["rms_error"]:.4f}</h4><p>RMS Error</p></div>', 
                   unsafe_allow_html=True)
    with stats_cols[1]:
        st.markdown(f'<div class="metric-card"><h4>{selected_result["stats"]["r_squared"]:.4f}</h4><p>R² Coefficient</p></div>', 
                   unsafe_allow_html=True)
    with stats_cols[2]:
        st.markdown(f'<div class="metric-card"><h4>{selected_result["stats"]["max_residual"]:.4f}</h4><p>Max Residual</p></div>', 
                   unsafe_allow_html=True)
    
    # Comparison table
    st.markdown("### 📋 Method Comparison Table")
    comparison_data = []
    for method, result in results.items():
        if result['success']:
            params_display = result['params'].copy()
            if method != 'PINN':
                params_display[1] = np.degrees(params_display[1])
            
            row = [method] + [f"{p:.2f}" for p in params_display]
            row.extend([
                f"{result['stats']['rms_error']:.4f}",
                f"{result['stats']['r_squared']:.4f}",
                f"{result['execution_time']:.2f}s"
            ])
            comparison_data.append(row)
    
    columns = ['Method'] + param_names + ['RMS Error', 'R²', 'Time']
    comparison_df = pd.DataFrame(comparison_data, columns=columns)
    st.dataframe(comparison_df, use_container_width=True)

def render_export_tools(results, selected_window, model_type, window_name, selected_method, selected_result):
    """Render export and publication tools"""
    
    st.subheader("📥 Export & Publication Tools")
    
    export_tab1, export_tab2, export_tab3 = st.tabs(["📊 Publication Plots", "📋 Data Export", "📄 Report Generation"])
    
    with export_tab1:
        st.markdown("### 🎨 High-Resolution Publication Plots")
        
        col1, col2 = st.columns(2)
        with col1:
            plot_width = st.number_input("Width (inches)", min_value=6, max_value=20, value=12)
            plot_height = st.number_input("Height (inches)", min_value=4, max_value=15, value=8)
            plot_dpi = st.number_input("Resolution (DPI)", min_value=150, max_value=600, value=300, step=50)
        
        with col2:
            show_grid = st.checkbox("Show Grid", value=True)
            show_legend = st.checkbox("Show Legend", value=True)
            font_size = st.number_input("Font Size", min_value=8, max_value=18, value=12)
        
        if st.button("🎨 Generate Publication Plot", type="primary"):
            generate_publication_plot(results, selected_window, model_type, window_name, 
                                    plot_width, plot_height, plot_dpi, show_grid, show_legend, font_size)
    
    with export_tab2:
        st.markdown("### 📊 Data Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Download Results Table", type="secondary"):
                # Generate CSV for download
                comparison_data = []
                param_names = ['x₀ (m)', 'α (deg)', 'h (m)', 'K']
                if model_type == "Inclined Sheet (5-param)":
                    param_names.append('a (m)')
                
                for method, result in results.items():
                    if result['success']:
                        params_display = result['params'].copy()
                        if method != 'PINN':
                            params_display[1] = np.degrees(params_display[1])
                        
                        row = [method] + [f"{p:.4f}" for p in params_display]
                        row.extend([
                            f"{result['stats']['rms_error']:.6f}",
                            f"{result['stats']['r_squared']:.6f}",
                            f"{result['stats']['max_residual']:.6f}",
                            f"{result['execution_time']:.3f}"
                        ])
                        comparison_data.append(row)
                
                columns = ['Method'] + param_names + ['RMS_Error', 'R_Squared', 'Max_Residual', 'Execution_Time_s']
                results_df = pd.DataFrame(comparison_data, columns=columns)
                
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"spinv_results_{window_name}_{model_type.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if selected_method and selected_result:
                if st.button("📄 Download Synthetic Data", type="secondary"):
                    # Generate synthetic data CSV
                    synthetic_data = pd.DataFrame({
                        'Position_m': selected_window['position'],
                        'Observed_mV': selected_window['data'],
                        'Synthetic_mV': selected_result['synthetic'],
                        'Residual_mV': selected_result['synthetic'] - selected_window['data']
                    })
                    
                    csv_buffer = io.StringIO()
                    synthetic_data.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Synthetic Data",
                        data=csv_buffer.getvalue(),
                        file_name=f"spinv_synthetic_{window_name}_{selected_method.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
    
    with export_tab3:
        st.markdown("### 📄 Comprehensive Report")
        
        if selected_method and selected_result:
            # Generate comprehensive report
            params = selected_result['params'].copy()
            param_names = ['x₀ (m)', 'α (deg)', 'h (m)', 'K']
            
            if selected_method != 'PINN':
                params[1] = np.degrees(params[1])
            
            report = f"""
SP_Inv - Geophysical Inversion Analysis Report
===============================================

Analysis Information:
--------------------
Window: {window_name}
Model Type: {model_type}
Selected Method: {selected_method}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Data Summary:
-------------
Data Points: {len(selected_window['position'])}
Distance Range: {selected_window['position'].min():.2f} to {selected_window['position'].max():.2f} m
Data Range: {selected_window['data'].min():.2f} to {selected_window['data'].max():.2f} mV

Optimized Parameters:
--------------------
"""
            for i, name in enumerate(param_names):
                report += f"{name}: {params[i]:.4f}\n"
            
            # Add fixed 'a' info for inclined sheet
            if model_type == "Inclined Sheet (5-param)":
                fixed_a = st.session_state.get('fixed_a', 10.0)
                report += f"\nFixed Parameter:\n"
                report += f"a (sheet half-length): {fixed_a:.4f} m (user-defined)\n"
            
            report += f"""
Fit Statistics:
---------------
RMS Error: {selected_result['stats']['rms_error']:.6f}
R² Coefficient: {selected_result['stats']['r_squared']:.6f}
Maximum Residual: {selected_result['stats']['max_residual']:.6f}
Execution Time: {selected_result['execution_time']:.3f} seconds

Method Comparison:
-----------------
"""
            for method, result in results.items():
                if result['success']:
                    report += f"{method}: RMS = {result['stats']['rms_error']:.4f}, R² = {result['stats']['r_squared']:.4f}\n"
            
            report += f"""
Citation:
---------
Copyright: Peter Adetokunbo
Cite as: Peter Adetokunbo*, Michael Ayuk Ayuk, Ayodeji Adekunle Eluyemi, 
Oluseun Adetola Sanuade, and Farag Mewafy (2025). Physics-Informed Neural Networks for 
Simultaneous Multi-Anomaly Inversion of Self-Potential Data. In Press

Generated by SP_Inv - Geophysical Self-Potential Inversion Platform
"""
            
            st.download_button(
                label="📄 Download Complete Report",
                data=report,
                file_name=f"SPinv_Report_{window_name}_{selected_method.replace(' ', '_')}.txt",
                mime="text/plain",
                type="primary"
            )

def generate_publication_plot(results, selected_window, model_type, window_name, 
                            plot_width, plot_height, plot_dpi, show_grid, show_legend, font_size):
    """Generate high-quality publication plot"""
    
    plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
    
    fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_height), dpi=plot_dpi)
    
    # Plot observed data
    ax.plot(selected_window['position'], selected_window['data'], 
           'ko', markersize=8, label='Observed Data', 
           markerfacecolor='white', markeredgewidth=2, markeredgecolor='black', zorder=10)
    
    # Plot synthetic data for each method
    method_colors = {'Least Squares': '#2E86AB', 'Global Optimization': '#A23B72', 'PINN': '#F18F01'}
    method_styles = {'Least Squares': '-.', 'Global Optimization': '--', 'PINN': '-'}
    
    for method, result in results.items():
        if result['success']:
            color = method_colors.get(method, '#666666')
            style = method_styles.get(method, '-')
            ax.plot(selected_window['position'], result['synthetic'],
                   color=color, linestyle=style, linewidth=4,
                   label=f'{method} (RMS: {result["stats"]["rms_error"]:.3f})', zorder=5)
    
    ax.set_xlabel('Distance (m)', fontsize=font_size+2, fontweight='bold')
    ax.set_ylabel('Self-Potential Field (mV)', fontsize=font_size+2, fontweight='bold')
    ax.set_title(f'Self-Potential Inversion Results\n{window_name} - {model_type}', 
                fontsize=font_size+3, fontweight='bold', pad=25)
    
    if show_legend:
        ax.legend(fontsize=font_size, loc='best', framealpha=0.95, shadow=True, 
                 fancybox=True, borderpad=1)
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    ax.tick_params(labelsize=font_size, width=1.5, length=6)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout(pad=3.0)
    
    st.pyplot(fig, dpi=plot_dpi)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format='png', dpi=plot_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        png_buffer.seek(0)
        
        st.download_button(
            label="📥 PNG",
            data=png_buffer.getvalue(),
            file_name=f"SPinv_{window_name}_{model_type.replace(' ', '_')}_{plot_dpi}dpi.png",
            mime="image/png"
        )
    
    with col2:
        svg_buffer = io.StringIO()
        fig.savefig(svg_buffer, format='svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        svg_buffer.seek(0)
        
        st.download_button(
            label="📥 SVG",
            data=svg_buffer.getvalue(),
            file_name=f"SPinv_{window_name}_{model_type.replace(' ', '_')}.svg",
            mime="image/svg+xml"
        )
    
    with col3:
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        pdf_buffer.seek(0)
        
        st.download_button(
            label="📥 PDF",
            data=pdf_buffer.getvalue(),
            file_name=f"SPinv_{window_name}_{model_type.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    
    plt.close(fig)

# Main Application
def main():
    # Initialize
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        tabs = ["📊 Data Management", "🔧 Model Configuration", "🚀 Inversion Setup", "📈 Results & Analysis"]
        selected_tab = st.radio("", tabs, key="nav_tabs")
        
        st.markdown("---")
        render_status_panel()
        
        st.markdown("---")
        st.markdown("### 📚 Quick Help")
        with st.expander("🔍 Workflow Guide"):
            st.markdown("""
            1. **Data Management**: Upload CSV/TXT files
            2. **Model Configuration**: Select forward model
            3. **Inversion Setup**: Choose methods and run
            4. **Results & Analysis**: Compare and export
            """)
        
        with st.expander("📖 Model Guide"):
            st.markdown("""
            - **Inclined Sheet**: Ore veins, faults
            - **Sphere**: Ore bodies, cavities  
            - **Horizontal Cylinder**: Pipes, flows
            - **Vertical Cylinder**: Wells, shoots
            """)
    
    # Main content based on selected tab
    if selected_tab == "📊 Data Management":
        render_data_management_tab()
    elif selected_tab == "🔧 Model Configuration":
        render_model_configuration_tab()
    elif selected_tab == "🚀 Inversion Setup":
        render_inversion_setup_tab()
    elif selected_tab == "📈 Results & Analysis":
        render_results_analysis_tab()

if __name__ == "__main__":
    main()