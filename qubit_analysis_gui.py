import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("husl")

class QubitAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qubit Error Analysis Framework")
        self.root.geometry("950x700")  # زيادة العرض قليلاً
        self.root.configure(bg='#f0f0f0')
        
        # Load and process data
        self.load_data()
        
        # Create main interface with scrollbar
        self.create_main_interface_with_scroll()
    
    def load_data(self):
        """Load and process the Excel data"""
        try:
            # Read data from Excel file
            file_path = 'IBM-data.xlsx'
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            
            # Clean data - remove empty rows
            self.df_clean = df.dropna(subset=['Qubit']).copy()
            self.df_clean['Qubit'] = self.df_clean['Qubit'].astype(int)
            
            # Extract two-qubit errors
            self.two_qubit_df = self.extract_two_qubit_errors(df)
            
            # Prepare single qubit data
            self.prepare_single_qubit_data()
            
            # Calculate all metrics
            self.calculate_all_metrics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.root.quit()
    
    def extract_two_qubit_errors(self, df):
        """Extract two-qubit gate errors from the data"""
        two_qubit_errors = []
        
        for idx, row in df.iterrows():
            # Search for CZ errors in columns M, N, O
            if pd.notna(row.get('CZ error')):
                cz_error_str = str(row['CZ error'])
                if '_' in cz_error_str and ':' in cz_error_str:
                    parts = cz_error_str.split(':')
                    qubit_pair = parts[0].strip()
                    error_value = float(parts[1].strip())
                    two_qubit_errors.append({
                        'qubit_pair': qubit_pair,
                        'cz_error': error_value,
                        'qubit1': int(qubit_pair.split('_')[0]),
                        'qubit2': int(qubit_pair.split('_')[1])
                    })
            
            # Search in Gate length column
            if pd.notna(row.get('Gate length (ns)')):
                gate_len_str = str(row['Gate length (ns)'])
                if '_' in gate_len_str and ':' in gate_len_str:
                    parts = gate_len_str.split(':')
                    qubit_pair = parts[0].strip()
                    gate_length = float(parts[1].strip())
                    
                    existing = next((item for item in two_qubit_errors 
                                   if item.get('qubit_pair') == qubit_pair), None)
                    if existing:
                        existing['gate_length'] = gate_length
                    else:
                        two_qubit_errors.append({
                            'qubit_pair': qubit_pair,
                            'gate_length': gate_length,
                            'qubit1': int(qubit_pair.split('_')[0]),
                            'qubit2': int(qubit_pair.split('_')[1])
                        })
            
            # Search in RZZ error column
            if pd.notna(row.get('RZZ error')):
                rzz_str = str(row['RZZ error'])
                if '_' in rzz_str and ':' in rzz_str:
                    parts = rzz_str.split(':')
                    qubit_pair = parts[0].strip()
                    rzz_error = float(parts[1].strip())
                    
                    existing = next((item for item in two_qubit_errors 
                                   if item.get('qubit_pair') == qubit_pair), None)
                    if existing:
                        existing['rzz_error'] = rzz_error
                    else:
                        two_qubit_errors.append({
                            'qubit_pair': qubit_pair,
                            'rzz_error': rzz_error,
                            'qubit1': int(qubit_pair.split('_')[0]),
                            'qubit2': int(qubit_pair.split('_')[1])
                        })
        
        return pd.DataFrame(two_qubit_errors)
    
    def prepare_single_qubit_data(self):
        """Prepare single qubit calibration data"""
        self.single_qubit_data = self.df_clean[[
            'Qubit', 'T1 (us)', 'T2 (us)', 
            'Readout assignment error', 'Prob meas0 prep1', 'Prob meas1 prep0',
            'ID error', 'RX error', '√x (sx) error', 'Pauli-X error',
            'MEASURE error'
        ]].copy()
        
        # Rename columns for clarity
        self.single_qubit_data.columns = [
            'qubit', 't1_us', 't2_us', 
            'readout_error', 'prob_meas0_prep1', 'prob_meas1_prep0',
            'id_error', 'rx_error', 'sx_error', 'pauli_x_error',
            'measure_error'
        ]
        
        # Convert to numeric
        for col in self.single_qubit_data.columns[1:]:
            self.single_qubit_data[col] = pd.to_numeric(
                self.single_qubit_data[col], errors='coerce'
            )
        
        # Remove missing values
        self.single_qubit_data = self.single_qubit_data.dropna(subset=['readout_error'])
        
        # Apply classifications
        self.apply_classifications()
    
    def apply_classifications(self):
        """Apply different classification methods"""
        
        # 1. Spatial Classification
        def spatial_classification(qubit_num):
            if qubit_num < 50:
                return 'A_Low_Range'
            elif qubit_num < 100:
                return 'B_Mid_Range'
            elif qubit_num < 130:
                return 'C_High_Range'
            else:
                return 'D_Very_High_Range'
        
        self.single_qubit_data['spatial_category'] = self.single_qubit_data['qubit'].apply(
            spatial_classification
        )
        
        # 2. Error-Rate Classification
        def error_rate_classification(error_rate):
            if error_rate < 0.005:
                return 'Low_Error'
            elif error_rate < 0.01:
                return 'Medium_Error'
            elif error_rate < 0.02:
                return 'High_Error'
            else:
                return 'Very_High_Error'
        
        self.single_qubit_data['error_category'] = self.single_qubit_data['readout_error'].apply(
            error_rate_classification
        )
        
        # 3. Coherence-Based Classification
        def coherence_classification(t1, t2):
            t1_t2_avg = (t1 + t2) / 2
            if t1_t2_avg > 300:
                return 'High_Coherence'
            elif t1_t2_avg > 200:
                return 'Medium_Coherence'
            elif t1_t2_avg > 100:
                return 'Low_Coherence'
            else:
                return 'Very_Low_Coherence'
        
        self.single_qubit_data['coherence_category'] = self.single_qubit_data.apply(
            lambda row: coherence_classification(row['t1_us'], row['t2_us']), axis=1
        )
    
    def calculate_all_metrics(self):
        """Calculate all metrics for the analysis"""
        
        # Basic totals
        self.E_total = self.single_qubit_data['readout_error'].sum()
        self.N = len(self.single_qubit_data)
        self.avg_error = self.E_total / self.N
        
        # Calculate metrics for each classification type
        self.spatial_metrics = self.calculate_category_metrics('spatial_category')
        self.error_metrics = self.calculate_category_metrics('error_category')
        self.coherence_metrics = self.calculate_category_metrics('coherence_category')
        
        # Build covariance matrix
        self.cov_matrix, self.qubit_list = self.build_covariance_matrix()
        
        # Calculate correlated total
        self.E_total_corr = self.calculate_correlated_total()
        
        # Calculate decoder weights
        self.calculate_decoder_weights()
    
    def calculate_category_metrics(self, category_col, error_col='readout_error'):
        """Calculate quantitative metrics for classification"""
        categories = self.single_qubit_data[category_col].unique()
        results = []
        
        for cat in categories:
            cat_df = self.single_qubit_data[self.single_qubit_data[category_col] == cat]
            E_A = cat_df[error_col].sum()
            cat_size = len(cat_df)
            R_A = E_A / self.E_total
            avg_per_qubit_cat = E_A / cat_size if cat_size > 0 else 0
            avg_per_qubit_total = self.E_total / self.N
            D_A = avg_per_qubit_cat / avg_per_qubit_total if avg_per_qubit_total > 0 else 0
            
            results.append({
                'category': cat,
                'size': cat_size,
                'E_A': E_A,
                'R_A': R_A,
                'D_A': D_A,
                'avg_error': avg_per_qubit_cat
            })
        
        return pd.DataFrame(results).sort_values('R_A', ascending=False)
    
    def build_covariance_matrix(self):
        """Build approximate covariance matrix from two-qubit errors"""
        n_qubits = len(self.single_qubit_data)
        qubit_list = self.single_qubit_data['qubit'].tolist()
        qubit_to_idx = {q: i for i, q in enumerate(qubit_list)}
        
        cov_matrix = np.zeros((n_qubits, n_qubits))
        
        # Fill diagonal with readout error variances
        for i, q in enumerate(qubit_list):
            qubit_error = self.single_qubit_data[
                self.single_qubit_data['qubit'] == q
            ]['readout_error'].values[0]
            cov_matrix[i, i] = qubit_error ** 2
        
        # Fill off-diagonal elements based on two-qubit errors
        if not self.two_qubit_df.empty:
            for _, row in self.two_qubit_df.iterrows():
                if 'cz_error' in row and pd.notna(row['cz_error']):
                    q1 = row['qubit1']
                    q2 = row['qubit2']
                    
                    if q1 in qubit_to_idx and q2 in qubit_to_idx:
                        i = qubit_to_idx[q1]
                        j = qubit_to_idx[q2]
                        cov_matrix[i, j] = row['cz_error']
                        cov_matrix[j, i] = row['cz_error']
        
        return cov_matrix, qubit_list
    
    def calculate_correlated_total(self):
        """Calculate total error with correlations"""
        sum_e = self.single_qubit_data['readout_error'].sum()
        sum_cov_offdiag = np.sum(self.cov_matrix) - np.trace(self.cov_matrix)
        return sum_e + sum_cov_offdiag
    
    def calculate_decoder_weights(self, rho=0.3):
        """Calculate different decoder weight models"""
        
        # 1. Uniform Model
        self.uniform_weights = -np.log(np.ones(self.N) * self.avg_error)
        
        # 2. Individual Model
        individual_errors = self.single_qubit_data.sort_values('qubit')['readout_error'].values
        self.individual_weights = -np.log(individual_errors + 1e-10)
        
        # 3. Category-Correlation Model
        qubit_to_idx = {q: i for i, q in enumerate(self.qubit_list)}
        self.category_correlation_weights = np.zeros(self.N)
        
        for idx, row in self.single_qubit_data.sort_values('qubit').iterrows():
            q = row['qubit']
            e_i = row['readout_error']
            i = qubit_to_idx[q]
            
            correlation_sum = 0
            for j in range(self.N):
                if i != j and self.cov_matrix[i, j] > 0:
                    correlation_sum += rho * self.cov_matrix[i, j]
            
            self.category_correlation_weights[i] = -np.log(e_i + correlation_sum + 1e-10)
        
        # Calculate logical error rates
        self.calculate_logical_error_rates()
    
    def calculate_logical_error_rates(self, distance=3, p_th=0.01, C=1.0):
        """Estimate logical error rates for different models"""
        
        def estimate(weights):
            p_phys = np.exp(-weights)
            avg_p_phys = np.mean(p_phys)
            return C * (avg_p_phys / p_th) ** ((distance + 1) / 2)
        
        self.p_L_uniform = estimate(self.uniform_weights)
        self.p_L_individual = estimate(self.individual_weights)
        self.p_L_category = estimate(self.category_correlation_weights)
    
    def save_all_results(self):
        """Save all plots as images with descriptive names"""
        try:
            # Create results directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"analysis_results_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            saved_files = []
            
            # 1. Save readout error distribution
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(self.single_qubit_data['readout_error'], bins=30, 
                     edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_xlabel('Readout Error')
            ax1.set_ylabel('Number of Qubits')
            ax1.set_title('Readout Error Distribution')
            ax1.axvline(self.avg_error, color='red', linestyle='--', 
                        label=f'Mean: {self.avg_error:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            filename = f"{save_dir}/01_readout_error_distribution.png"
            fig1.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            saved_files.append(filename)
            
            # 2. Save spatial metrics bar charts
            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
            
            # R_A by spatial category
            self.spatial_metrics.plot(x='category', y='R_A', kind='bar', ax=ax2a,
                                      color='lightgreen', edgecolor='black', legend=False)
            ax2a.set_xlabel('Spatial Category')
            ax2a.set_ylabel('R_A (Relative Contribution)')
            ax2a.set_title('R_A by Spatial Category')
            ax2a.axhline(y=0.25, color='red', linestyle='--', label='Expected')
            ax2a.legend()
            ax2a.grid(True, alpha=0.3)
            
            # D_A by spatial category
            self.spatial_metrics.plot(x='category', y='D_A', kind='bar', ax=ax2b,
                                      color='lightcoral', edgecolor='black', legend=False)
            ax2b.set_xlabel('Spatial Category')
            ax2b.set_ylabel('D_A (Disproportionality Factor)')
            ax2b.set_title('D_A by Spatial Category')
            ax2b.axhline(y=1, color='red', linestyle='--', label='D_A = 1')
            ax2b.legend()
            ax2b.grid(True, alpha=0.3)
            
            fig2.tight_layout()
            filename = f"{save_dir}/02_spatial_category_metrics.png"
            fig2.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            saved_files.append(filename)
            
            # 3. Save error category metrics
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
            
            self.error_metrics.plot(x='category', y='R_A', kind='bar', ax=ax3a,
                                    color='lightgreen', edgecolor='black', legend=False)
            ax3a.set_xlabel('Error Category')
            ax3a.set_ylabel('R_A')
            ax3a.set_title('R_A by Error Category')
            ax3a.tick_params(axis='x', rotation=45)
            ax3a.grid(True, alpha=0.3)
            
            self.error_metrics.plot(x='category', y='D_A', kind='bar', ax=ax3b,
                                    color='lightcoral', edgecolor='black', legend=False)
            ax3b.set_xlabel('Error Category')
            ax3b.set_ylabel('D_A')
            ax3b.set_title('D_A by Error Category')
            ax3b.axhline(y=1, color='red', linestyle='--')
            ax3b.tick_params(axis='x', rotation=45)
            ax3b.grid(True, alpha=0.3)
            
            fig3.tight_layout()
            filename = f"{save_dir}/03_error_category_metrics.png"
            fig3.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            saved_files.append(filename)
            
            # 4. Save coherence category metrics
            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
            
            self.coherence_metrics.plot(x='category', y='R_A', kind='bar', ax=ax4a,
                                        color='lightgreen', edgecolor='black', legend=False)
            ax4a.set_xlabel('Coherence Category')
            ax4a.set_ylabel('R_A')
            ax4a.set_title('R_A by Coherence Category')
            ax4a.tick_params(axis='x', rotation=45)
            ax4a.grid(True, alpha=0.3)
            
            self.coherence_metrics.plot(x='category', y='D_A', kind='bar', ax=ax4b,
                                        color='lightcoral', edgecolor='black', legend=False)
            ax4b.set_xlabel('Coherence Category')
            ax4b.set_ylabel('D_A')
            ax4b.set_title('D_A by Coherence Category')
            ax4b.axhline(y=1, color='red', linestyle='--')
            ax4b.tick_params(axis='x', rotation=45)
            ax4b.grid(True, alpha=0.3)
            
            fig4.tight_layout()
            filename = f"{save_dir}/04_coherence_category_metrics.png"
            fig4.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig4)
            saved_files.append(filename)
            
            # 5. Save decoder weights comparison
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sample_qubits = range(min(30, self.N))
            ax5.plot(sample_qubits, self.uniform_weights[sample_qubits], 'o-', 
                    label='Uniform', alpha=0.7, markersize=4, linewidth=2)
            ax5.plot(sample_qubits, self.individual_weights[sample_qubits], 's-', 
                    label='Individual', alpha=0.7, markersize=4, linewidth=2)
            ax5.plot(sample_qubits, self.category_correlation_weights[sample_qubits], '^-', 
                    label='Category-Correlation', alpha=0.7, markersize=4, linewidth=2)
            ax5.set_xlabel('Qubit Index')
            ax5.set_ylabel('Weight')
            ax5.set_title('Decoder Weight Comparison (First 30 Qubits)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            filename = f"{save_dir}/05_decoder_weights_comparison.png"
            fig5.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig5)
            saved_files.append(filename)
            
            # 6. Save correlation matrix heatmap
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            small_cov = self.cov_matrix[:min(30, self.N), :min(30, self.N)]
            im = ax6.imshow(small_cov, cmap='RdBu_r', aspect='auto', vmin=-0.01, vmax=0.01)
            ax6.set_xlabel('Qubit Index')
            ax6.set_ylabel('Qubit Index')
            ax6.set_title('Correlation Matrix (First 30 Qubits)')
            plt.colorbar(im, ax=ax6)
            filename = f"{save_dir}/06_correlation_matrix.png"
            fig6.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig6)
            saved_files.append(filename)
            
            # 7. Save logical error rates comparison
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            models = ['Uniform', 'Individual', 'Category-Correlation']
            p_L_values = [self.p_L_uniform, self.p_L_individual, self.p_L_category]
            colors = ['skyblue', 'lightgreen', 'lightcoral']
            bars = ax7.bar(models, p_L_values, color=colors, edgecolor='black')
            ax7.set_xlabel('Decoder Model')
            ax7.set_ylabel('Logical Error Rate (p_L)')
            ax7.set_title('Logical Error Rate Comparison')
            ax7.set_yscale('log')
            for bar, val in zip(bars, p_L_values):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.6f}', ha='center', va='bottom', fontsize=10)
            ax7.grid(True, alpha=0.3, axis='y')
            filename = f"{save_dir}/07_logical_error_rates.png"
            fig7.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig7)
            saved_files.append(filename)
            
            # 8. Save summary statistics as text file
            with open(f"{save_dir}/08_summary_statistics.txt", 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("QUANTUM PROCESSOR ERROR ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Qubits: {self.N}\n")
                f.write(f"Total Error Budget (E_total): {self.E_total:.6f}\n")
                f.write(f"Average Error per Qubit: {self.avg_error:.6f}\n")
                f.write(f"Correlated Total (E_total_corr): {self.E_total_corr:.6f}\n")
                f.write(f"Two-Qubit Gates: {len(self.two_qubit_df) if not self.two_qubit_df.empty else 0}\n\n")
                
                f.write("Logical Error Rates:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Uniform Model: {self.p_L_uniform:.6f}\n")
                f.write(f"Individual Model: {self.p_L_individual:.6f}\n")
                f.write(f"Category-Correlation Model: {self.p_L_category:.6f}\n\n")
                
                f.write("Most Impactful Categories:\n")
                f.write("-" * 40 + "\n")
                top_spatial = self.spatial_metrics.sort_values('R_A', ascending=False).iloc[0]
                f.write(f"Spatial: {top_spatial['category']} - R_A = {top_spatial['R_A']:.4f}, D_A = {top_spatial['D_A']:.4f}\n")
                top_error = self.error_metrics.sort_values('R_A', ascending=False).iloc[0]
                f.write(f"Error: {top_error['category']} - R_A = {top_error['R_A']:.4f}, D_A = {top_error['D_A']:.4f}\n")
                top_coherence = self.coherence_metrics.sort_values('R_A', ascending=False).iloc[0]
                f.write(f"Coherence: {top_coherence['category']} - R_A = {top_coherence['R_A']:.4f}, D_A = {top_coherence['D_A']:.4f}\n")
            
            saved_files.append(f"{save_dir}/08_summary_statistics.txt")
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"✅ All results saved successfully!\n\n"
                f"📁 Folder: {save_dir}\n"
                f"📊 Files saved: {len(saved_files)}\n\n"
                f"Files saved:\n" + "\n".join([f"  • {os.path.basename(f)}" for f in saved_files])
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def create_main_interface_with_scroll(self):
        """Create the main GUI interface with scrollbar"""
        
        # Create a canvas and scrollbar
        canvas = tk.Canvas(self.root, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        # Configure the canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel for scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Now create all widgets inside scrollable_frame
        self.create_interface_widgets(scrollable_frame)
    
    def create_interface_widgets(self, parent):
        """Create all interface widgets inside the scrollable frame"""
        
        # Title
        title_label = tk.Label(
            parent, 
            text="🔬 Qubit Error Analysis Framework",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            parent,
            text="Category-Based Error Budgeting for Quantum Processors",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#34495e'
        )
        subtitle_label.pack(pady=10)
        
        # Data info frame
        info_frame = tk.Frame(parent, bg='#ffffff', relief=tk.RAISED, bd=2)
        info_frame.pack(pady=20, padx=50, fill=tk.X)
        
        tk.Label(
            info_frame,
            text=f"📊 Data Summary",
            font=("Arial", 14, "bold"),
            bg='#ffffff',
            fg='#2c3e50'
        ).pack(pady=10)
        
        stats_text = f"""
        Total Qubits: {self.N}
        Total Error Budget (E_total): {self.E_total:.6f}
        Average Error per Qubit: {self.avg_error:.6f}
        Two-Qubit Gates: {len(self.two_qubit_df) if not self.two_qubit_df.empty else 0}
        """
        
        tk.Label(
            info_frame,
            text=stats_text,
            font=("Arial", 12),
            bg='#ffffff',
            fg='#34495e',
            justify=tk.LEFT
        ).pack(pady=10, padx=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(parent, bg='#f0f0f0')
        buttons_frame.pack(pady=20)
        
        # Button style
        button_style = {
            'font': ("Arial", 12, "bold"),
            'width': 35,  # زيادة العرض قليلاً
            'height': 2,
            'bd': 3,
            'relief': tk.RAISED
        }
        
        # Step 7.1 Button
        btn_step1 = tk.Button(
            buttons_frame,
            text="📋 Step 7.1: Data Collection & Classification",
            bg='#3498db',
            fg='white',
            command=self.show_step1,
            **button_style
        )
        btn_step1.pack(pady=5)
        
        # Step 7.2 Button
        btn_step2 = tk.Button(
            buttons_frame,
            text="📊 Step 7.2: Category Budget Analysis",
            bg='#2ecc71',
            fg='white',
            command=self.show_step2,
            **button_style
        )
        btn_step2.pack(pady=5)
        
        # Correlation Extension Button
        btn_corr = tk.Button(
            buttons_frame,
            text="🔄 Correlation-Aware Extension",
            bg='#f39c12',
            fg='white',
            command=self.show_correlation,
            **button_style
        )
        btn_corr.pack(pady=5)
        
        # Step 7.3 Button
        btn_step3 = tk.Button(
            buttons_frame,
            text="🎯 Step 7.3: Decoder Integration",
            bg='#9b59b6',
            fg='white',
            command=self.show_step3,
            **button_style
        )
        btn_step3.pack(pady=5)
        
        # Visualization Button
        btn_viz = tk.Button(
            buttons_frame,
            text="📈 Visualization Dashboard",
            bg='#e74c3c',
            fg='white',
            command=self.show_visualization,
            **button_style
        )
        btn_viz.pack(pady=5)
        
        # Summary Button
        btn_summary = tk.Button(
            buttons_frame,
            text="📑 Summary Report",
            bg='#1abc9c',
            fg='white',
            command=self.show_summary,
            **button_style
        )
        btn_summary.pack(pady=5)
        
        # Save Results Button
        btn_save = tk.Button(
            buttons_frame,
            text="💾 Save All Results",
            bg='#8e44ad',
            fg='white',
            command=self.save_all_results,
            **button_style
        )
        btn_save.pack(pady=5)
        
        # Exit Button
        btn_exit = tk.Button(
            buttons_frame,
            text="❌ Exit",
            bg='#95a5a6',
            fg='white',
            command=self.root.quit,
            **button_style
        )
        btn_exit.pack(pady=20)
        
        # Add a footer with instructions
        footer_label = tk.Label(
            parent,
            text="👆 Use mouse wheel to scroll | Click any button to view results",
            font=("Arial", 10, "italic"),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        footer_label.pack(pady=10)
    
    def create_result_window(self, title):
        """Create a new window for displaying results"""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("800x600")
        window.configure(bg='#ffffff')
        
        # Title
        tk.Label(
            window,
            text=title,
            font=("Arial", 18, "bold"),
            bg='#ffffff',
            fg='#2c3e50'
        ).pack(pady=15)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        return window, notebook
    
    def show_step1(self):
        """Show Step 7.1 results"""
        window, notebook = self.create_result_window("Step 7.1: Data Collection & Classification")
        
        # Tab 1: Basic Data Info
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Basic Data")
        
        text_widget = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, "=" * 80 + "\n")
        text_widget.insert(tk.END, "STEP 7.1: DATA COLLECTION & CLASSIFICATION\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        
        text_widget.insert(tk.END, f"Total valid qubits: {self.N}\n\n")
        
        text_widget.insert(tk.END, "Readout Error Statistics:\n")
        text_widget.insert(tk.END, "-" * 40 + "\n")
        stats = self.single_qubit_data['readout_error'].describe()
        for stat, value in stats.items():
            text_widget.insert(tk.END, f"{stat}: {value:.6f}\n")
        
        # Tab 2: Category Distributions
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Category Distributions")
        
        text_widget2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=80, height=30)
        text_widget2.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget2.insert(tk.END, "CATEGORY DISTRIBUTIONS\n")
        text_widget2.insert(tk.END, "=" * 80 + "\n\n")
        
        text_widget2.insert(tk.END, "1. Spatial Categories:\n")
        text_widget2.insert(tk.END, "-" * 40 + "\n")
        for cat, count in self.single_qubit_data['spatial_category'].value_counts().items():
            text_widget2.insert(tk.END, f"{cat}: {count} qubits\n")
        
        text_widget2.insert(tk.END, "\n2. Error Categories:\n")
        text_widget2.insert(tk.END, "-" * 40 + "\n")
        for cat, count in self.single_qubit_data['error_category'].value_counts().items():
            text_widget2.insert(tk.END, f"{cat}: {count} qubits\n")
        
        text_widget2.insert(tk.END, "\n3. Coherence Categories:\n")
        text_widget2.insert(tk.END, "-" * 40 + "\n")
        for cat, count in self.single_qubit_data['coherence_category'].value_counts().items():
            text_widget2.insert(tk.END, f"{cat}: {count} qubits\n")
    
    def show_step2(self):
        """Show Step 7.2 results"""
        window, notebook = self.create_result_window("Step 7.2: Category Budget Analysis")
        
        # Tab 1: Spatial Metrics
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Spatial Metrics")
        
        text_widget = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, "SPATIAL CLASSIFICATION METRICS\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        text_widget.insert(tk.END, self.spatial_metrics.to_string(index=False))
        text_widget.insert(tk.END, "\n\nInterpretation: D_A > 1 means category contributes more than its fair share per qubit")
        
        # Tab 2: Error Metrics
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Error Metrics")
        
        text_widget2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=80, height=30)
        text_widget2.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget2.insert(tk.END, "ERROR RATE CLASSIFICATION METRICS\n")
        text_widget2.insert(tk.END, "=" * 80 + "\n\n")
        text_widget2.insert(tk.END, self.error_metrics.to_string(index=False))
        
        # Tab 3: Coherence Metrics
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="Coherence Metrics")
        
        text_widget3 = scrolledtext.ScrolledText(tab3, wrap=tk.WORD, width=80, height=30)
        text_widget3.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget3.insert(tk.END, "COHERENCE CLASSIFICATION METRICS\n")
        text_widget3.insert(tk.END, "=" * 80 + "\n\n")
        text_widget3.insert(tk.END, self.coherence_metrics.to_string(index=False))
    
    def show_correlation(self):
        """Show Correlation-Aware Extension results"""
        window, notebook = self.create_result_window("Correlation-Aware Extension")
        
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Correlation Results")
        
        text_widget = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, "CORRELATION-AWARE EXTENSION\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        
        text_widget.insert(tk.END, f"Covariance matrix shape: {self.cov_matrix.shape}\n")
        text_widget.insert(tk.END, f"Sum of off-diagonal elements: {np.sum(self.cov_matrix) - np.trace(self.cov_matrix):.6f}\n\n")
        
        text_widget.insert(tk.END, f"E_total (without correlations): {self.E_total:.6f}\n")
        text_widget.insert(tk.END, f"E_total_corr (with correlations): {self.E_total_corr:.6f}\n")
        text_widget.insert(tk.END, f"Difference: {self.E_total_corr - self.E_total:.6f}\n")
        
        # Show top correlations
        if not self.two_qubit_df.empty and 'cz_error' in self.two_qubit_df.columns:
            text_widget.insert(tk.END, "\nTop 10 Two-Qubit Correlations:\n")
            text_widget.insert(tk.END, "-" * 40 + "\n")
            top_correlations = self.two_qubit_df.nlargest(10, 'cz_error')[['qubit_pair', 'cz_error']]
            text_widget.insert(tk.END, top_correlations.to_string(index=False))
    
    def show_step3(self):
        """Show Step 7.3 results"""
        window, notebook = self.create_result_window("Step 7.3: Decoder Integration")
        
        # Tab 1: Weight Models
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Weight Models")
        
        text_widget = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, "DECODER WEIGHT MODELS\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        
        # Weight comparison for first 10 qubits
        text_widget.insert(tk.END, "Weight Comparison (First 10 Qubits):\n")
        text_widget.insert(tk.END, "-" * 60 + "\n")
        comparison_df = pd.DataFrame({
            'qubit': self.qubit_list[:10],
            'uniform': self.uniform_weights[:10],
            'individual': self.individual_weights[:10],
            'category_corr': self.category_correlation_weights[:10]
        })
        text_widget.insert(tk.END, comparison_df.to_string(index=False))
        
        text_widget.insert(tk.END, "\n\nWeight Statistics:\n")
        text_widget.insert(tk.END, "-" * 40 + "\n")
        text_widget.insert(tk.END, f"Uniform model - Mean: {np.mean(self.uniform_weights):.4f}, Std: {np.std(self.uniform_weights):.4f}\n")
        text_widget.insert(tk.END, f"Individual model - Mean: {np.mean(self.individual_weights):.4f}, Std: {np.std(self.individual_weights):.4f}\n")
        text_widget.insert(tk.END, f"Category-Correlation model - Mean: {np.mean(self.category_correlation_weights):.4f}, Std: {np.std(self.category_correlation_weights):.4f}\n")
        
        # Tab 2: Logical Error Rates
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Logical Error Rates")
        
        text_widget2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=80, height=30)
        text_widget2.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget2.insert(tk.END, "LOGICAL ERROR RATE ESTIMATION\n")
        text_widget2.insert(tk.END, "=" * 80 + "\n\n")
        
        text_widget2.insert(tk.END, f"Uniform Model: {self.p_L_uniform:.6f}\n")
        text_widget2.insert(tk.END, f"Individual Model: {self.p_L_individual:.6f}\n")
        text_widget2.insert(tk.END, f"Category-Correlation Model: {self.p_L_category:.6f}\n")
        
        # Calculate improvement
        improvement_vs_uniform = (self.p_L_uniform - self.p_L_category) / self.p_L_uniform * 100
        improvement_vs_individual = (self.p_L_individual - self.p_L_category) / self.p_L_individual * 100
        
        text_widget2.insert(tk.END, f"\nImprovement Analysis:\n")
        text_widget2.insert(tk.END, f"Category-Correlation vs Uniform: {improvement_vs_uniform:.2f}% better\n")
        text_widget2.insert(tk.END, f"Category-Correlation vs Individual: {improvement_vs_individual:.2f}% better\n")
    
    def show_visualization(self):
        """Show visualization dashboard"""
        window = tk.Toplevel(self.root)
        window.title("Visualization Dashboard")
        window.geometry("1200x800")
        window.configure(bg='#ffffff')
        
        # Create figure with subplots
        fig = plt.Figure(figsize=(14, 10))
        
        # 1. Readout error distribution
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.hist(self.single_qubit_data['readout_error'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('Readout Error')
        ax1.set_ylabel('Number of Qubits')
        ax1.set_title('Readout Error Distribution')
        ax1.axvline(self.avg_error, color='red', linestyle='--', label=f'Mean: {self.avg_error:.4f}')
        ax1.legend()
        
        # 2. R_A by spatial category
        ax2 = fig.add_subplot(2, 3, 2)
        self.spatial_metrics.plot(x='category', y='R_A', kind='bar', ax=ax2, 
                                  color='lightgreen', edgecolor='black', legend=False)
        ax2.set_xlabel('Spatial Category')
        ax2.set_ylabel('R_A (Relative Contribution)')
        ax2.set_title('R_A by Spatial Category')
        ax2.axhline(y=0.25, color='red', linestyle='--', label='Expected')
        ax2.legend()
        
        # 3. D_A by spatial category
        ax3 = fig.add_subplot(2, 3, 3)
        self.spatial_metrics.plot(x='category', y='D_A', kind='bar', ax=ax3,
                                  color='lightcoral', edgecolor='black', legend=False)
        ax3.set_xlabel('Spatial Category')
        ax3.set_ylabel('D_A (Disproportionality Factor)')
        ax3.set_title('D_A by Spatial Category')
        ax3.axhline(y=1, color='red', linestyle='--', label='D_A = 1')
        ax3.legend()
        
        # 4. Weight comparison
        ax4 = fig.add_subplot(2, 3, 4)
        sample_qubits = range(min(20, self.N))
        ax4.plot(sample_qubits, self.uniform_weights[sample_qubits], 'o-', 
                label='Uniform', alpha=0.7, markersize=4)
        ax4.plot(sample_qubits, self.individual_weights[sample_qubits], 's-', 
                label='Individual', alpha=0.7, markersize=4)
        ax4.plot(sample_qubits, self.category_correlation_weights[sample_qubits], '^-', 
                label='Category-Correlation', alpha=0.7, markersize=4)
        ax4.set_xlabel('Qubit Index')
        ax4.set_ylabel('Weight')
        ax4.set_title('Decoder Weight Comparison (First 20 Qubits)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Covariance matrix heatmap (subset)
        ax5 = fig.add_subplot(2, 3, 5)
        small_cov = self.cov_matrix[:20, :20]
        im = ax5.imshow(small_cov, cmap='RdBu_r', aspect='auto', vmin=-0.01, vmax=0.01)
        ax5.set_xlabel('Qubit Index')
        ax5.set_ylabel('Qubit Index')
        ax5.set_title('Correlation Matrix (First 20 Qubits)')
        plt.colorbar(im, ax=ax5)
        
        # 6. Logical error rates comparison
        ax6 = fig.add_subplot(2, 3, 6)
        models = ['Uniform', 'Individual', 'Category-Correlation']
        p_L_values = [self.p_L_uniform, self.p_L_individual, self.p_L_category]
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        bars = ax6.bar(models, p_L_values, color=colors, edgecolor='black')
        ax6.set_xlabel('Decoder Model')
        ax6.set_ylabel('Logical Error Rate (p_L)')
        ax6.set_title('Logical Error Rate Comparison')
        ax6.set_yscale('log')
        for bar, val in zip(bars, p_L_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.6f}', ha='center', va='bottom', fontsize=9)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def show_summary(self):
        """Show summary report"""
        window, notebook = self.create_result_window("Summary Report")
        
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Executive Summary")
        
        text_widget = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, "=" * 80 + "\n")
        text_widget.insert(tk.END, "QUANTUM PROCESSOR ERROR ANALYSIS SUMMARY\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        
        text_widget.insert(tk.END, "✓ Step 7.1: Data Collection & Classification\n")
        text_widget.insert(tk.END, f"  - Total qubits analyzed: {self.N}\n")
        text_widget.insert(tk.END, f"  - Total error budget (E_total): {self.E_total:.6f}\n")
        text_widget.insert(tk.END, f"  - Average error per qubit: {self.avg_error:.6f}\n")
        text_widget.insert(tk.END, "  - Three classification methods applied\n\n")
        
        text_widget.insert(tk.END, "✓ Step 7.2: Category Budget Analysis\n")
        text_widget.insert(tk.END, "  - Calculated R_A (relative contribution) for each category\n")
        text_widget.insert(tk.END, "  - Calculated D_A (disproportionality factor) for each category\n")
        text_widget.insert(tk.END, f"  - Correlation-aware E_total_corr: {self.E_total_corr:.6f}\n\n")
        
        text_widget.insert(tk.END, "✓ Step 7.3: Decoder Integration\n")
        text_widget.insert(tk.END, "  - Built 3 decoder weight models\n")
        text_widget.insert(tk.END, f"  - Logical error rates:\n")
        text_widget.insert(tk.END, f"    * Uniform: {self.p_L_uniform:.6f}\n")
        text_widget.insert(tk.END, f"    * Individual: {self.p_L_individual:.6f}\n")
        text_widget.insert(tk.END, f"    * Category-Correlation: {self.p_L_category:.6f}\n\n")
        
        # Most impactful categories
        text_widget.insert(tk.END, "🎯 Most Impactful Categories:\n")
        text_widget.insert(tk.END, "-" * 40 + "\n")
        
        top_spatial = self.spatial_metrics.sort_values('R_A', ascending=False).iloc[0]
        text_widget.insert(tk.END, f"Spatial: {top_spatial['category']} - R_A = {top_spatial['R_A']:.4f}, D_A = {top_spatial['D_A']:.4f}\n")
        
        top_error = self.error_metrics.sort_values('R_A', ascending=False).iloc[0]
        text_widget.insert(tk.END, f"Error: {top_error['category']} - R_A = {top_error['R_A']:.4f}, D_A = {top_error['D_A']:.4f}\n")
        
        top_coherence = self.coherence_metrics.sort_values('R_A', ascending=False).iloc[0]
        text_widget.insert(tk.END, f"Coherence: {top_coherence['category']} - R_A = {top_coherence['R_A']:.4f}, D_A = {top_coherence['D_A']:.4f}\n")

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = QubitAnalysisGUI(root)
    root.mainloop()