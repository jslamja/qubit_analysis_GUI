"""
save_analysis_to_excel.py
ملف منفصل لحفظ جميع نتائج التحليل في ملف Excel منظم
يتم الحفظ تلقائياً في مجلد Results داخل المشروع مع ترقيم حسب وقت سحب البيانات
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class SaveResultsToExcel:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Save Analysis Results to Excel")
        self.root.geometry("650x550")
        self.root.configure(bg='#f0f0f0')
        
        # Record data pull time (when the program starts)
        self.data_pull_time = datetime.now()
        
        # Load and process data
        self.load_data()
        
        # Create interface
        self.create_interface()
    
    def load_data(self):
        """Load and process the Excel data (same as original)"""
        try:
            # Read data from Excel file
            file_path = 'IBM-data.xlsx'
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File not found: {file_path}\nPlease make sure IBM-data.xlsx is in the same folder.")
                self.root.quit()
                return
            
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
            
            # Prepare all results for Excel
            self.prepare_excel_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.root.quit()
    
    def extract_two_qubit_errors(self, df):
        """Extract two-qubit gate errors from the data"""
        two_qubit_errors = []
        
        for idx, row in df.iterrows():
            # Search for CZ errors
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
                'Category': cat,
                'Number_of_Qubits': cat_size,
                'E_A (Total Error)': E_A,
                'R_A (Relative Contribution)': R_A,
                'D_A (Disproportionality)': D_A,
                'Average_Error_per_Qubit': avg_per_qubit_cat
            })
        
        return pd.DataFrame(results).sort_values('R_A (Relative Contribution)', ascending=False)
    
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
    
    def prepare_excel_data(self):
        """Prepare all results in a structured format for Excel"""
        
        # 1. Summary Statistics
        self.summary_stats = pd.DataFrame({
            'Metric': [
                'Total Qubits (N)',
                'Total Error Budget (E_total)',
                'Average Error per Qubit',
                'Correlated Total (E_total_corr)',
                'Number of Two-Qubit Gates',
                'Data Pull Time',
                'Analysis Creation Time'
            ],
            'Value': [
                self.N,
                f"{self.E_total:.6f}",
                f"{self.avg_error:.6f}",
                f"{self.E_total_corr:.6f}",
                len(self.two_qubit_df) if not self.two_qubit_df.empty else 0,
                self.data_pull_time.strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ],
            'Description': [
                'Number of qubits analyzed',
                'Sum of readout errors for all qubits',
                'E_total / N',
                'Total error including correlations',
                'Number of two-qubit gates in the device',
                'When the original data was pulled',
                'When this analysis was created'
            ]
        })
        
        # 2. Readout Error Statistics
        error_stats = self.single_qubit_data['readout_error'].describe()
        self.readout_stats = pd.DataFrame({
            'Statistic': error_stats.index,
            'Value': error_stats.values,
            'Description': [
                'Number of qubits',
                'Mean readout error',
                'Standard deviation',
                'Minimum error',
                '25th percentile',
                'Median (50th percentile)',
                '75th percentile',
                'Maximum error'
            ]
        })
        
        # 3. Category Distributions
        spatial_dist = self.single_qubit_data['spatial_category'].value_counts().reset_index()
        spatial_dist.columns = ['Category', 'Count']
        spatial_dist['Type'] = 'Spatial'
        
        error_dist = self.single_qubit_data['error_category'].value_counts().reset_index()
        error_dist.columns = ['Category', 'Count']
        error_dist['Type'] = 'Error Rate'
        
        coherence_dist = self.single_qubit_data['coherence_category'].value_counts().reset_index()
        coherence_dist.columns = ['Category', 'Count']
        coherence_dist['Type'] = 'Coherence'
        
        self.category_distributions = pd.concat([spatial_dist, error_dist, coherence_dist], ignore_index=True)
        
        # 4. Classification Metrics
        self.spatial_metrics_display = self.spatial_metrics.copy()
        self.spatial_metrics_display.insert(0, 'Classification_Type', 'Spatial')
        
        self.error_metrics_display = self.error_metrics.copy()
        self.error_metrics_display.insert(0, 'Classification_Type', 'Error Rate')
        
        self.coherence_metrics_display = self.coherence_metrics.copy()
        self.coherence_metrics_display.insert(0, 'Classification_Type', 'Coherence')
        
        self.all_metrics = pd.concat([
            self.spatial_metrics_display,
            self.error_metrics_display,
            self.coherence_metrics_display
        ], ignore_index=True)
        
        # 5. Top Two-Qubit Correlations
        if not self.two_qubit_df.empty and 'cz_error' in self.two_qubit_df.columns:
            self.top_correlations = self.two_qubit_df.nlargest(20, 'cz_error')[
                ['qubit_pair', 'cz_error', 'gate_length', 'rzz_error']
            ].copy()
            self.top_correlations.columns = ['Qubit_Pair', 'CZ_Error', 'Gate_Length_ns', 'RZZ_Error']
        else:
            self.top_correlations = pd.DataFrame({
                'Qubit_Pair': ['No data'],
                'CZ_Error': [np.nan],
                'Gate_Length_ns': [np.nan],
                'RZZ_Error': [np.nan]
            })
        
        # 6. Decoder Weights (sample of first 30 qubits)
        sample_size = min(30, self.N)
        self.decoder_weights = pd.DataFrame({
            'Qubit_Index': list(range(sample_size)),
            'Qubit_ID': self.qubit_list[:sample_size],
            'Readout_Error': self.single_qubit_data.sort_values('qubit')['readout_error'].values[:sample_size],
            'Uniform_Weight': self.uniform_weights[:sample_size],
            'Individual_Weight': self.individual_weights[:sample_size],
            'Category_Correlation_Weight': self.category_correlation_weights[:sample_size]
        })
        
        # 7. Logical Error Rates
        self.logical_error_rates = pd.DataFrame({
            'Decoder_Model': ['Uniform', 'Individual', 'Category-Correlation'],
            'Logical_Error_Rate (p_L)': [
                f"{self.p_L_uniform:.6f}",
                f"{self.p_L_individual:.6f}",
                f"{self.p_L_category:.6f}"
            ],
            'Description': [
                'Using average error for all qubits',
                'Using individual qubit errors',
                'Using categories and correlations'
            ]
        })
        
        # Calculate improvements
        impr_vs_uniform = (self.p_L_uniform - self.p_L_category) / self.p_L_uniform * 100
        impr_vs_individual = (self.p_L_individual - self.p_L_category) / self.p_L_individual * 100
        
        self.improvement_analysis = pd.DataFrame({
            'Comparison': [
                'Category-Correlation vs Uniform',
                'Category-Correlation vs Individual'
            ],
            'Improvement_%': [f"{impr_vs_uniform:.2f}%", f"{impr_vs_individual:.2f}%"],
            'Interpretation': [
                'Better' if impr_vs_uniform > 0 else 'Worse',
                'Better' if impr_vs_individual > 0 else 'Worse'
            ]
        })
        
        # 8. Most Impactful Categories
        top_spatial = self.spatial_metrics.sort_values('R_A (Relative Contribution)', ascending=False).iloc[0]
        top_error = self.error_metrics.sort_values('R_A (Relative Contribution)', ascending=False).iloc[0]
        top_coherence = self.coherence_metrics.sort_values('R_A (Relative Contribution)', ascending=False).iloc[0]
        
        self.impactful_categories = pd.DataFrame({
            'Classification_Type': ['Spatial', 'Error Rate', 'Coherence'],
            'Category': [
                top_spatial['Category'],
                top_error['Category'],
                top_coherence['Category']
            ],
            'R_A': [
                f"{top_spatial['R_A (Relative Contribution)']:.4f}",
                f"{top_error['R_A (Relative Contribution)']:.4f}",
                f"{top_coherence['R_A (Relative Contribution)']:.4f}"
            ],
            'D_A': [
                f"{top_spatial['D_A (Disproportionality)']:.4f}",
                f"{top_error['D_A (Disproportionality)']:.4f}",
                f"{top_coherence['D_A (Disproportionality)']:.4f}"
            ],
            'Interpretation': [
                '>1 means worse than average' if top_spatial['D_A (Disproportionality)'] > 1 else '<1 means better than average',
                '>1 means worse than average' if top_error['D_A (Disproportionality)'] > 1 else '<1 means better than average',
                '>1 means worse than average' if top_coherence['D_A (Disproportionality)'] > 1 else '<1 means better than average'
            ]
        })
        
        # 9. Readme / Explanation Sheet
        self.readme = pd.DataFrame({
            'Section': [
                'Summary Statistics',
                'Readout Error Statistics',
                'Category Distributions',
                'Classification Metrics',
                'Top Two-Qubit Correlations',
                'Decoder Weights Sample',
                'Logical Error Rates',
                'Improvement Analysis',
                'Most Impactful Categories',
                'Full Qubit Data'
            ],
            'Description': [
                'Overall metrics about the quantum processor',
                'Statistical distribution of readout errors',
                'Number of qubits in each category',
                'R_A and D_A metrics for all categories',
                'Top 20 two-qubit gates with highest error rates',
                'Decoder weights for first 30 qubits (all models)',
                'Estimated logical error rates for each decoder model',
                'Percentage improvement using category-correlation model',
                'Categories that contribute most to total error budget',
                'Complete dataset for all qubits'
            ],
            'Sheet_Name': [
                'Summary',
                'Error_Statistics',
                'Category_Distributions',
                'All_Metrics',
                'Top_Correlations',
                'Decoder_Weights',
                'Logical_Error_Rates',
                'Improvement',
                'Impactful_Categories',
                'Full_Qubit_Data'
            ]
        })
    
    def save_to_excel(self):
        """Save all results to an Excel file in the Results folder"""
        try:
            # Create Results folder if it doesn't exist
            results_folder = "Results"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                print(f"Created folder: {results_folder}")
            
            # Generate filename with timestamp (using data pull time)
            timestamp = self.data_pull_time.strftime("%Y%m%d_%H%M%S")
            filename = f"qubit_analysis_results_{timestamp}.xlsx"
            file_path = os.path.join(results_folder, filename)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                
                # 1. Readme / Instructions sheet
                self.readme.to_excel(writer, sheet_name='Readme', index=False)
                
                # 2. Summary Statistics
                self.summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                
                # 3. Readout Error Statistics
                self.readout_stats.to_excel(writer, sheet_name='Error_Statistics', index=False)
                
                # 4. Category Distributions
                self.category_distributions.to_excel(writer, sheet_name='Category_Distributions', index=False)
                
                # 5. All Classification Metrics
                self.all_metrics.to_excel(writer, sheet_name='All_Metrics', index=False)
                
                # 6. Top Correlations
                self.top_correlations.to_excel(writer, sheet_name='Top_Correlations', index=False)
                
                # 7. Decoder Weights Sample
                self.decoder_weights.to_excel(writer, sheet_name='Decoder_Weights', index=False)
                
                # 8. Logical Error Rates
                self.logical_error_rates.to_excel(writer, sheet_name='Logical_Error_Rates', index=False)
                
                # 9. Improvement Analysis
                self.improvement_analysis.to_excel(writer, sheet_name='Improvement', index=False)
                
                # 10. Most Impactful Categories
                self.impactful_categories.to_excel(writer, sheet_name='Impactful_Categories', index=False)
                
                # 11. Full Qubit Data
                self.single_qubit_data.to_excel(writer, sheet_name='Full_Qubit_Data', index=False)
                
                # 12. Two-Qubit Gates Full Data
                if not self.two_qubit_df.empty:
                    self.two_qubit_df.to_excel(writer, sheet_name='Two_Qubit_Gates', index=False)
                
                # Adjust column widths
                workbook = writer.book
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Success message
            messagebox.showinfo(
                "✅ Success",
                f"All results saved successfully!\n\n"
                f"📁 Folder: {results_folder}\n"
                f"📄 File: {filename}\n"
                f"📍 Full path: {os.path.abspath(file_path)}\n\n"
                f"⏰ Data pull time: {self.data_pull_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📊 Sheets created: 12\n\n"
                f"The filename includes the timestamp of when you started the program."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Excel file:\n{str(e)}")
    
    def create_interface(self):
        """Create a simple interface with a single button"""
        
        # Title
        title_label = tk.Label(
            self.root,
            text="📊 Save Analysis Results to Excel",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=30)
        
        # Description
        description = """
        This tool will save all analytical results from the quantum processor data
        into a single, well-organized Excel file.
        
        ✅ Automatic saving to 'Results' folder
        ✅ Filename includes data pull timestamp
        ✅ 12 organized sheets with descriptions
        """
        
        desc_label = tk.Label(
            self.root,
            text=description,
            font=("Arial", 11),
            bg='#f0f0f0',
            fg='#34495e',
            justify=tk.LEFT,
            wraplength=500
        )
        desc_label.pack(pady=20, padx=40)
        
        # Data pull time display
        time_frame = tk.Frame(self.root, bg='#e8f4f8', relief=tk.GROOVE, bd=2)
        time_frame.pack(pady=10, padx=50, fill=tk.X)
        
        tk.Label(
            time_frame,
            text=f"⏰ Data Pull Time: {self.data_pull_time.strftime('%Y-%m-%d %H:%M:%S')}",
            font=("Arial", 11, "bold"),
            bg='#e8f4f8',
            fg='#2980b9'
        ).pack(pady=8)
        
        tk.Label(
            time_frame,
            text="This timestamp will be used in the filename",
            font=("Arial", 9),
            bg='#e8f4f8',
            fg='#7f8c8d'
        ).pack(pady=2)
        
        # Data summary
        if hasattr(self, 'N'):
            summary_frame = tk.Frame(self.root, bg='#ffffff', relief=tk.RAISED, bd=2)
            summary_frame.pack(pady=20, padx=50, fill=tk.X)
            
            tk.Label(
                summary_frame,
                text="📋 Data Loaded Successfully",
                font=("Arial", 12, "bold"),
                bg='#ffffff',
                fg='#27ae60'
            ).pack(pady=10)
            
            stats_text = f"""
            Total Qubits: {self.N}
            Total Error Budget: {self.E_total:.6f}
            Average Error: {self.avg_error:.6f}
            Two-Qubit Gates: {len(self.two_qubit_df) if not self.two_qubit_df.empty else 0}
            """
            
            tk.Label(
                summary_frame,
                text=stats_text,
                font=("Arial", 11),
                bg='#ffffff',
                fg='#34495e',
                justify=tk.LEFT
            ).pack(pady=10, padx=20)
        
        # Save button
        save_button = tk.Button(
            self.root,
            text="💾 SAVE TO RESULTS FOLDER",
            font=("Arial", 16, "bold"),
            bg='#27ae60',
            fg='white',
            command=self.save_to_excel,
            width=25,
            height=2,
            bd=4,
            relief=tk.RAISED,
            cursor="hand2"
        )
        save_button.pack(pady=20)
        
        # Info about automatic naming
        info_label = tk.Label(
            self.root,
            text="📁 File will be saved as: Results/qubit_analysis_results_YYYYMMDD_HHMMSS.xlsx",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        info_label.pack(pady=5)
        
        # Exit button
        exit_button = tk.Button(
            self.root,
            text="❌ Exit",
            font=("Arial", 12),
            bg='#e74c3c',
            fg='white',
            command=self.root.quit,
            width=15,
            height=1,
            bd=3,
            relief=tk.RAISED
        )
        exit_button.pack(pady=15)

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = SaveResultsToExcel(root)
    root.mainloop()