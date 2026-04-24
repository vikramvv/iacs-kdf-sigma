import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO

def read_kdf_file(file_content):
    """
    Read KDF file content and return a pandas DataFrame
    """
    try:
        # Read the content as a string
        content = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
        
        # Split into lines and find data start
        lines = content.strip().split('\n')
        
        # Find the header line (contains column names)
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Points' in line and 'Time' in line:
                header_idx = i
                break
        
        # Extract header and data
        header = lines[header_idx].split('\t')
        data_lines = lines[header_idx + 1:]
        
        # Parse data
        data = []
        for line in data_lines:
            if line.strip():  # Skip empty lines
                values = line.split('\t')
                if len(values) == len(header):
                    data.append(values)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        return df
    
    except Exception as e:
        st.error(f"Error reading KDF file: {str(e)}")
        return None

def calculate_resistivity(resistance_ohm, length_mm, width_mm, height_mm):
    """
    Calculate resistivity from resistance and dimensions
    """
    # Convert dimensions from mm to m
    length_m = length_mm / 1000
    width_m = width_mm / 1000
    height_m = height_mm / 1000
    
    # Calculate cross-sectional area
    area_m2 = width_m * height_m
    
    # Calculate resistivity (ρ = R * A / L)
    resistivity = resistance_ohm * area_m2 / length_m
    
    return resistivity

def resistivity_to_iacs(resistivity_ohm_m):
    """
    Convert resistivity to %IACS
    """
    # Standard resistivity of annealed copper at 20°C: 1.7241 × 10^-8 ohm⋅m
    copper_resistivity = 1.7241e-8  # ohm⋅m
    
    # %IACS = (copper_resistivity / sample_resistivity) × 100
    iacs_percent = (copper_resistivity / resistivity_ohm_m) * 100
    
    return iacs_percent

def main():
    st.set_page_config(page_title="Multi-KDF IACS Analyzer", page_icon="⚡", layout="wide")
    
    st.title("⚡ Multi-KDF %IACS Analyzer")
    st.markdown("Upload multiple KDF files to compare %IACS values across samples")
    
    # File upload - multiple files
    uploaded_files = st.file_uploader("Choose KDF files", 
                                     type=['kdf', 'txt', 'csv'], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Initialize session state for dimensions if not exists
        if 'dimensions' not in st.session_state:
            st.session_state.dimensions = {}
        
        # Default dimensions
        default_length = 60.0
        default_width = 5.5
        default_height = 10.0
        
        st.subheader("📏 Set Dimensions for Each File")
        st.markdown("*Default: Length=60mm, Width=5.5mm, Height=10mm*")
        
        # Create dimension inputs for each file
        dimensions = {}
        results_data = []
        
        # Create columns for dimension inputs
        cols = st.columns(min(3, len(uploaded_files)))
        
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            col_idx = i % 3
            
            with cols[col_idx]:
                st.write(f"**{filename}**")
                
                # Get or set default dimensions
                if filename not in st.session_state.dimensions:
                    st.session_state.dimensions[filename] = {
                        'length': default_length,
                        'width': default_width, 
                        'height': default_height
                    }
                
                length = st.number_input(f"Length (mm)", 
                                       min_value=0.001,
                                       value=st.session_state.dimensions[filename]['length'],
                                       step=0.01,
                                       format="%.2f",
                                       key=f"length_{filename}")
                
                width = st.number_input(f"Width (mm)",
                                      min_value=0.001, 
                                      value=st.session_state.dimensions[filename]['width'],
                                      step=0.01,
                                      format="%.2f",
                                      key=f"width_{filename}")
                
                height = st.number_input(f"Height (mm)",
                                       min_value=0.001,
                                       value=st.session_state.dimensions[filename]['height'], 
                                       step=0.01,
                                       format="%.2f",
                                       key=f"height_{filename}")
                
                # Store dimensions
                dimensions[filename] = {
                    'length': length,
                    'width': width,
                    'height': height
                }
                
                st.session_state.dimensions[filename] = dimensions[filename]
                
                # Show cross-sectional area
                area = width * height
                st.caption(f"Area: {area:.2f} mm²")
        
        st.divider()
        
        # Process each file
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_content = uploaded_file.read()
            df = read_kdf_file(file_content)
            
            if df is not None and 'Resistance' in df.columns:
                # Get dimensions for this file
                dims = dimensions[filename]
                
                # Calculate resistance statistics
                resistance_mean = df['Resistance'].mean()
                resistance_std = df['Resistance'].std()
                
                # Calculate resistivity
                resistivity_mean = calculate_resistivity(
                    resistance_mean, dims['length'], dims['width'], dims['height']
                )
                resistivity_std = calculate_resistivity(
                    resistance_std, dims['length'], dims['width'], dims['height']
                )
                
                # Calculate %IACS
                iacs_mean = resistivity_to_iacs(resistivity_mean)
                iacs_std = (resistivity_std / resistivity_mean) * iacs_mean  # Propagate error
                
                # Store results
                results_data.append({
                    'Filename': filename.replace('.kdf', '').replace('.txt', '').replace('.csv', ''),
                    'Length (mm)': dims['length'],
                    'Width (mm)': dims['width'],
                    'Height (mm)': dims['height'],
                    'Area (mm²)': dims['width'] * dims['height'],
                    'Resistance Mean (Ω)': resistance_mean,
                    'Resistance Std (Ω)': resistance_std,
                    'Resistivity (Ω⋅m)': resistivity_mean,
                    'IACS Mean (%)': iacs_mean,
                    'IACS Std (%)': iacs_std,
                    'Measurements': len(df)
                })
            else:
                st.error(f"Could not process {filename} - no resistance data found")
        
        if results_data:
            # Create results DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Main plot: Filename vs %IACS with error bars
            st.subheader("📊 %IACS Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results_df['Filename'],
                y=results_df['IACS Mean (%)'],
                error_y=dict(
                    type='data',
                    array=results_df['IACS Std (%)'],
                    visible=True,
                    color='red',
                    thickness=2,
                    width=5
                ),
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color='blue',
                    line=dict(width=2, color='darkblue')
                ),
                line=dict(width=2, color='blue'),
                name='%IACS ± Std Dev'
            ))
            
            fig.update_layout(
                title='%IACS Comparison Across Samples',
                xaxis_title='Sample',
                yaxis_title='%IACS',
                height=500,
                xaxis=dict(tickangle=45),
                showlegend=True,
                template='plotly_white'
            )
            
            # Add horizontal line at 100% IACS (pure copper reference)
            fig.add_hline(y=100, 
                         line_dash="dash", 
                         line_color="orange",
                         annotation_text="100% IACS (Pure Cu)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average %IACS", f"{results_df['IACS Mean (%)'].mean():.2f}")
            with col2:
                st.metric("Std Dev %IACS", f"{results_df['IACS Mean (%)'].std():.2f}")
            with col3:
                st.metric("Max %IACS", f"{results_df['IACS Mean (%)'].max():.2f}")
            with col4:
                st.metric("Min %IACS", f"{results_df['IACS Mean (%)'].min():.2f}")
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            # Format the display dataframe
            display_df = results_df.copy()
            display_df['Resistance Mean (Ω)'] = display_df['Resistance Mean (Ω)'].apply(lambda x: f"{x:.3e}")
            display_df['Resistance Std (Ω)'] = display_df['Resistance Std (Ω)'].apply(lambda x: f"{x:.3e}")
            display_df['Resistivity (Ω⋅m)'] = display_df['Resistivity (Ω⋅m)'].apply(lambda x: f"{x:.3e}")
            display_df['IACS Mean (%)'] = display_df['IACS Mean (%)'].apply(lambda x: f"{x:.2f}")
            display_df['IACS Std (%)'] = display_df['IACS Std (%)'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download results
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Results (CSV)",
                data=csv_results,
                file_name="multi_kdf_iacs_results.csv",
                mime="text/csv"
            )
            
            # Quick copy for %IACS values only
            iacs_summary = results_df[['Filename', 'IACS Mean (%)', 'IACS Std (%)']].copy()
            csv_iacs = iacs_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download %IACS Summary (CSV)",
                data=csv_iacs,
                file_name="iacs_summary.csv",
                mime="text/csv"
            )
            
        else:
            st.error("No valid KDF files with resistance data were processed.")
    
    else:
        st.info("Please upload KDF files to get started")
        
        # Show example
        st.subheader("Expected Workflow")
        st.markdown("""
        1. Upload multiple KDF files
        2. Set length, width, height for each sample (defaults: 60mm × 5.5mm × 10mm)
        3. View %IACS comparison plot with error bars
        4. Download results for further analysis
        """)
        
        st.subheader("About %IACS")
        st.markdown("""
        **%IACS (International Annealed Copper Standard)**:
        - 100% IACS = Pure annealed copper conductivity
        - Higher values = Better conductors
        - Typical ranges: Aluminum ~61%, Silver ~106%
        """)

if __name__ == "__main__":
    main()