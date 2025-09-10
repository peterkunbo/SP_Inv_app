# SP_Inv - Geophysical Self-Potential Inversion

A professional web-based application for quantitative self-potential data analysis using multi-method optimization approaches.

## Overview

SP_Inv is an advanced geophysical inversion platform that combines traditional optimization methods with modern physics-informed neural networks (PINNs) for self-potential data interpretation. The application provides a user-friendly interface for analyzing subsurface geological structures from self-potential measurements.

## Features

### Forward Models
- **Inclined Sheet (5-parameter)**: For ore veins, fault zones, and geological contacts
- **Sphere**: For ore bodies, cavities, and localized sources
- **Horizontal Cylinder**: For pipes, horizontal ore bodies, and flow channels
- **Vertical Cylinder**: For wells, vertical ore shoots, and subsurface chimneys

### Optimization Methods
- **Least Squares**: Fast local optimization with gradient-based methods
- **Global Optimization**: Robust differential evolution for global minima
- **Physics-Informed Neural Networks (PINN)**: AI-based optimization with physics constraints

### Key Capabilities
- Interactive data visualization with anomaly window selection
- Multi-method comparison and automatic solution selection
- Publication-quality plot generation (PNG, SVG, PDF)
- Comprehensive statistical analysis and model fitting
- Professional reporting and data export

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the application files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run sp_inv_app.py
   ```
4. **Access the web interface** at `http://localhost:8501`

## Data Format Requirements

Your input data should be a **two-column format**:
- **Column 1**: Distance/Position (in meters)
- **Column 2**: Self-potential measurements (in millivolts)

**Supported file formats:**
- CSV files (`.csv`)
- Text files (`.txt`) with space or tab separation

**Example data format:**
```
0.0    -15.2
1.0    -18.7
2.0    -22.1
3.0    -25.8
...
```

## How to Use

### Step 1: Data Management
1. **Upload Data**: Use the file uploader to load your CSV or TXT file
2. **Review Dataset**: Check the dataset overview metrics (points, distance range, data range)
3. **Visualize Data**: Examine your data in the interactive plot
4. **Define Windows**: Select specific anomaly regions for analysis by setting start/end positions

### Step 2: Model Configuration
1. **Select Forward Model**: Choose the geological model that best represents your target:
   - Use **Inclined Sheet** for dipping geological structures
   - Use **Sphere** for compact 3D bodies
   - Use **Cylinders** for elongated structures
2. **Review Model Theory**: Check the equations and applications for your selected model
3. **Parameter Constraints**: Review the automatic parameter bounds

### Step 3: Inversion Setup & Execution
1. **Choose Analysis Window**: Select which defined window to analyze
2. **Select Methods**: Choose optimization methods:
   - **Least Squares**: For quick local optimization
   - **Global Optimization**: For robust global search
   - **PINN**: For physics-constrained AI optimization
3. **Configure Parameters**: Adjust advanced settings if needed:
   - PINN epochs and weights
   - Optimization iterations and tolerance
4. **Execute Inversion**: Run the multi-method analysis

### Step 4: Results Analysis & Export
1. **Solution Selection**: 
   - **Auto**: Automatically select best RMS error solution
   - **Manual**: Choose based on your criteria
   - **Compare All**: Review all method results
2. **Analyze Results**: Review fitted parameters, statistics, and model comparison
3. **Generate Plots**: Create publication-quality figures with custom settings
4. **Export Data**: Download results as CSV, synthetic data, or comprehensive reports

## Model Parameters

### Inclined Sheet (5-parameter)
- **x₀**: Horizontal position of sheet center (m)
- **α**: Dip angle from horizontal (degrees)
- **h**: Depth to sheet center (m)
- **K**: Intensity coefficient
- **a**: Half-length of sheet (m) - *optimized but excluded from results tables*

### Other Models (4-parameter)
- **x₀**: Horizontal position of center (m)
- **α**: Polarization angle (degrees)
- **h**: Depth to center (m)
- **K**: Intensity coefficient

## Output Files

### Results Export
- **CSV Tables**: Parameter comparison across methods
- **Synthetic Data**: Model predictions and residuals
- **High-Resolution Plots**: PNG (300+ DPI), SVG, PDF formats
- **Comprehensive Reports**: Complete analysis documentation

### Statistics Provided
- **RMS Error**: Root mean square error
- **R² Coefficient**: Coefficient of determination
- **Maximum Residual**: Largest absolute residual
- **Execution Time**: Method performance timing

## Best Practices

### Data Preparation
- Ensure uniform data spacing when possible
- Remove obvious noise or outliers before analysis
- Use sufficient data points (minimum 20-30 points per anomaly)

### Model Selection
- Choose models based on expected geological targets
- Consider the geological context and available prior information
- Use multiple models for comparison when target geometry is uncertain

### Method Selection
- Use **Least Squares** for quick initial analysis
- Apply **Global Optimization** for robust final results
- Employ **PINN** for noisy data or complex scenarios
- Compare multiple methods for validation

### Window Definition
- Define windows to isolate individual anomalies
- Include sufficient background data on both sides
- Avoid windows with multiple overlapping anomalies

## Troubleshooting

### Common Issues
- **"CSV file must have at least 2 columns"**: Check your data format
- **"Window must contain at least 4 data points"**: Increase window size
- **Inversion failures**: Try different initial parameters or method combinations
- **Poor fits**: Consider different forward models or check data quality

### Performance Tips
- Use smaller PINN epoch counts for initial testing
- Reduce global optimization population size for faster results
- Process one window at a time for large datasets

## Citation

**Copyright:** Peter Adetokunbo

**Cite as:** Peter Adetokunbo*, Oluseun Adetola Sanuade, Michael Ayuk Ayuk, Ayodeji Adekunle Eluyemi and Farag Mewafy (2025). Physics-Informed Neural Networks for Simultaneous Multi-Anomaly Inversion of Self-Potential Data. *In Press*

## Technical Requirements

- **CPU**: Modern multi-core processor recommended
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Network**: Internet connection required for initial package downloads

## Support

For technical issues or questions about the application:
- Check data format requirements
- Review model selection guidelines
- Verify parameter bounds are reasonable
- Try different optimization methods for comparison

## Version Information

**Current Version**: 1.0  
**Last Updated**: 2025  
**Compatibility**: Python 3.8+, Streamlit 1.28+
