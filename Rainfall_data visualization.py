"""
Problem 2: Rainfall Dashboard for Madhya Pradesh and Neighboring States
Interactive Plotly Dash Dashboard (1947-2017)

Objective: Compare rainfall patterns in Madhya Pradesh with Gujarat, 
           Rajasthan, and Maharashtra from 1947 to 2017.

Author: [Your Name]
Date: May 16, 2026

Usage: python prompt_bob_shell.py
       Then open browser at http://127.0.0.1:8050/
"""

# ============================================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Rainfall Dashboard - Data Loading and Preprocessing")
print("=" * 80)

# ============================================================================
# SECTION 2: LOAD AND EXPLORE DATA
# ============================================================================

# Load the rainfall data
df = pd.read_csv('DSM301_MSE_PROB2_DATA.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"  - Shape: {df.shape}")
print(f"  - Columns: {len(df.columns)}")
print(f"  - Year range: {df['YEAR'].min()} - {df['YEAR'].max()}")

# ============================================================================
# SECTION 3: DATA PREPROCESSING AND FILTERING
# ============================================================================

# Define state subdivisions mapping
STATE_SUBDIVISIONS = {
    'Madhya Pradesh': ['West Madhya Pradesh', 'East Madhya Pradesh'],
    'Gujarat': ['Gujarat Region', 'Saurashtra & Kutch'],
    'Rajasthan': ['West Rajasthan', 'East Rajasthan'],
    'Maharashtra': ['Konkan & Goa', 'Vidarbha', 'Marathwada']
}

# Flatten the subdivision list
all_subdivisions = [sub for subs in STATE_SUBDIVISIONS.values() for sub in subs]

# Filter data for relevant subdivisions and year range (1947-2017)
df_filtered = df[
    (df['SUBDIVISION'].isin(all_subdivisions)) & 
    (df['YEAR'] >= 1947) & 
    (df['YEAR'] <= 2017)
].copy()

# Add state column based on subdivision
def get_state(subdivision):
    """Map subdivision to its parent state"""
    for state, subs in STATE_SUBDIVISIONS.items():
        if subdivision in subs:
            return state
    return None

df_filtered['STATE'] = df_filtered['SUBDIVISION'].apply(get_state)

# Convert month columns to numeric, handling 'NA' values
month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

for col in month_cols:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

print(f"\n✓ Data filtered for target states (1947-2017)")
print(f"  - Filtered records: {len(df_filtered)}")
print(f"  - States: {', '.join(STATE_SUBDIVISIONS.keys())}")

# ============================================================================
# SECTION 4: AGGREGATE DATA BY STATE
# ============================================================================

# Aggregate rainfall data by state and year (average across subdivisions)
df_state_year = df_filtered.groupby(['STATE', 'YEAR'])[month_cols].mean().reset_index()

# Calculate annual rainfall
df_state_year['ANNUAL'] = df_state_year[month_cols].sum(axis=1)

# Melt data for easier plotting by month
df_melted = df_filtered.melt(
    id_vars=['STATE', 'SUBDIVISION', 'YEAR'],
    value_vars=month_cols,
    var_name='MONTH',
    value_name='RAINFALL'
)

# Aggregate by state, year, and month
df_state_month = df_melted.groupby(['STATE', 'YEAR', 'MONTH'])['RAINFALL'].mean().reset_index()

print(f"\n✓ Data aggregated by state, year, and month")
print(f"  - State-Year records: {len(df_state_year)}")
print(f"  - State-Month records: {len(df_state_month)}")

# ============================================================================
# SECTION 5: INITIALIZE DASH APPLICATION
# ============================================================================

# Initialize Dash app
app = Dash(__name__)

# Define color scheme for states
STATE_COLORS = {
    'Madhya Pradesh': '#1f77b4',
    'Gujarat': '#ff7f0e',
    'Rajasthan': '#2ca02c',
    'Maharashtra': '#d62728'
}

print(f"\n✓ Dash application initialized")

# ============================================================================
# SECTION 6: CREATE DASHBOARD LAYOUT
# ============================================================================

app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1(
            'Rainfall Analysis Dashboard: Madhya Pradesh and Neighboring States (1947-2017)',
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '10px',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '28px'
            }
        ),
        html.P(
            'Interactive comparison of rainfall patterns across Madhya Pradesh, Gujarat, Rajasthan, and Maharashtra',
            style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'fontSize': '16px',
                'marginBottom': '30px'
            }
        )
    ], style={
        'backgroundColor': '#ecf0f1',
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px'
    }),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label(
                'Select States:',
                style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '14px'}
            ),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': state, 'value': state} for state in STATE_COLORS.keys()],
                value=list(STATE_COLORS.keys()),
                multi=True,
                style={'width': '100%'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label(
                'Select Year Range:',
                style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '14px'}
            ),
            dcc.RangeSlider(
                id='year-slider',
                min=1947,
                max=2017,
                step=1,
                value=[1947, 2017],
                marks={year: str(year) for year in range(1947, 2018, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={
        'marginBottom': '30px',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px'
    }),
    
    # Visualization Row 1: Annual Rainfall Trend
    html.Div([
        html.Div([
            dcc.Graph(id='annual-rainfall-line')
        ], style={'width': '100%'})
    ], style={'marginBottom': '20px'}),
    
    # Visualization Row 2: Monthly Distribution and Box Plot
    html.Div([
        html.Div([
            dcc.Graph(id='monthly-rainfall-bar')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            dcc.Graph(id='seasonal-rainfall-box')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'marginBottom': '20px'}),
    
    # Visualization Row 3: Heatmap
    html.Div([
        html.Div([
            dcc.Graph(id='rainfall-heatmap')
        ], style={'width': '100%'})
    ])
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

print(f"✓ Dashboard layout created")

# ============================================================================
# SECTION 7: DEFINE CALLBACK FUNCTIONS
# ============================================================================

@app.callback(
    [
        Output('annual-rainfall-line', 'figure'),
        Output('monthly-rainfall-bar', 'figure'),
        Output('seasonal-rainfall-box', 'figure'),
        Output('rainfall-heatmap', 'figure')
    ],
    [
        Input('state-dropdown', 'value'),
        Input('year-slider', 'value')
    ]
)
def update_graphs(selected_states, year_range):
    """
    Update all dashboard graphs based on user selections
    
    Args:
        selected_states: List of selected state names
        year_range: List with [start_year, end_year]
    
    Returns:
        Tuple of 4 plotly figures
    """
    # Handle single state selection
    if not isinstance(selected_states, list):
        selected_states = [selected_states]
    
    # Filter data based on selections
    filtered_year = df_state_year[
        (df_state_year['STATE'].isin(selected_states)) &
        (df_state_year['YEAR'] >= year_range[0]) &
        (df_state_year['YEAR'] <= year_range[1])
    ]
    
    filtered_month = df_state_month[
        (df_state_month['STATE'].isin(selected_states)) &
        (df_state_month['YEAR'] >= year_range[0]) &
        (df_state_month['YEAR'] <= year_range[1])
    ]
    
    # ========================================================================
    # GRAPH 1: Annual Rainfall Trend Line Chart
    # ========================================================================
    fig1 = go.Figure()
    
    for state in selected_states:
        state_data = filtered_year[filtered_year['STATE'] == state]
        fig1.add_trace(go.Scatter(
            x=state_data['YEAR'],
            y=state_data['ANNUAL'],
            mode='lines+markers',
            name=state,
            line=dict(color=STATE_COLORS[state], width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Rainfall: %{y:.1f} mm<br>' +
                         '<extra></extra>'
        ))
    
    fig1.update_layout(
        title={
            'text': 'Annual Rainfall Trend Over Years',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title='Year',
        yaxis_title='Rainfall (mm)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # ========================================================================
    # GRAPH 2: Average Monthly Rainfall Bar Chart
    # ========================================================================
    monthly_avg = filtered_month.groupby(['STATE', 'MONTH'])['RAINFALL'].mean().reset_index()
    fig2 = go.Figure()
    
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    for state in selected_states:
        state_data = monthly_avg[monthly_avg['STATE'] == state]
        state_data['MONTH'] = pd.Categorical(
            state_data['MONTH'], 
            categories=month_order, 
            ordered=True
        )
        state_data = state_data.sort_values('MONTH')
        
        fig2.add_trace(go.Bar(
            x=state_data['MONTH'],
            y=state_data['RAINFALL'],
            name=state,
            marker_color=STATE_COLORS[state],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Avg Rainfall: %{y:.1f} mm<br>' +
                         '<extra></extra>'
        ))
    
    fig2.update_layout(
        title={
            'text': 'Average Monthly Rainfall Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title='Month',
        yaxis_title='Average Rainfall (mm)',
        barmode='group',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # ========================================================================
    # GRAPH 3: Seasonal Rainfall Box Plot
    # ========================================================================
    fig3 = go.Figure()
    
    for state in selected_states:
        state_data = filtered_year[filtered_year['STATE'] == state]
        fig3.add_trace(go.Box(
            y=state_data['ANNUAL'],
            name=state,
            marker_color=STATE_COLORS[state],
            boxmean='sd',
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Rainfall: %{y:.1f} mm<br>' +
                         '<extra></extra>'
        ))
    
    fig3.update_layout(
        title={
            'text': 'Annual Rainfall Distribution (Box Plot)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        yaxis_title='Annual Rainfall (mm)',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # ========================================================================
    # GRAPH 4: Rainfall Heatmap
    # ========================================================================
    if len(selected_states) == 1:
        # Single state: Year vs Month heatmap
        state = selected_states[0]
        heatmap_data = filtered_month[filtered_month['STATE'] == state].pivot(
            index='MONTH', columns='YEAR', values='RAINFALL'
        )
        heatmap_data = heatmap_data.reindex(month_order)
        
        fig4 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, '#ffffff'], [0.5, '#234567'], [1, '#001122']],
            colorbar=dict(title='Rainfall (mm)'),
            hovertemplate='Year: %{x}<br>' +
                         'Month: %{y}<br>' +
                         'Rainfall: %{z:.1f} mm<br>' +
                         '<extra></extra>'
        ))
        
        fig4.update_layout(
            title={
                'text': f'Rainfall Heatmap: {state}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='Year',
            yaxis_title='Month',
            template='plotly_white',
            height=400
        )
    else:
        # Multiple states: State vs Month comparison heatmap
        avg_by_month = filtered_month.groupby(['STATE', 'MONTH'])['RAINFALL'].mean().reset_index()
        heatmap_data = avg_by_month.pivot(index='MONTH', columns='STATE', values='RAINFALL')
        heatmap_data = heatmap_data.reindex(month_order)
        
        fig4 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, '#ffffff'], [0.5, '#234567'], [1, '#001122']],
            colorbar=dict(title='Avg Rainfall (mm)'),
            hovertemplate='State: %{x}<br>' +
                         'Month: %{y}<br>' +
                         'Avg Rainfall: %{z:.1f} mm<br>' +
                         '<extra></extra>'
        ))
        
        fig4.update_layout(
            title={
                'text': 'Average Monthly Rainfall Comparison Across States',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='State',
            yaxis_title='Month',
            template='plotly_white',
            height=400
        )
    
    return fig1, fig2, fig3, fig4

print(f"✓ Callback functions defined")

# ============================================================================
# SECTION 8: RUN THE DASHBOARD APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("STARTING DASH APPLICATION")
    print("=" * 80)
    print("\n📊 Dashboard Features:")
    print("   • Interactive state selection (multi-select)")
    print("   • Year range filtering (1947-2017)")
    print("   • 4 dynamic visualizations:")
    print("     1. Annual rainfall trend (line chart)")
    print("     2. Monthly rainfall distribution (bar chart)")
    print("     3. Rainfall distribution analysis (box plot)")
    print("     4. Rainfall heatmap (year×month or state×month)")
    print("\n🌐 Dashboard URL: http://127.0.0.1:8050/")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=True, port=8050)
