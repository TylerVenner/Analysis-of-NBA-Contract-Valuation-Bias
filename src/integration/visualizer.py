import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_bias_map_3d(fitter, attribution_matrix, bias_labels, player_metadata=None, 
                     title="Latent Structure of Economic Bias (3D)", output_path="bias_map_3d.html"):
    """
    Generates an interactive 3D map of Players and Bias Factors.
    """
    # --- 1. Data Validation & Setup ---
    player_coords = fitter.player_coords
    factor_coords = fitter.factor_coords
    
    if player_coords.shape[1] != 3:
        raise ValueError(f"Fitter has {player_coords.shape[1]} dimensions, but 3D plot requires 3.")

    # Ensure we use the DataFrame Index (Player Names) for labeling
    player_names = attribution_matrix.index.tolist()
    n_players = len(player_names)
    
    # Calculate Marker Sizes (Uncertainty)
    player_sizes = 3 
    factor_sizes = 12

    # --- 2. Prepare Hover Text ---
    player_hover = []
    for i, name in enumerate(player_names):
        meta_str = f"Type: {player_metadata.iloc[i]}" if player_metadata is not None else ""
        row_values = attribution_matrix.iloc[i]
        
        # Show top 3 strongest biases (positive or negative magnitude)
        # We sort by absolute value to see what DRIVES the salary, regardless of direction
        top_factors = row_values.abs().sort_values(ascending=False).head(3).index
        factors_str = "<br>".join([f"{f}: {row_values[f]:.2f}" for f in top_factors])
        
        text = (f"<b>{name}</b><br>{meta_str}<br>"
                f"<b>Key Drivers:</b><br>{factors_str}")
        player_hover.append(text)

    factor_hover = []
    for i, label in enumerate(bias_labels):
        text = (f"<b>{label}</b><br>")
        factor_hover.append(text)

    fig = go.Figure()

    # --- 3. Add Player Traces (Colored by Metadata) ---
    if player_metadata is not None:
        unique_types = player_metadata.unique()
        # Distinct color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, p_type in enumerate(unique_types):
            mask = (player_metadata == p_type).values
            indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter3d(
                x=player_coords[indices, 0],
                y=player_coords[indices, 1],
                z=player_coords[indices, 2],
                mode='markers', # Markers only (names are on hover to prevent clutter)
                name=str(p_type),
                marker=dict(
                    size=4,
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(width=0) 
                ),
                text=[player_names[j] for j in indices],
                hovertext=[player_hover[j] for j in indices],
                hoverinfo='text'
            ))
    else:
        # Single trace
        fig.add_trace(go.Scatter3d(
            x=player_coords[:, 0],
            y=player_coords[:, 1],
            z=player_coords[:, 2],
            mode='markers',
            name='Players',
            marker=dict(
                size=player_sizes,
                color='#1f77b4',
                opacity=0.8
            ),
            text=player_names,
            hovertext=player_hover,
            hoverinfo='text'
        ))

    # --- 4. Add Bias Factor Trace (The Anchors) ---
    fig.add_trace(go.Scatter3d(
        x=factor_coords[:, 0],
        y=factor_coords[:, 1],
        z=factor_coords[:, 2],
        mode='markers+text', # Text ALWAYS visible for Factors
        name='Bias Factors',
        marker=dict(
            size=10,
            color='#FFD700', # Gold
            symbol='diamond',
            opacity=1.0,
            line=dict(width=2, color='black')
        ),
        text=bias_labels, # Static text
        textposition="top center",
        textfont=dict(size=14, color='white', family="Arial Black"),
        hovertext=factor_hover,
        hoverinfo='text'
    ))

    # --- 5. 3D Scene Layout (Clean "Void" Look) ---
    # This template hides the grid lines, zero lines, and axis labels
    axis_template = dict(
        showgrid=False,
        zeroline=False,
        showbackground=False,
        showticklabels=False, # Hide numbers on axes
        title='',             # Hide axis titles
        visible=False         # Hide the axis line itself
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color='white')),
        template="plotly_dark",
        paper_bgcolor='#111111', # Very dark grey background
        
        scene=dict(
            xaxis=axis_template,
            yaxis=axis_template,
            zaxis=axis_template,
            aspectmode='data', # Preserve geometric distances
            dragmode='orbit'   # Orbit allows smoother rotation
        ),
        
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=0.05,
            bgcolor="rgba(0,0,0,0)", # Transparent legend
            font=dict(color="white", size=12)
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=900
    )

    fig.write_html(output_path)
    print(f"3D Map saved to: {output_path}")