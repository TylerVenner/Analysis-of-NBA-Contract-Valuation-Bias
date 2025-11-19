import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _scale_variances_to_marker_size(log_variances_nd, min_size=2, max_size=10):
    """
    Helper: Converts log-variances into a scaled marker size.
    Adjusted for 3D rendering (smaller scalars required compared to 2D pixels).
    """
    # Sum variances across all 3 dimensions
    total_variances = np.sum(np.exp(log_variances_nd), axis=1)
    log_v = np.log(total_variances + 1e-9)
    
    min_v, max_v = np.min(log_v), np.max(log_v)
    if max_v == min_v: 
        return np.full_like(total_variances, min_size)
    
    # Min-Max scale
    return min_size + (log_v - min_v) / (max_v - min_v) * (max_size - min_size)

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

    # Use the index of the attribution matrix as Player Names
    player_names = attribution_matrix.index.tolist()
    n_players = len(player_names)
    
    # Calculate Marker Sizes (Uncertainty)
    player_sizes = _scale_variances_to_marker_size(fitter.player_lvars, 3, 8) # Slightly larger for visibility
    factor_sizes = _scale_variances_to_marker_size(fitter.factor_lvars, 6, 15)

    # --- 2. Prepare Hover Text ---
    player_hover = []
    for i, name in enumerate(player_names):
        meta_str = f"Type: {player_metadata.iloc[i]}" if player_metadata is not None else ""
        row_values = attribution_matrix.iloc[i]
        # Show top 3 strongest biases (positive or negative magnitude)
        top_factors = row_values.abs().sort_values(ascending=False).head(3).index
        factors_str = "<br>".join([f"{f}: {row_values[f]:.2f}" for f in top_factors])
        
        text = (f"<b>{name}</b><br>{meta_str}<br>"
                f"<i>Uncertainty: {player_sizes[i]:.1f}</i><br>"
                f"<b>Key Drivers:</b><br>{factors_str}")
        player_hover.append(text)

    factor_hover = []
    for i, label in enumerate(bias_labels):
        text = (f"<b>{label}</b><br>"
                f"Latent Uncertainty: {factor_sizes[i]:.1f}")
        factor_hover.append(text)

    fig = go.Figure()

    # --- 3. Add Player Traces (Colored by Metadata) ---
    if player_metadata is not None:
        unique_types = player_metadata.unique()
        # Standard Tableau 10 palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, p_type in enumerate(unique_types):
            mask = (player_metadata == p_type).values
            indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter3d(
                x=player_coords[indices, 0],
                y=player_coords[indices, 1],
                z=player_coords[indices, 2],
                mode='markers', # Only markers, text is on hover
                name=str(p_type),
                marker=dict(
                    size=player_sizes[indices],
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(width=0) 
                ),
                # We set 'text' to names, but mode='markers' means it only shows on hover
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
        mode='markers+text',
        name='Bias Factors',
        marker=dict(
            size=factor_sizes,
            color='#FFD700', # Gold
            symbol='diamond',
            opacity=1.0
        ),
        text=bias_labels,
        textposition="top center",
        textfont=dict(size=12, color='white', family="Arial Black"),
        hovertext=factor_hover,
        hoverinfo='text'
    ))

    # --- 5. 3D Scene Layout (Clean "Void" Look) ---
    axis_template = dict(
        showgrid=False,      # No grid lines
        zeroline=False,      # No zero line
        showbackground=False, # No background plane color
        showticklabels=False, # No numeric ticks
        title='',             # No axis titles
        visible=False         # Hide the axis line itself
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color='white')),
        template="plotly_dark",
        paper_bgcolor='#111111',
        
        # The Scene Dictionary controls 3D behavior
        scene=dict(
            xaxis=axis_template,
            yaxis=axis_template,
            zaxis=axis_template,
            # Critical: Use 'data' aspectmode so X, Y, and Z scales are equal.
            aspectmode='data',
            dragmode='orbit' # Better for exploring 3D structure
        ),
        
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=0.05,
            bgcolor="rgba(0,0,0,0)", # Transparent legend
            font=dict(color="white")
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=900
    )

    fig.write_html(output_path)
    print(f"3D Map saved to: {output_path}")