import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _scale_variances_to_marker_size(log_variances_nd, min_size=4, max_size=20):
    """
    Helper: Converts log-variances into a scaled marker size.
    Used to visually indicate model uncertainty for each point.
    """
    # Sum variances across dimensions to get total uncertainty
    total_variances = np.sum(np.exp(log_variances_nd), axis=1)
    log_v = np.log(total_variances + 1e-9)
    
    min_v, max_v = np.min(log_v), np.max(log_v)
    if max_v == min_v: 
        return np.full_like(total_variances, min_size)
    
    # Min-Max scale to the desired pixel range
    return min_size + (log_v - min_v) / (max_v - min_v) * (max_size - min_size)

def plot_bias_map_2d(fitter, attribution_matrix, bias_labels, player_metadata=None, 
                     title="Bias Attribution Map", output_path="bias_map.html"):
    """
    Generates an interactive 2D map of Players and Bias Factors.
    
    Args:
        fitter: The trained BiasMapFitter object.
        attribution_matrix (pd.DataFrame): The scaled L matrix (for hover data).
        bias_labels (list): Names of the bias factors (columns of L).
        player_metadata (pd.Series, optional): Categorical data for players (e.g. Contract Type).
                                               Index must match attribution_matrix index.
    """
    # --- 1. Setup Data ---
    player_coords = fitter.player_coords
    factor_coords = fitter.factor_coords
    player_names = attribution_matrix.index.tolist()
    n_players = len(player_names)
    
    # Calculate Marker Sizes based on Uncertainty (Learned Variance)
    player_sizes = _scale_variances_to_marker_size(fitter.player_lvars, 5, 15)
    factor_sizes = _scale_variances_to_marker_size(fitter.factor_lvars, 15, 30)

    # --- 2. Prepare Hover Text ---
    # Players: Show Name, Metadata, and Top 3 Biases
    player_hover = []
    for i, name in enumerate(player_names):
        meta_str = f"Type: {player_metadata.iloc[i]}" if player_metadata is not None else ""
        
        # Find top 3 contributing factors for this player
        row_values = attribution_matrix.iloc[i]
        top_factors = row_values.sort_values(ascending=False).head(3)
        factors_str = "<br>".join([f"{f}: {v:.2f}" for f, v in top_factors.items()])
        
        text = (f"<b>{name}</b><br>{meta_str}<br>"
                f"<i>Uncertainty: {player_sizes[i]:.1f}</i><br>"
                f"<b>Top Biases:</b><br>{factors_str}")
        player_hover.append(text)

    # Factors: Show Name and Uncertainty
    factor_hover = []
    for i, label in enumerate(bias_labels):
        text = (f"<b>{label}</b><br>"
                f"Latent Uncertainty: {factor_sizes[i]:.1f}")
        factor_hover.append(text)

    fig = go.Figure()

    # --- 3. Add Player Traces (Colored by Metadata) ---
    if player_metadata is not None:
        unique_types = player_metadata.unique()
        # Use a distinct color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, p_type in enumerate(unique_types):
            # Get indices for this group
            mask = (player_metadata == p_type).values
            indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter(
                x=player_coords[indices, 0],
                y=player_coords[indices, 1],
                mode='markers',
                name=str(p_type),
                marker=dict(
                    size=player_sizes[indices],
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[player_names[j] for j in indices],
                hovertext=[player_hover[j] for j in indices],
                hoverinfo='text'
            ))
    else:
        # Single trace if no metadata
        fig.add_trace(go.Scatter(
            x=player_coords[:, 0],
            y=player_coords[:, 1],
            mode='markers',
            name='Players',
            marker=dict(
                size=player_sizes,
                color='#1f77b4',
                opacity=0.7
            ),
            text=player_names,
            hovertext=player_hover,
            hoverinfo='text'
        ))

    # --- 4. Add Bias Factor Trace (The "Anchors") ---
    fig.add_trace(go.Scatter(
        x=factor_coords[:, 0],
        y=factor_coords[:, 1],
        mode='markers+text',
        name='Bias Factors',
        marker=dict(
            size=factor_sizes,
            color='#FFD700', # Gold
            symbol='diamond',
            line=dict(width=2, color='black')
        ),
        text=bias_labels,
        textposition="top center",
        textfont=dict(size=14, color='white', family="Arial Black"),
        hovertext=factor_hover,
        hoverinfo='text'
    ))

    # --- 5. Formatting ---
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color='white')),
        template="plotly_dark",
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis=dict(showgrid=True, gridcolor='#333', showticklabels=False, title="Latent Dimension 1"),
        yaxis=dict(showgrid=True, gridcolor='#333', showticklabels=False, title="Latent Dimension 2"),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=20, r=20, b=20, t=60),
        height=800
    )

    # Save
    fig.write_html(output_path)
    print(f"Map saved to: {output_path}")

# Note: A 3D version would follow the exact same logic but use go.Scatter3d 
# and [:, 2] for the z-axis.