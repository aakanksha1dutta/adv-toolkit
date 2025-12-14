# utils/dashboard.py - Interactive Streamlit Dashboard for Adversarial Robustness

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
from typing import Dict
from PIL import Image
import io


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def create_metrics_overview(metrics: Dict):
    """
    Create interactive metrics overview cards
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 📊 Attack Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Attack Success Rate",
            value=f"{metrics['attack_success_rate']:.2f}%",
            delta=f"{metrics['fooled_samples']}/{metrics['total_samples']} fooled",
            delta_color="inverse"
        )
    
    with col2:  # Changed from col3
        st.metric(
            label="Clean Accuracy",
            value=f"{metrics['clean_accuracy']:.2f}%",
            delta=f"{metrics['correctly_classified']} correct"
        )
    
    with col3:  # Changed from col4
        st.metric(
            label="Fooling Rate",
            value=f"{metrics['fooling_rate']:.2f}%",
            delta="of correctly classified",
            delta_color="inverse"
        )


def create_perturbation_metrics(metrics: Dict):
    """
    Create interactive perturbation metrics display
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 📏 Perturbation Magnitude")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average L2 Norm",
            value=f"{metrics['avg_l2_norm']:.4f}",
            delta=f"Max: {metrics['max_l2_norm']:.4f}"
        )
    
    with col2:
        st.metric(
            label="Average L∞ Norm",
            value=f"{metrics['avg_linf_norm']:.4f}",
            delta=f"Max: {metrics['max_linf_norm']:.4f}"
        )
    
    with col3:
        st.metric(
            label="Average L0 Norm",
            value=f"{metrics['avg_l0_norm']:.0f}",
            delta="pixels changed"
        )


def create_perceptual_metrics(metrics: Dict):
    """
    Create interactive perceptual similarity metrics
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 👁️ Perceptual Similarity")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="SSIM",
            value=f"{metrics['avg_ssim']:.4f}",
            delta="Higher = More Similar",
            help="Structural Similarity Index (0-1)"
        )
    
    with col2:
        st.metric(
            label="PSNR",
            value=f"{metrics['avg_psnr']:.2f} dB",
            delta="Higher = Better",
            help="Peak Signal-to-Noise Ratio"
        )
    
    with col3:
        st.metric(
            label="LPIPS",
            value=f"{metrics['avg_lpips']:.4f}",
            delta="Lower = More Similar",
            help="Learned Perceptual Image Patch Similarity"
        )
    
    with col4:
        st.metric(
            label="MSE",
            value=f"{metrics['avg_mse']:.6f}",
            delta="Lower = Better",
            help="Mean Squared Error"
        )


def plot_interactive_distributions(metrics: Dict):
    """
    Create interactive distribution plots using Plotly
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 📈 Metric Distributions")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3 = st.tabs(["Perturbation Norms", "Perceptual Metrics", "All Metrics"])
    
    with tab1:
        # L2, Linf, L0 distributions
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("L2 Norm Distribution", "L∞ Norm Distribution", "L0 Norm Distribution")
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['l2_norms'], name='L2', 
                        marker_color='skyblue', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['linf_norms'], name='L∞',
                        marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['l0_norms'], name='L0',
                        marker_color='lightgreen', showlegend=False),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="L2 Distance", row=1, col=1)
        fig.update_xaxes(title_text="L∞ Distance", row=1, col=2)
        fig.update_xaxes(title_text="# Pixels Changed", row=1, col=3)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Perceptual metrics
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=("SSIM", "PSNR", "LPIPS", "MSE")
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['ssim_scores'], name='SSIM',
                        marker_color='plum', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['psnr_scores'], name='PSNR',
                        marker_color='wheat', showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['lpips_scores'], name='LPIPS',
                        marker_color='lightsteelblue', showlegend=False),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Histogram(x=metrics['mse_scores'], name='MSE',
                        marker_color='lightsalmon', showlegend=False),
            row=1, col=4
        )
        
        fig.update_xaxes(title_text="SSIM Score", row=1, col=1)
        fig.update_xaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text="LPIPS Distance", row=1, col=3)
        fig.update_xaxes(title_text="MSE", row=1, col=4)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Interactive selection of metrics to compare
        col1, col2 = st.columns(2)
        
        metric_options = {
            "L2 Norm": "l2_norms",
            "L∞ Norm": "linf_norms",
            "L0 Norm": "l0_norms",
            "SSIM": "ssim_scores",
            "PSNR": "psnr_scores",
            "LPIPS": "lpips_scores",
            "MSE": "mse_scores"
        }
        
        with col1:
            selected_metric1 = st.selectbox(
                "Select first metric",
                list(metric_options.keys()),
                index=0
            )
        
        with col2:
            selected_metric2 = st.selectbox(
                "Select second metric",
                list(metric_options.keys()),
                index=1
            )
        
        # Create side-by-side histograms
        fig = make_subplots(rows=1, cols=2, subplot_titles=(selected_metric1, selected_metric2))
        
        fig.add_trace(
            go.Histogram(x=metrics[metric_options[selected_metric1]], 
                        name=selected_metric1, marker_color='#636EFA'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=metrics[metric_options[selected_metric2]], 
                        name=selected_metric2, marker_color='#EF553B'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def plot_interactive_scatter(metrics: Dict):
    """
    Create interactive scatter plots comparing metrics
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 🔍 Metric Relationships")
    
    # Metric selection
    col1, col2 = st.columns(2)
    
    metric_options = {
        "L2 Norm": "l2_norms",
        "L∞ Norm": "linf_norms",
        "L0 Norm": "l0_norms",
        "SSIM": "ssim_scores",
        "PSNR": "psnr_scores",
        "LPIPS": "lpips_scores",
        "MSE": "mse_scores"
    }
    
    with col1:
        x_metric = st.selectbox(
            "X-axis metric",
            list(metric_options.keys()),
            index=1,  # L∞ by default
            key="scatter_x"
        )
    
    with col2:
        y_metric = st.selectbox(
            "Y-axis metric",
            list(metric_options.keys()),
            index=5,  # LPIPS by default
            key="scatter_y"
        )
    
    # Create scatter plot
    x_data = metrics[metric_options[x_metric]]
    y_data = metrics[metric_options[y_metric]]
    
    # Calculate correlation
    corr = np.corrcoef(x_data, y_data)[0, 1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            size=8,
            color=x_data,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=x_metric),
            line=dict(width=0.5, color='white')
        ),
        text=[f"Sample {i}" for i in range(len(x_data))],
        hovertemplate=f"<b>{x_metric}</b>: %{{x:.4f}}<br>" +
                      f"<b>{y_metric}</b>: %{{y:.4f}}<br>" +
                      "<b>%{text}</b><extra></extra>"
    ))
    
    fig.update_layout(
        title=f"{y_metric} vs {x_metric} (Correlation: {corr:.3f})",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation interpretation
    if abs(corr) > 0.7:
        st.info(f"🔗 Strong {'positive' if corr > 0 else 'negative'} correlation detected ({corr:.3f})")
    elif abs(corr) > 0.4:
        st.info(f"🔗 Moderate {'positive' if corr > 0 else 'negative'} correlation detected ({corr:.3f})")
    else:
        st.info(f"🔗 Weak correlation detected ({corr:.3f})")


def plot_correlation_matrix(metrics: Dict):
    """
    Create interactive correlation heatmap
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 🔥 Correlation Heatmap")
    
    # Create DataFrame
    data = {
        "L2": metrics['l2_norms'],
        "L∞": metrics['linf_norms'],
        "L0": metrics['l0_norms'],
        "SSIM": metrics['ssim_scores'],
        "PSNR": metrics['psnr_scores'],
        "LPIPS": metrics['lpips_scores'],
        "MSE": metrics['mse_scores'],
    }
    
    if metrics.get('confidence_drops') is not None:
        data["Conf_Drop"] = metrics['confidence_drops']
    
    df = pd.DataFrame(data)
    corr = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Metric Correlation Matrix",
        height=600,
        xaxis={'side': 'bottom'},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_metrics_radar(metrics: Dict):
    """
    Create radar chart for normalized metrics
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    st.markdown("### 🎯 Metrics Radar Chart")
    
    # Normalize metrics to 0-1 scale for radar chart
    categories = ['Attack\nSuccess', 'Avg L∞', 'Avg LPIPS', '1-SSIM', '1-PSNR\n(normalized)']
    
    values = [
        metrics['attack_success_rate'] / 100,
        min(metrics['avg_linf_norm'] / 0.1, 1),  # Normalize assuming max 0.1
        metrics['avg_lpips'],
        1 - metrics['avg_ssim'],
        1 - min(metrics['avg_psnr'] / 50, 1)  # Normalize assuming max 50 dB
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Attack Profile',
        line_color='#636EFA'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=500,
        title="Attack Profile (Higher = Stronger Attack)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Note: Metrics are normalized to 0-1 scale. Higher values indicate stronger attack effects.")


def plot_interactive_image_comparison(
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    true_labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    metrics: Dict,
    max_display: int = 50
):
    """
    Create interactive image comparison viewer
    """
    st.markdown("### 🖼️ Interactive Image Explorer")
    
    n_images = min(len(clean_imgs), max_display)
    
    # Simplified options - only show what works with fooled samples
    col1, col2 = st.columns(2)
    
    with col1:
        sort_option = st.selectbox(
            "Sort by",
            ["Image Index", "L∞ Norm (High to Low)", "LPIPS (High to Low)", 
             "SSIM (Low to High)"]
        )
    
    with col2:
        st.metric("Total Fooled Images", n_images)
    
    # No filtering - all images are already fooled samples
    valid_indices = list(range(n_images))  # Use list instead of tensor

    # Sort indices
    if sort_option == "L∞ Norm (High to Low)":
        linf_values = metrics['linf_norms']
        valid_indices = sorted(valid_indices, key=lambda i: linf_values[i], reverse=True)
    elif sort_option == "LPIPS (High to Low)":
        lpips_values = metrics['lpips_scores']
        valid_indices = sorted(valid_indices, key=lambda i: lpips_values[i], reverse=True)
    elif sort_option == "SSIM (Low to High)":
        ssim_values = metrics['ssim_scores']
        valid_indices = sorted(valid_indices, key=lambda i: ssim_values[i])

    # Image selector
    selected_idx = st.slider(
        "Select image",
        0,
        len(valid_indices) - 1,
        0
    )

    
    img_idx = valid_indices[selected_idx]
    
    # Display selected image pair
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Clean Image")
        clean_pil = tensor_to_pil(clean_imgs[img_idx])
        st.image(clean_pil, use_container_width=True)
        
        is_correct = clean_preds[img_idx].item() == true_labels[img_idx].item()
        if is_correct:
            st.success(f"✅ Prediction: {clean_preds[img_idx].item()} (Correct)")
        else:
            st.error(f"❌ Prediction: {clean_preds[img_idx].item()} (Wrong)")
        st.info(f"Ground Truth: {true_labels[img_idx].item()}")
    
    with col2:
        st.markdown("#### Adversarial Image")
        adv_pil = tensor_to_pil(adv_imgs[img_idx])
        st.image(adv_pil, use_container_width=True)
        
        fooled = (clean_preds[img_idx].item() == true_labels[img_idx].item()) and \
                 (adv_preds[img_idx].item() != true_labels[img_idx].item())
        
        if fooled:
            st.error(f"❌ Prediction: {adv_preds[img_idx].item()} (Fooled!)")
        else:
            st.success(f"✅ Prediction: {adv_preds[img_idx].item()}")
    
    # Display metrics for this image
    st.markdown("#### Metrics for this Image")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("L2 Norm", f"{metrics['l2_norms'][img_idx]:.4f}")
        st.metric("L∞ Norm", f"{metrics['linf_norms'][img_idx]:.4f}")
    
    with col2:
        st.metric("L0 Norm", f"{metrics['l0_norms'][img_idx]:.0f}")
        st.metric("MSE", f"{metrics['mse_scores'][img_idx]:.6f}")
    
    with col3:
        st.metric("SSIM", f"{metrics['ssim_scores'][img_idx]:.4f}")
        st.metric("PSNR", f"{metrics['psnr_scores'][img_idx]:.2f} dB")
    
    with col4:
        st.metric("LPIPS", f"{metrics['lpips_scores'][img_idx]:.4f}")
        if metrics.get('confidence_drops') is not None:
            st.metric("Conf. Drop", f"{metrics['confidence_drops'][img_idx]:.4f}")
    
    # Perturbation visualization
    st.markdown("#### Perturbation Heatmap")
    
    clean_np = clean_imgs[img_idx].cpu().numpy()
    adv_np = adv_imgs[img_idx].cpu().numpy()
    diff = np.abs(adv_np - clean_np)
    perturbation = np.mean(diff, axis=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=perturbation,
        colorscale='Hot',
        colorbar=dict(title="Magnitude")
    ))
    
    fig.update_layout(
        title=f"Perturbation Magnitude (L∞: {metrics['linf_norms'][img_idx]:.4f})",
        height=400,
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_summary_bars(metrics: Dict):
    """
    Create interactive bar charts for key metrics
    """
    st.markdown("### 📊 Summary Comparison")
    
    # Attack success metrics
    fig1 = go.Figure()
    
    categories = ['Attack\nSuccess Rate', 'Clean\nAccuracy', 'Fooling\nRate']  # Removed 'Robust\nAccuracy'
    values = [
        metrics['attack_success_rate'],
        metrics['clean_accuracy'],
        metrics['fooling_rate']
    ]
    colors = ['#ff6b6b', '#4dabf7', '#ffd43b'] 
    
    fig1.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        marker_color=colors,
        hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
    ))
    
    fig1.update_layout(
        title="Attack Success Metrics (%)",
        yaxis_title="Percentage",
        height=400,
        yaxis_range=[0, 105],
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)


def create_full_dashboard(
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    true_labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    metrics: Dict
):
    """
    Create complete interactive dashboard
    
    Args:
        clean_imgs: Clean images
        adv_imgs: Adversarial images
        true_labels: Ground truth labels
        clean_preds: Clean predictions
        adv_preds: Adversarial predictions
        metrics: Dictionary from calculate_all_metrics
    """
    # Metrics Overview
    create_metrics_overview(metrics)
    
    st.markdown("---")
    
    # Perturbation and Perceptual Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        create_perturbation_metrics(metrics)
    
    with col2:
        create_perceptual_metrics(metrics)
    
    st.markdown("---")
    
    # Interactive visualizations
    plot_summary_bars(metrics)
    
    st.markdown("---")
    
    plot_metrics_radar(metrics)
    
    st.markdown("---")
    
    plot_interactive_distributions(metrics)
    
    st.markdown("---")
    
    plot_interactive_scatter(metrics)
    
    st.markdown("---")
    
    plot_correlation_matrix(metrics)
    
    st.markdown("---")
    
    plot_interactive_image_comparison(
        clean_imgs, adv_imgs, true_labels, 
        clean_preds, adv_preds, metrics
    )