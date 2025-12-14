# main.py - Streamlit Adversarial Robustness Toolkit

import streamlit as st
import torch
import os
import tempfile
import pandas as pd
import json

from core.data_loader import get_dataloader_from_upload, get_dataloader_smart
from core.attacker import bia_attack
from core.loader_checkpoint import load_gan
from core.model_loader import load_torchscript_model_smart
from utils.transforms import load_transform_smart
from utils.dashboard import create_full_dashboard
from utils.metrics import calculate_all_metrics, print_metrics_summary, metrics_to_dataframe

# ============= Global Configuration =============

# Page config
st.set_page_config(
    page_title="Adversarial Robustness Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'args' not in st.session_state:
    st.session_state.args = None
if 'attack_complete' not in st.session_state:
    st.session_state.attack_complete = False


# ============= Helper Functions =============

class Args:
    """Configuration class for attack parameters"""
    def __init__(self, target_model_source=None, model_type="vgg19", RN=True, DA=False, 
                 eps=0.0314, user_transform_source=None, shuffle_data=False):
        self.model_type = model_type
        self.RN = RN
        self.DA = DA
        self.eps = eps
        self.loaded_target_model = None
        self.user_transform = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.shuffle_data = shuffle_data
        
        # Load target model (works with path OR upload)
        if target_model_source:
            self.loaded_target_model, param_count = load_torchscript_model_smart(
                target_model_source, device=self.device
            )
            st.sidebar.success(f"✓ Model loaded ({param_count:,} parameters)")
        else:
            default_model_path = os.path.join("default_model", "cifar10.ts")
            if not os.path.exists(default_model_path):
                st.sidebar.error(f"❌ Default model not found at {default_model_path}")
                st.stop()
            self.loaded_target_model, param_count = load_torchscript_model_smart(
                default_model_path, device=self.device
            )
            st.sidebar.info(f"Using default CIFAR-10 model ({param_count:,} parameters)")
        
        # Load user transform (works with path OR upload)
        if user_transform_source:
            self.user_transform = load_transform_smart(user_transform_source)
            st.sidebar.success("✓ Custom transforms loaded")


# ============= Streamlit UI =============

st.title("🛡️ Adversarial Robustness Toolkit")
st.markdown(f"Test your model's robustness")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Model upload
st.sidebar.subheader("1. Target Model")
target_model = st.sidebar.file_uploader(
    "Upload TorchScript model (.ts)",
    type=["ts"],
    help="Upload your PyTorch TorchScript model to test"
)
use_default = st.sidebar.checkbox("Use default CIFAR-10 model", value=(target_model is None))

# Model type for attacker (fixed to vgg19)
st.sidebar.subheader("2. Attack Configuration")
model_type = "vgg19"
st.sidebar.caption("🎯 Using VGG19-based attacker")

# Attack parameters
st.sidebar.subheader("3. Attack Parameters")
eps = st.sidebar.slider(
    "Epsilon (ε)",
    min_value=0.0,
    max_value=64.0/255.0,
    value=8.0/255.0,
    step=1.0/255.0,
    format="%.4f",
    help="Maximum perturbation budget (L∞ norm)"
)

# Attack boost options
attack_strategy = st.sidebar.radio(
    "Attack Boost",
    ["Random Norm (RN)", "Domain Agnostic Attention (DA)", "None"],
    index=0,
    help="Attack boosts increase probability of fooling models"
)
rn = (attack_strategy == "Random Norm (RN)")
da = (attack_strategy == "Domain Agnostic Attention (DA)")

if rn or da:
    st.sidebar.caption(f"✓ Using: {attack_strategy}")
else:
    st.sidebar.caption("✓ No attack boost")

# Transform upload
st.sidebar.subheader("4. Custom Transforms (Optional)")
transform_json = st.sidebar.file_uploader(
    "Upload Transform JSON (.json)",
    type=["json"],
    help="Upload custom preprocessing transforms that must be applied to tensors"
)

# Dataset upload
st.sidebar.subheader("5. Test Dataset")
dataset_file = st.sidebar.file_uploader(
    "Upload test dataset (.zip)",
    type=["zip"],
    help="Upload a zip file containing test images"
)

n_examples = st.sidebar.number_input(
    "Number of examples to test",
    min_value=1,
    max_value=1000,
    value=64,
    step=1
)

shuffle_data = st.sidebar.checkbox("Shuffle dataset", value=True)

# Show device info
device_icon = "🖥️ GPU" if torch.cuda.is_available() else "💻 CPU"
st.sidebar.markdown(f"**Device:** {device_icon}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎨 Visualizations", "📈 Detailed Metrics", "📥 Download"])

# ============= Tab 1: Overview =============
with tab1:
    st.header("Attack Overview")
    
    st.markdown(f"""
    ### How it works:
    1. **Upload Model**: Use your PyTorch TorchScript model or the default CIFAR-10 model
    2. **Configure Attack**: Select attack boost (RN/DA) to increase attack strength
    3. **Set Epsilon**: Choose the maximum perturbation budget
    4. **Upload Dataset**: Provide test images as a zip file
    5. **Run Attack**: Generate adversarial examples
    6. **Analyze Results**: Explore interactive visualizations and download results
    
    ### About the Attack:
    Series of attacks that generates 
    imperceptible perturbations to fool deep learning models without requiring 
    gradient information from the target model.
    """)
    
    # Run attack button
    run_disabled = dataset_file is None
    
    if st.button(f"🚀 Run Attack", type="primary", disabled=run_disabled, use_container_width=True):
        with st.spinner("Running attack... This may take a few minutes."):
            try:
                # Create Args object with uploaded files
                args = Args(
                    target_model_source=target_model if not use_default else None,
                    model_type=model_type,
                    RN=rn,
                    DA=da,
                    eps=eps,
                    user_transform_source=transform_json,
                    shuffle_data=shuffle_data
                )
                
                # Load dataloader directly from uploaded zip
                with st.spinner("Loading dataset..."):
                    loader = get_dataloader_from_upload(
                        dataset_file,
                        n_examples=n_examples,
                        shuffle=shuffle_data
                    )
                
                # Load attacker
                with st.spinner("Loading attacker..."):
                    attacker = load_gan(args)
                
                # Run attack
                with st.spinner("Generating adversarial examples..."):
                    results = bia_attack(
                        attacker,
                        args.loaded_target_model,
                        loader,
                        args.device,
                        args.eps,
                        user_transform=args.user_transform
                    )
                
                # Calculate metrics
                with st.spinner("Calculating metrics..."):
                    # Check if any samples were fooled
                    if results['num_fooled'] == 0:
                        st.warning("⚠️ Attack complete, but no samples were fooled! The model is very robust.")
                        st.info("This could mean: (1) The model is well-defended, (2) Epsilon is too small, or (3) The attack strategy needs adjustment.")
                        
                        # Store minimal results
                        st.session_state.results = results
                        st.session_state.metrics = None
                        st.session_state.args = args
                        st.session_state.attack_complete = False  # Don't show viz tabs
                        st.stop()
                    
                    metrics = calculate_all_metrics(
                        clean_imgs=results["clean_imgs"],
                        adv_imgs=results["adv_imgs"],
                        true_labels=results["true_labels"],
                        clean_preds=results["clean_preds"],
                        adv_preds=results["adv_preds"],
                        clean_probs=results.get("clean_probs"),
                        adv_probs=results.get("adv_probs"),
                        total_samples=results.get("total_samples"),
                        clean_correct=results.get("clean_correct"),
                        device=args.device
                    )
                
                # Store in session state
                st.session_state.results = results
                st.session_state.metrics = metrics
                st.session_state.args = args
                st.session_state.attack_complete = True
                
                st.success(f"✅ Attack complete! Fooled {results['num_fooled']} out of {results['total_samples']} samples")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during attack: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    # with col2:
    #     st.info(f"""
    #     **Current Configuration:**
        
    #     📱 **Device:** {device_icon}
        
    #     🎯 **Attack:** {ATTACK_NAME}
        
    #     ⚡ **Boost:** {attack_strategy if attack_strategy != "None" else "None"}
        
    #     📊 **Dataset:** {'✓ Loaded' if dataset_file else '✗ Not loaded'}
        
    #     🔧 **Model:** {'Default CIFAR-10' if use_default else '✓ Custom' if target_model else '✗ Not loaded'}
        
    #     ⚙️ **Epsilon:** {eps:.4f}
        
    #     📦 **Samples:** {n_examples}
    #     """)
    
    # Display results summary if attack is complete
    if st.session_state.attack_complete and st.session_state.metrics:
        st.markdown("---")
        st.subheader("⚡ Attack Results Summary")
        
        metrics = st.session_state.metrics
        
        st.markdown("#### Attack Success")
        col1, col3, col4 = st.columns(3)
        col1.metric("Attack Success Rate", f"{metrics['attack_success_rate']:.2f}%", 
                   delta=f"{metrics['fooled_samples']} fooled", delta_color="inverse")
        # col2.metric("Robust Accuracy", f"{metrics['robust_accuracy']:.2f}%",
        #            delta=f"{metrics['total_samples'] - metrics['fooled_samples']} survived")
        col3.metric("Clean Accuracy", f"{metrics['clean_accuracy']:.2f}%")
        col4.metric("Fooling Rate", f"{metrics['fooling_rate']:.2f}%",
                   help="% of correctly classified samples that were fooled")
        
        st.markdown("#### Perturbation & Similarity")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg L∞ Norm", f"{metrics['avg_linf_norm']:.4f}",
                   help="Maximum pixel-wise perturbation")
        col2.metric("Avg L2 Norm", f"{metrics['avg_l2_norm']:.4f}",
                   help="Euclidean distance")
        col3.metric("Avg SSIM", f"{metrics['avg_ssim']:.4f}",
                   help="Structural Similarity (higher = more similar)")
        col4.metric("Avg LPIPS", f"{metrics['avg_lpips']:.4f}",
                   help="Learned Perceptual Similarity (lower = more similar)")

# ============= Tab 2: Visualizations =============
with tab2:
    if st.session_state.attack_complete and st.session_state.results and st.session_state.metrics:
        st.header("Interactive Visualizations")
        
        # Create full dashboard
        create_full_dashboard(
            clean_imgs=st.session_state.results["clean_imgs"],
            adv_imgs=st.session_state.results["adv_imgs"],
            true_labels=st.session_state.results["true_labels"],
            clean_preds=st.session_state.results["clean_preds"],
            adv_preds=st.session_state.results["adv_preds"],
            metrics=st.session_state.metrics
        )
    else:
        st.info("👈 Run a BIA attack from the Overview tab to see visualizations")
        st.markdown("""
        ### Available Visualizations:
        - 📊 **Metrics Overview**: Real-time performance cards with deltas
        - 📏 **Perturbation Analysis**: L2, L∞, L0 norm distributions
        - 👁️ **Perceptual Metrics**: SSIM, PSNR, LPIPS distributions
        - 📈 **Interactive Histograms**: Tabbed metric comparisons
        - 🔍 **Scatter Plots**: Explore relationships between metrics
        - 🔥 **Correlation Matrix**: Interactive heatmap of metric correlations
        - 🎯 **Radar Chart**: Normalized attack profile
        - 🖼️ **Image Explorer**: Filter, sort, and analyze individual examples
        - 🌡️ **Perturbation Heatmaps**: Visualize where changes occur
        """)

# ============= Tab 3: Detailed Metrics =============
with tab3:
    if st.session_state.attack_complete and st.session_state.metrics:
        st.header("Detailed Metrics Analysis")
        
        metrics = st.session_state.metrics
        
        # Show detailed metrics table
        st.subheader("📋 Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Attack Success Metrics")
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Total Samples",
                    "Correctly Classified (Clean)",
                    "Fooled Samples",
                    "Attack Success Rate",
                    # "Robust Accuracy",
                    "Clean Accuracy",
                    "Fooling Rate"
                ],
                "Value": [
                    metrics['total_samples'],
                    metrics['correctly_classified'],
                    metrics['fooled_samples'],
                    f"{metrics['attack_success_rate']:.2f}%",
                    # f"{metrics['robust_accuracy']:.2f}%",
                    f"{metrics['clean_accuracy']:.2f}%",
                    f"{metrics['fooling_rate']:.2f}%"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Perturbation & Perceptual Metrics")
            pert_df = pd.DataFrame({
                "Metric": [
                    "Avg L2 Norm",
                    "Max L2 Norm",
                    "Avg L∞ Norm",
                    "Max L∞ Norm",
                    "Avg L0 Norm",
                    "Avg SSIM",
                    "Avg PSNR",
                    "Avg LPIPS",
                    "Avg MSE"
                ],
                "Value": [
                    f"{metrics['avg_l2_norm']:.4f}",
                    f"{metrics['max_l2_norm']:.4f}",
                    f"{metrics['avg_linf_norm']:.4f}",
                    f"{metrics['max_linf_norm']:.4f}",
                    f"{metrics['avg_l0_norm']:.0f} pixels",
                    f"{metrics['avg_ssim']:.4f}",
                    f"{metrics['avg_psnr']:.2f} dB",
                    f"{metrics['avg_lpips']:.4f}",
                    f"{metrics['avg_mse']:.6f}"
                ]
            })
            st.dataframe(pert_df, hide_index=True, use_container_width=True)
        
        # Per-image metrics table
        st.markdown("---")
        st.subheader("📊 Per-Image Metrics")
        st.markdown("Detailed metrics for each adversarial example generated")
        
        per_image_df = metrics_to_dataframe(metrics)
        st.dataframe(per_image_df, use_container_width=True, height=400)
        
        # Statistical summary
        st.markdown("---")
        st.subheader("📈 Statistical Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**L∞ Norm Statistics**")
            st.write(f"Mean: {metrics['avg_linf_norm']:.4f}")
            st.write(f"Median: {pd.Series(metrics['linf_norms']).median():.4f}")
            st.write(f"Std Dev: {pd.Series(metrics['linf_norms']).std():.4f}")
            st.write(f"Max: {metrics['max_linf_norm']:.4f}")
        
        with col2:
            st.markdown("**SSIM Statistics**")
            st.write(f"Mean: {metrics['avg_ssim']:.4f}")
            st.write(f"Median: {pd.Series(metrics['ssim_scores']).median():.4f}")
            st.write(f"Std Dev: {pd.Series(metrics['ssim_scores']).std():.4f}")
            st.write(f"Min: {min(metrics['ssim_scores']):.4f}")
        
        with col3:
            st.markdown("**LPIPS Statistics**")
            st.write(f"Mean: {metrics['avg_lpips']:.4f}")
            st.write(f"Median: {pd.Series(metrics['lpips_scores']).median():.4f}")
            st.write(f"Std Dev: {pd.Series(metrics['lpips_scores']).std():.4f}")
            st.write(f"Max: {max(metrics['lpips_scores']):.4f}")
        
    else:
        st.info("👈 Run an attack from the Overview tab to see detailed metrics")
        st.markdown("""
        ### Metrics Included:
        - **Attack Success**: Success rate, robust accuracy, fooling rate
        - **Perturbation Norms**: L0, L2, L∞ distances
        - **Perceptual Similarity**: SSIM, PSNR, LPIPS
        - **Statistical Analysis**: Mean, median, std dev for all metrics
        - **Per-Image Breakdown**: Individual metrics for each example
        """)

# ============= Tab 4: Download =============
with tab4:
    if st.session_state.attack_complete and st.session_state.results and st.session_state.metrics:
        st.header("📥 Download Results")
        
        st.markdown("""
        Export your attack results for further analysis or reporting.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Metrics Report")
            
            # JSON download - all metrics
            metrics_report = {
                "attack_configuration": {
                    "epsilon": st.session_state.args.eps,
                    "random_norm": st.session_state.args.RN,
                    "domain_agnostic_attention": st.session_state.args.DA,
                    "device": st.session_state.args.device,
                },
                "attack_success_metrics": {
                    "total_samples": st.session_state.metrics['total_samples'],
                    "correctly_classified": st.session_state.metrics['correctly_classified'],
                    "fooled_samples": st.session_state.metrics['fooled_samples'],
                    "attack_success_rate": st.session_state.metrics['attack_success_rate'],
                    # "robust_accuracy": st.session_state.metrics['robust_accuracy'],
                    "clean_accuracy": st.session_state.metrics['clean_accuracy'],
                    "fooling_rate": st.session_state.metrics['fooling_rate']
                },
                "perturbation_metrics": {
                    "avg_l2_norm": st.session_state.metrics['avg_l2_norm'],
                    "max_l2_norm": st.session_state.metrics['max_l2_norm'],
                    "avg_linf_norm": st.session_state.metrics['avg_linf_norm'],
                    "max_linf_norm": st.session_state.metrics['max_linf_norm'],
                    "avg_l0_norm": st.session_state.metrics['avg_l0_norm']
                },
                "perceptual_metrics": {
                    "avg_ssim": st.session_state.metrics['avg_ssim'],
                    "avg_psnr": st.session_state.metrics['avg_psnr'],
                    "avg_lpips": st.session_state.metrics['avg_lpips'],
                    "avg_mse": st.session_state.metrics['avg_mse']
                }
            }
            
            metrics_json = json.dumps(metrics_report, indent=2)
            
            st.download_button(
                label="📥 Download Metrics Report (JSON)",
                data=metrics_json,
                file_name="attack_metrics_report.json",
                mime="application/json",
                use_container_width=True,
                help="Download comprehensive metrics report"
            )
        
        with col2:
            st.subheader("🖼️ Adversarial Images")
            
            # Download adversarial images as ZIP
            import zipfile
            import io
            from PIL import Image
            
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Save only adversarial images
                for i in range(len(st.session_state.results["adv_imgs"])):
                    adv_img = st.session_state.results["adv_imgs"][i]
                    img_array = adv_img.cpu().permute(1, 2, 0).numpy()
                    img_array = (img_array * 255).astype('uint8')
                    img_pil = Image.fromarray(img_array)
                    
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format='PNG')
                    
                    true_label = st.session_state.results["true_labels"][i].item()
                    adv_pred = st.session_state.results["adv_preds"][i].item()
                    clean_pred = st.session_state.results["clean_preds"][i].item()
                    zip_file.writestr(
                        f"img_{i:04d}_true{true_label}_clean{clean_pred}_adv{adv_pred}.png",
                        img_buffer.getvalue()
                    )
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Adversarial Images (ZIP)",
                data=zip_buffer,
                file_name="adversarial_images.zip",
                mime="application/zip",
                use_container_width=True,
                help="Download adversarial images with labeled filenames"
            )
            
            st.caption(f"📦 Contains {len(st.session_state.results['adv_imgs'])} adversarial images")
            st.caption("Filename format: img_XXXX_trueY_cleanZ_advW.png")
        
    else:
        st.info("👈 Run an attack from the Overview tab to download results")
        st.markdown("""
        ### Available Downloads:
        - **Metrics Report (JSON)**: Comprehensive attack and perturbation metrics
        - **Adversarial Images (ZIP)**: All generated adversarial examples
        """)

# ============= Footer =============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Adversarial Robustness Toolkit</strong> | Built with Streamlit + PyTorch</p>
</div>
""", unsafe_allow_html=True)


