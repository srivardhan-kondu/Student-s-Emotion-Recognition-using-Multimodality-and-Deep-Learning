"""
Streamlit Dashboard for Multimodal Emotion Recognition System.
Implements FR1-FR5, FR15-FR17.
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from fusion.multimodal_predictor import MultimodalEmotionPredictor
from config import DASHBOARD_CONFIG
from utils.voice_recorder import VoiceRecorder

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG['title'],
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.predictor = MultimodalEmotionPredictor()
        st.session_state.predictor.load_models()
    st.success("✅ Models loaded successfully!")

if 'history' not in st.session_state:
    st.session_state.history = []


def plot_emotion_distribution(emotion_scores):
    """Create emotion distribution bar chart."""
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=scores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9'],
            text=[f'{s:.2%}' for s in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )
    
    return fig


def plot_emotion_radar(emotion_scores):
    """Create radar chart for emotion scores."""
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=emotions,
        fill='toself',
        marker_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        title="Emotion Radar Chart"
    )
    
    return fig


def display_result(result):
    """Display emotion prediction result."""
    if result is None:
        st.error("❌ No prediction available")
        return
    
    # Main emotion display
    st.markdown(f"""
    <div class="emotion-card">
        <h1>Detected Emotion: {result['emotion'].upper()}</h1>
        <h3>Confidence: {result['confidence']:.2%}</h3>
        <p>Modalities Used: {result['modalities_used']}/3</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Emotion distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_emotion_distribution(result['emotion_scores']), use_container_width=True)
    
    with col2:
        st.plotly_chart(plot_emotion_radar(result['emotion_scores']), use_container_width=True)
    
    # Individual modality results
    st.subheader("📊 Individual Modality Results")
    
    mod_cols = st.columns(3)
    
    individual = result.get('individual_results', {})
    
    with mod_cols[0]:
        st.markdown("**👁️ Facial Recognition**")
        if individual.get('facial'):
            st.success(f"Emotion: {individual['facial']['emotion']}")
            st.info(f"Confidence: {individual['facial']['confidence']:.2%}")
        else:
            st.warning("Not available")
    
    with mod_cols[1]:
        st.markdown("**🎤 Speech Analysis**")
        if individual.get('speech'):
            st.success(f"Emotion: {individual['speech']['emotion']}")
            st.info(f"Confidence: {individual['speech']['confidence']:.2%}")
        else:
            st.warning("Not available")
    
    with mod_cols[2]:
        st.markdown("**📝 Text Analysis**")
        if individual.get('text'):
            st.success(f"Emotion: {individual['text']['emotion']}")
            st.info(f"Confidence: {individual['text']['confidence']:.2%}")
        else:
            st.warning("Not available")


# Main app
st.markdown('<h1 class="main-header">😊 Student Emotion Recognition System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Fusion settings
    st.subheader("Fusion Configuration")
    fusion_type = st.selectbox(
        "Fusion Strategy",
        ["calibrated", "weighted", "adaptive", "voting"],
        index=0,
        help="'calibrated' = temperature-scaled + confidence-gated (best)"
    )
    
    if fusion_type in ["weighted", "adaptive"]:
        st.write("**Modality Weights**")
        facial_weight = st.slider("Facial Weight", 0.0, 1.0, 0.4, 0.1)
        speech_weight = st.slider("Speech Weight", 0.0, 1.0, 0.3, 0.1)
        text_weight = st.slider("Text Weight", 0.0, 1.0, 0.3, 0.1)
        
        # Update fusion weights
        st.session_state.predictor.fusion.fusion_type = fusion_type
        st.session_state.predictor.fusion.update_weights(facial_weight, speech_weight, text_weight)
    
    st.divider()
    
    # History
    st.subheader("📜 Prediction History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history[-5:])):
            st.caption(f"{i+1}. {entry['emotion']} ({entry['confidence']:.1%})")
    else:
        st.caption("No predictions yet")
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Multimodal", "👁️ Image", "🎤 Audio", "📝 Text"])

# Multimodal Tab (FR3, FR4, FR5)
with tab1:
    st.header("Multimodal Emotion Recognition")
    st.write("Upload image, audio, and/or text for comprehensive emotion analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Image Input")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key='multi_image')
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🎵 Audio Input")
        audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'ogg'], key='multi_audio')
        if audio_file:
            st.audio(audio_file)
    
    with col3:
        st.subheader("✍️ Text Input")
        text_input = st.text_area("Enter text", height=150, key='multi_text')
    
    if st.button("🚀 Analyze All Modalities", type="primary", use_container_width=True):
        # Validate that at least one input is provided
        has_input = image_file or audio_file or (text_input and text_input.strip())
        if not has_input:
            st.warning("⚠️ Please upload an image, audio file, or enter text before analyzing!")
        else:
          with st.spinner("Analyzing..."):
            # Save uploaded files temporarily
            image_path = None
            audio_path = None
            
            if image_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(image_file.read())
                    image_path = tmp.name
            
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_file.read())
                    audio_path = tmp.name
            
            text = text_input if text_input.strip() else None
            
            # Predict
            result = st.session_state.predictor.predict_multimodal(
                image_path=image_path,
                audio_path=audio_path,
                text=text
            )
            
            # Clean up temp files
            if image_path:
                os.unlink(image_path)
            if audio_path:
                os.unlink(audio_path)
            
            # Display result
            display_result(result)
            
            # Add to history
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'modalities': result['modalities_used']
            })

# Image Tab (FR3)
with tab2:
    st.header("Facial Emotion Recognition")
    
    image_source = st.radio("Select Image Source", ["Upload", "Camera"], horizontal=True)
    
    if image_source == "Upload":
        uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key='image_only')
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Facial Emotion", type="primary"):
                with st.spinner("Analyzing facial expression..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name)
                        result = st.session_state.predictor.predict_from_image(tmp.name)
                        os.unlink(tmp.name)
                    
                    if result:
                        display_result({'emotion': result['emotion'], 
                                      'confidence': result['confidence'],
                                      'emotion_scores': dict(zip(['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise'], 
                                                                result['probabilities'])),
                                      'modalities_used': 1,
                                      'individual_results': {'facial': result}})
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            if st.button("Analyze Facial Emotion", type="primary"):
                with st.spinner("Analyzing facial expression..."):
                    image = Image.open(camera_image)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name)
                        result = st.session_state.predictor.predict_from_image(tmp.name)
                        os.unlink(tmp.name)
                    
                    if result:
                        display_result({'emotion': result['emotion'], 
                                      'confidence': result['confidence'],
                                      'emotion_scores': dict(zip(['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise'], 
                                                                result['probabilities'])),
                                      'modalities_used': 1,
                                      'individual_results': {'facial': result}})

# Audio Tab (FR4)
with tab3:
    st.header("Speech Emotion Analysis")

    audio_input_mode = st.radio(
        "Select Audio Source",
        ["📁 Upload File", "🎙️ Record Voice"],
        horizontal=True
    )

    # ─── Upload File ────────────────────────────────────────────────────────
    if audio_input_mode == "📁 Upload File":
        uploaded_audio = st.file_uploader(
            "Upload Audio File", type=['wav', 'mp3', 'ogg'], key='audio_only'
        )

        if uploaded_audio:
            st.audio(uploaded_audio)

            if st.button("Analyze Speech Emotion", type="primary"):
                with st.spinner("Analyzing speech emotion..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(uploaded_audio.read())
                        result = st.session_state.predictor.predict_from_audio(tmp.name)
                        os.unlink(tmp.name)

                    if result:
                        display_result({
                            'emotion': result['emotion'],
                            'confidence': result['confidence'],
                            'emotion_scores': dict(zip(
                                ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise'],
                                result['probabilities']
                            )),
                            'modalities_used': 1,
                            'individual_results': {'speech': result}
                        })

    # ─── Live Voice Recording ────────────────────────────────────────────────
    else:
        st.markdown("### 🎙️ Live Voice Recording")

        # Check microphone availability
        mic_available = VoiceRecorder.is_available()
        if not mic_available:
            st.error(
                "⚠️ No microphone detected or sounddevice not installed.\n\n"
                "Install with: `pip install sounddevice scipy`"
            )
        else:
            st.success("✅ Microphone ready")

            col_dur, col_btn = st.columns([2, 1])
            with col_dur:
                duration = st.slider(
                    "Recording Duration (seconds)", min_value=3, max_value=10,
                    value=5, step=1,
                    help="Speak into your microphone during this time"
                )
            with col_btn:
                st.write("")  # spacing
                record_clicked = st.button(
                    "⏺️ Record & Analyze", type="primary",
                    use_container_width=True,
                    disabled=not mic_available
                )

            if record_clicked:
                recorder = VoiceRecorder(sample_rate=22050)

                # Countdown
                placeholder = st.empty()
                for remaining in range(duration, 0, -1):
                    placeholder.info(
                        f"🎙️ Recording... {remaining}s remaining — speak now!"
                    )
                    import time
                    time.sleep(1)
                placeholder.empty()

                with st.spinner("Processing recording..."):
                    try:
                        audio_array, wav_path = recorder.record(duration=duration)

                        # Show waveform
                        st.markdown("**📊 Recorded Waveform**")
                        t = np.linspace(0, duration, len(audio_array))
                        fig_wave = go.Figure()
                        fig_wave.add_trace(go.Scatter(
                            x=t, y=audio_array,
                            mode='lines',
                            line=dict(color='#667eea', width=1),
                            name='Waveform'
                        ))
                        fig_wave.update_layout(
                            xaxis_title="Time (s)",
                            yaxis_title="Amplitude",
                            height=200,
                            margin=dict(l=20, r=20, t=20, b=30),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_wave, use_container_width=True)

                        # Analyze emotion
                        result = st.session_state.predictor.predict_from_audio(wav_path)
                        recorder.cleanup(wav_path)

                        if result:
                            display_result({
                                'emotion': result['emotion'],
                                'confidence': result['confidence'],
                                'emotion_scores': dict(zip(
                                    ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise'],
                                    result['probabilities']
                                )),
                                'modalities_used': 1,
                                'individual_results': {'speech': result}
                            })
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now(),
                                'emotion': result['emotion'],
                                'confidence': result['confidence'],
                                'modalities': ['speech (live)']
                            })
                    except Exception as e:
                        st.error(f"Recording failed: {e}")
                        st.info("Make sure your microphone is connected and sounddevice is installed.")

# Text Tab (FR5)
with tab4:
    st.header("Text Emotion Analysis")
    
    text_input_only = st.text_area("Enter text for emotion analysis", height=200, key='text_only')
    
    if st.button("Analyze Text Emotion", type="primary"):
        if text_input_only.strip():
            with st.spinner("Analyzing text emotion..."):
                result = st.session_state.predictor.predict_from_text(text_input_only)
                
                if result:
                    display_result({'emotion': result['emotion'], 
                                  'confidence': result['confidence'],
                                  'emotion_scores': dict(zip(['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise'], 
                                                            result['probabilities'])),
                                  'modalities_used': 1,
                                  'individual_results': {'text': result}})
        else:
            st.warning("Please enter some text to analyze")

# Footer
st.divider()
st.caption("🎓 Student Emotion Recognition System | Multimodal Deep Learning")
