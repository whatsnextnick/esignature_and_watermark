import streamlit as st
import cv2
import numpy as np
import io

# --- Utility Functions ---

def process_signature(image_data):
    """
    Processes the uploaded image to create a blue, transparent e-signature (RGBA).
    """
    # 1. Decode the image data from Streamlit uploader
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    sig_org = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if sig_org is None:
        return None

    # 2. Convert to Grayscale
    gray_img = cv2.cvtColor(sig_org, cv2.COLOR_BGR2GRAY)

    # 3. Thresholding to create a mask (signature = white, background = black)
    # This mask serves as the Alpha channel (255=opaque, 0=transparent)
    # Threshold value (180) may need adjustment.
    _, mask = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)

    # 4. Create the blue color foreground (BGR)
    blue_color = (255, 0, 0) # BGR for blue
    blue_sig_bgr = np.full(sig_org.shape, blue_color, dtype=np.uint8)

    # 5. Create an RGBA image (with Alpha Channel for transparency)
    # Convert blue BGR to RGB for web compatibility
    blue_sig_rgb = cv2.cvtColor(blue_sig_bgr, cv2.COLOR_BGR2RGB)

    r, g, b = cv2.split(blue_sig_rgb)
    
    # Merge color channels with the mask as the alpha channel
    watermark_rgba = cv2.merge((r, g, b, mask))

    return watermark_rgba

def apply_watermark(doc_img_bytes, watermark_rgba, scale_factor, position):
    """
    Applies the watermark (RGBA) onto the document image.
    """
    # 1. Decode Document Image
    doc_bytes = np.asarray(bytearray(doc_img_bytes.read()), dtype=np.uint8)
    doc_img = cv2.imdecode(doc_bytes, cv2.IMREAD_COLOR)
    
    if doc_img is None:
        return None

    doc_h, doc_w, _ = doc_img.shape
    
    # 2. Resize Watermark (Signature)
    wm_h, wm_w, _ = watermark_rgba.shape
    
    # Calculate new width based on document width and scale factor
    new_wm_w = int(doc_w * scale_factor / 100)
    
    # Resize watermark while maintaining aspect ratio
    scale = new_wm_w / wm_w
    new_wm_h = int(wm_h * scale)
    
    # Handle case where scaling makes it too big for the height
    if new_wm_h > doc_h:
        new_wm_h = int(doc_h * 0.5) # Max 50% of document height
        scale = new_wm_h / wm_h
        new_wm_w = int(wm_w * scale)


    resized_wm = cv2.resize(watermark_rgba, (new_wm_w, new_wm_h), interpolation=cv2.INTER_AREA)
    
    # Split the resized watermark into BGR (for blending) and Alpha (mask)
    b, g, r, alpha = cv2.split(cv2.cvtColor(resized_wm, cv2.COLOR_RGBA2BGRA))
    watermark_bgr = cv2.merge((b, g, r))
    
    # Convert alpha channel to a float for blending (0.0 to 1.0)
    alpha_float = alpha.astype(float) / 255.0
    
    # 3. Calculate Position (Top-Left Corner)
    
    # Default is Bottom-Right with a 20-pixel margin
    margin = 20
    if position == "Top-Left":
        x = margin
        y = margin
    elif position == "Center":
        x = (doc_w - new_wm_w) // 2
        y = (doc_h - new_wm_h) // 2
    else: # Bottom-Right (default)
        x = doc_w - new_wm_w - margin
        y = doc_h - new_wm_h - margin

    # Ensure coordinates are within bounds (should be if margins are used)
    x = max(0, x)
    y = max(0, y)
    
    # Define the Region of Interest (ROI) on the document
    roi = doc_img[y:y + new_wm_h, x:x + new_wm_w]
    
    # 4. Blend the watermark onto the ROI
    for c in range(0, 3): # Iterate over B, G, R channels
        # Formula: ROI_new = (alpha * Watermark_color) + ( (1 - alpha) * ROI_old )
        roi[:, :, c] = (alpha_float * watermark_bgr[:, :, c]) + \
                       ((1.0 - alpha_float) * roi[:, :, c])
                       
    # Replace the ROI in the document image with the blended result
    doc_img[y:y + new_wm_h, x:x + new_wm_w] = roi

    # Convert the final BGR document image to RGB for Streamlit display
    return cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB)

# --- Streamlit UI ---

st.title("Digital E-Signature & Watermark Tool")
st.markdown("---")

# Initialize session state for the signature if it doesn't exist
if 'processed_signature' not in st.session_state:
    st.session_state['processed_signature'] = None
    
# --- 1. E-Signature Converter Section ---
st.header("1. E-Signature Converter")
st.markdown("Upload your handwritten signature (on white paper) to generate a blue, transparent digital file.")

sig_file = st.file_uploader("Upload Signature Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="signature_uploader")

if sig_file is not None:
    # Process and store the signature in session state
    st.session_state['processed_signature'] = process_signature(sig_file)

    if st.session_state['processed_signature'] is not None:
        st.subheader("Processed E-Signature Preview")
        
        # Prepare image for display/download
        proc_sig = st.session_state['processed_signature']
        
        # Convert the processed NumPy array (RGBA) back to a PNG byte stream
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(proc_sig, cv2.COLOR_RGBA2BGRA))
        io_buf = io.BytesIO(buffer)
        
        # Display the processed image on a light gray background to show transparency
        st.markdown(
            """
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; display: inline-block;">
            """,
            unsafe_allow_html=True
        )
        st.image(io_buf, caption="Processed Transparent Signature", use_column_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download Button
        st.download_button(
            label="Download Transparent Signature (PNG)",
            data=io_buf.getvalue(),
            file_name="e_signature_transparent.png",
            mime="image/png"
        )
        st.success("Signature processed successfully! You can now apply it as a watermark.")
    else:
        st.error("Could not process the signature file. Please ensure it's a valid image.")

st.markdown("---")

# --- 2. Watermark Tool Section ---
st.header("2. Watermark Tool")
st.markdown("Apply the generated e-signature onto a document or image.")

if st.session_state['processed_signature'] is not None:
    doc_file = st.file_uploader("Upload Document/Image to Watermark (JPG/PNG)", type=["jpg", "jpeg", "png"], key="document_uploader")

    if doc_file is not None:
        
        st.subheader("Watermark Settings")
        
        # Watermark position control
        position = st.selectbox(
            "Select Watermark Position:",
            ("Bottom-Right (Default)", "Center", "Top-Left")
        )
        
        # Watermark size control (as a percentage of document width)
        scale_factor = st.slider(
            "Signature Size (% of Document Width):",
            min_value=10, max_value=50, value=25, step=5
        )
        
        if st.button("Apply Watermark"):
            with st.spinner('Applying watermark...'):
                # Apply the watermark
                watermarked_img_rgb = apply_watermark(
                    doc_file, 
                    st.session_state['processed_signature'], 
                    scale_factor, 
                    position
                )
                
            if watermarked_img_rgb is not None:
                st.subheader("Watermarked Document")
                
                # Convert watermarked image back to bytes for download
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(watermarked_img_rgb, cv2.COLOR_RGB2BGR))
                io_buf = io.BytesIO(buffer)
                
                st.image(watermarked_img_rgb, caption="Document with Watermark", use_column_width=True)
                
                # Download Button
                st.download_button(
                    label="Download Watermarked Document (PNG)",
                    data=io_buf.getvalue(),
                    file_name="watermarked_document.png",
                    mime="image/png"
                )
            else:
                st.error("Could not process the document file.")
    
else:
    st.warning("Please upload and process a signature in Section 1 before using the Watermark Tool.")