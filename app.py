import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time

from transformers import DonutProcessor, VisionEncoderDecoderModel

from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_soft",
    [
        "#0081a7",  # Cerulean
        "#00afb9",  # Tropical Teal
        "#fdfcdc",  # Light Yellow (centro)
        "#fed9b7",  # Soft Apricot
        "#f07167",  # Vibrant Coral
    ]
)

custom_plotly_scale = [
    [0.0, "#0081a7"],   # Cerulean
    [0.25, "#00afb9"],  # Tropical Teal
    [0.5, "#fdfcdc"],   # Light Yellow
    [0.75, "#fed9b7"],  # Soft Apricot
    [1.0, "#f07167"],   # Vibrant Coral
]

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Donut AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# CSS PRO
# =========================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #e07a5f;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}
h2, h3 {
    color: #3d405b;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL (OPTIMIZADO)
# =========================
@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa"
    )
    model.eval()
    return processor, model

processor, model = load_model()
device = "cpu"
model.to(device)

# =========================
# SESSION STATE
# =========================
if "image" not in st.session_state:
    st.session_state.image = None

if "result" not in st.session_state:
    st.session_state.result = None
    
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    selected = option_menu(
        menu_title="🚀 Donut AI",
        options=[
            "Dashboard",
            "Arquitectura",
            "Embeddings",
            "Patches",
            "Attention",
            "QKV",
            "Resultados",
            "Limitaciones"
        ],
        icons=[
            "house","grid","cpu","eye","layers","type","gear"
        ],
        menu_icon="cpu",
        default_index=0,
        styles={
            "container": {"background-color": "#e07a5f"},
            "icon": {"color": "#e5e7eb"},
            "menu-title": {"color": "#e5e7eb"},
            "nav-link": {"color": "#e5e7eb", "font-size": "18px"},
            "nav-link-selected": {"background-color": "#3d405b"},
        }
    )

# =========================
# HEADER
# =========================
col_title, col_help = st.columns([0.25, 0.75])

with col_title:
    st.markdown("## 📊 Donut Transformer Dashboard")

with col_help:
    with st.popover("❓"):
        st.markdown("""
### 🧭 Cómo usar la app
1. 📤 Sube una imagen  
2. ❓ Escribe una pregunta  
3. 🚀 Ejecuta el modelo  
4. 🔍 Explora el análisis en el menú lateral
""")

st.caption("Análisis visual del modelo Encoder–Decoder")

# =========================
# FUNCIONES
# =========================
def draw_grid(image, size):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(0, w, size):
        for j in range(0, h, size):
            draw.rectangle([i, j, i+size, j+size], outline="red", width=1)
    return img

# =========================
# DASHBOARD 
# =========================
if selected == "Dashboard":

    col_info, col_link = st.columns([0.17, 0.83])
    with col_info:
        st.markdown("### 📌 Información del modelo")
    with col_link:
        st.link_button("📄", "https://arxiv.org/pdf/1910.13461")

    col1, col2, col3, col4 = st.columns(4)

    # =========================
    # 📊 MÉTRICAS 
    # =========================
    if st.session_state.history:
        last = st.session_state.history[-1]
        data = [
            ("⚡ Tiempo", f"{last['latency']} s"),
            ("🧠 Tokens", last["tokens"]),
            ("📄 Longitud", last["chars"]),
            ("🚀 Estado", "Procesado")
        ]
    else:
        data = [
            ("⚡ Tiempo", "--"),
            ("🧠 Tokens", "--"),
            ("📄 Longitud", "--"),
            ("🚀 Estado", "Esperando")
        ]

    for col, (title, value) in zip([col1, col2, col3, col4], data):
        with col:
            with st.container(border=True):
                st.markdown(f"<span style='color:gray'>{title}</span>", unsafe_allow_html=True)
                st.markdown(f"## {value}")

    st.markdown("### 🧾 Entrada del modelo")

    col1, col2 = st.columns([0.65, 1.75])

    # =========================
    # 🧠 IZQUIERDA 
    # =========================
    with col1:
        with st.container(border=True):

            uploaded_file = st.file_uploader("Subir imagen", type=["jpg","png","jpeg"])
            question = st.text_input("Pregunta", placeholder="What is the total?")
            run = st.button("🚀 Ejecutar", use_container_width=True)

            # 🗑 limpiar historial
            if st.button("🗑 Limpiar historial"):
                st.session_state.history = []
                st.rerun()

            # =========================
            # 💬 HISTORIAL
            # =========================
            if st.session_state.history:
                st.markdown("---")
                st.markdown("### 💬 Historial")

                for item in reversed(st.session_state.history):
                    st.markdown(f"**❓ {item['question']}**")
                    st.markdown(f"**✅ {item['answer']}**")
                    st.caption(f"⏱ {item['latency']} s")
                    st.markdown("---")

    # =========================
    # 🖼️ DERECHA (IMAGEN)
    # =========================
    with col2:
        with st.container(border=True):

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.image = image.copy()
                st.image(image, use_container_width=True)
            else:
                st.info("Vista previa")

    # =========================
    # 🚀 INFERENCIA
    # =========================
    if run and st.session_state.image and question:

        start_time = time.time()

        image = st.session_state.image
        image_resized = image.resize((384,384))

        pixel_values = processor(image_resized, return_tensors="pt").pixel_values.to(device)

        # 🔥 prompt correcto
        task_prompt = f"<s_docvqa><s_question>{question}</s_question>"
        decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            gen = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512
            )

        end_time = time.time()

        # 🔥 decodificación correcta
        decoded = processor.batch_decode(gen, skip_special_tokens=False)[0]

        import re
        text = re.sub(r"<.*?>", "", decoded).strip()

        latency = round(end_time - start_time, 2)

        # =========================
        # 💾 GUARDAR RESULTADO ACTUAL 
        # =========================
        st.session_state.result = {
            "text": text,
            "pixel_values": pixel_values.cpu(),   # 🔥 NECESARIO
            "tokens": len(gen[0]),
            "chars": len(text),
            "latency": latency
        }

        # =========================
        # 💬 HISTORIAL
        # =========================
        st.session_state.history.append({
            "question": question,
            "answer": text,
            "latency": latency,
            "tokens": len(gen[0]),
            "chars": len(text)
        })

        # 🔄 refrescar UI
        st.rerun()
        
# =========================
# ARQUITECTURA
# =========================
elif selected == "Arquitectura":

    st.markdown("## 🧠 Arquitectura Donut")

    st.info("""
Donut es un modelo Transformer encoder–decoder:

• Encoder (Swin Transformer) → procesa la imagen  
• Decoder (BART) → genera texto estructurado  
""")

    st.markdown("### 🔄 Flujo del modelo")

    st.code("""
Imagen (input)
↓
Patches (16x16)
↓
Embeddings (vector 1024)
↓
Encoder (Transformer)
↓
Decoder (BART)
↓
Texto / JSON (output)
""")

    st.markdown("### 📥 Entrada")

    st.write("""
• Imagen de documento  
• Prompt (pregunta)
""")

    st.markdown("### 📤 Salida")

    st.write("""
• Texto generado  
• Puede convertirse en JSON estructurado
""")

    st.success("""
🚀 Innovación clave:
Donut NO usa OCR → es end-to-end
""")
    
    
# =========================
# EMBEDDINGS
# =========================
elif selected == "Embeddings" and st.session_state.result:

    st.markdown("## 🧠 Embeddings del encoder")

    with torch.no_grad():
        outputs = model.encoder(
            st.session_state.result["pixel_values"]
        )

    embeddings = outputs.last_hidden_state[0]

    st.write(f"Shape embeddings: {embeddings.shape}")

    idx = st.slider("Seleccionar token", 0, embeddings.shape[0]-1, 0)

    vec = embeddings[idx].cpu().numpy()

    st.markdown("### 🔢 Vector (primeros valores)")
    st.code(vec[:20])

    st.markdown("### 🎨 Visualización 2D")

    vec_2d = vec.reshape(32, 32)

    fig, ax = plt.subplots()
    im = ax.imshow(vec_2d, cmap=custom_cmap)
    ax.axis("off")

    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

    st.info("""
Cada token de imagen se convierte en un vector de alta dimensión (embedding).
Este vector contiene información visual (texto, bordes, estructura).
""")
    
# =========================
# PATCHES 
# =========================
elif selected == "Patches" and st.session_state.image:

    image = st.session_state.image
    col_left, col_right = st.columns(2)

    # =========================
    # FUNCIÓN PARA RESALTAR PATCH
    # =========================
    def draw_grid_highlight(image, size, highlight_idx=None):
        img = image.copy()
        draw = ImageDraw.Draw(img)

        grid_size = image.size[0] // size
        idx_counter = 0

        for i in range(grid_size):
            for j in range(grid_size):

                x = j * size
                y = i * size

                if idx_counter == highlight_idx:
                    draw.rectangle(
                        [x, y, x+size, y+size],
                        outline="yellow",
                        width=4
                    )
                else:
                    draw.rectangle(
                        [x, y, x+size, y+size],
                        outline="red",
                        width=1
                    )

                idx_counter += 1

        return img

    # =========================
    # 🧠 IZQUIERDA (OPTIMIZADA + COMPLETA)
    # =========================
    with col_left:
        with st.container(border=True):

            st.markdown("### 🧩 Tokens visuales del modelo")

            st.info("""
    El modelo Donut divide la imagen en:

    • Tamaño imagen: 384 × 384  
    • Patch interno: 16 × 16  
    • Total: 24 × 24 = **576 tokens visuales**
    """)

            # 🔥 PARÁMETROS
            patch_size = 16
            grid_size = 384 // patch_size

            img = np.array(image.resize((384,384)))

            patches = []
            coords = []

            for i in range(grid_size):
                for j in range(grid_size):
                    y = i * patch_size
                    x = j * patch_size
                    patches.append(img[y:y+patch_size, x:x+patch_size])
                    coords.append((y, x))

            # =========================
            # 🔥 FILA 1: MÉTRICAS SUPERIORES
            # =========================
            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Tokens", len(patches))

            with m2:
                st.metric("Dimensión", "1024")

            with m3:
                st.metric("Grid", "24×24")

            # =========================
            # 🎯 FILA 2: SELECTOR
            # =========================
            idx = st.slider("Seleccionar token", 0, 575, 0)

            # =========================
            # 🔥 FILA 3: PATCH + INFO + IMPORTANCIA
            # =========================
            c1, c2, c3 = st.columns([1, 1.2, 1])

            with c1:
                st.image(patches[idx], width=90)

            with c2:
                st.markdown(f"**Token:** {idx}")
                st.markdown(f"**Posición:** {coords[idx]}")

            # =========================
            # 🧠 EMBEDDING
            # =========================
            if st.session_state.result:

                with torch.no_grad():
                    emb = model.encoder(
                        st.session_state.result["pixel_values"]
                    ).last_hidden_state

                patch_vector = emb[0][idx + 1].cpu().numpy()

                importance = np.linalg.norm(patch_vector)

                with c3:
                    st.metric("Importancia", round(importance, 2))

                # =========================
                # 🔢 EMBEDDING TEXTO
                # =========================
                st.markdown("#### 🧠 Embedding del token")

                st.markdown("**Primeros valores:**")
                st.code(patch_vector[:20])

                # =========================
                # 🔥 ESPACIO LATENTE
                # =========================
                st.markdown("#### 🎨 Espacio latente")

                patch_img = patches[idx]

                emb_2d = patch_vector.reshape(32,32)

                # 🔥 normalización (CLAVE)
                emb_norm = (emb_2d - emb_2d.min()) / (emb_2d.max() - emb_2d.min())

                fig, ax = plt.subplots()

                im = ax.imshow(
                    emb_norm,
                    cmap=custom_cmap
                )

                ax.axis("off")

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Intensidad de activación")

                st.pyplot(fig)

                st.caption("""
                🔵 Baja activación → 🟡 Media → 🔴 Alta activación  

                Distribución interna del embedding (espacio latente)
                """)

                # =========================
                # 🔥 OVERLAY SOBRE EL PATCH
                # =========================
                st.markdown("#### 🔍 Embedding sobre el patch")

                emb_resized = Image.fromarray(
                    (emb_norm * 255).astype(np.uint8)
                ).resize((patch_size, patch_size))

                col_o1, col_o2 = st.columns([1,1])

                with col_o1:
                    alpha = st.slider("Transparencia", 0.1, 1.0, 0.5)

                fig2, ax2 = plt.subplots()

                ax2.imshow(patch_img)

                overlay = ax2.imshow(
                    emb_resized,
                    cmap=custom_cmap,
                    alpha=alpha
                )

                ax2.axis("off")

                cbar2 = plt.colorbar(overlay, ax=ax2)
                cbar2.set_label("Intensidad de activación")

                st.pyplot(fig2)

                st.caption("""
                🔵 Baja activación → 🟡 Media → 🔴 Alta activación  

                El color muestra qué partes del patch activan más al modelo
                """)

                # =========================
                # 🧠 EXPLICACIÓN FINAL
                # =========================
                st.info(f"""
                🔍 Interpretación:

                • Token {idx} representa una región 16×16 de la imagen  
                • Se transforma en un vector de 1024 dimensiones  
                • Este vector codifica patrones visuales (texto, bordes, estructura)  

                ➡️ Este embedding será usado en el mecanismo de atención (Q, K, V)
                """)

            else:
                st.warning("Primero ejecuta el modelo en el Dashboard")

            # =========================
            # RESUMEN FINAL
            # =========================
            st.markdown(f"""
    **Token {idx} → Patch → Embedding (1024 dimensiones)**
    """)
        
    # =========================
    # 🖼️ DERECHA
    # =========================
    with col_right:
        with st.container(border=True):

            st.markdown("#### 🖼️ Grid real del modelo (24 × 24)")

            grid_img = draw_grid_highlight(
                image.resize((384,384)),
                patch_size,
                highlight_idx=idx
            )

            st.image(grid_img, use_container_width=True)

            st.caption(f"Token {idx} resaltado en amarillo")


# =========================
# ATTENTION 
# =========================
elif selected == "Attention" and st.session_state.result:

    st.markdown("### 🔥 Attention del modelo")

    with st.container(border=True):

        with torch.no_grad():
            outputs = model.encoder(
                st.session_state.result["pixel_values"],
                output_attentions=True
            )

        att = outputs.attentions

        layer = st.slider("Capa", 0, len(att)-1, len(att)-1)

        att_layer = att[layer][0]  # (heads, tokens, tokens)

        # =========================
        # 🔥 MATRIZ COMPLETA (TOKEN vs TOKEN)
        # =========================
        att_matrix = att_layer.mean(0).cpu().numpy()

        st.markdown("#### 🧠 Matriz de atención completa")

        fig_full, ax_full = plt.subplots()
        im_full = ax_full.imshow(att_matrix, cmap="inferno")
        ax_full.set_title("Relación entre TODOS los tokens")
        ax_full.axis("off")

        plt.colorbar(im_full, ax=ax_full)
        st.pyplot(fig_full)

        st.caption("""
Cada fila = qué mira un token  
Cada columna = a qué token presta atención  

👉 Esto es el corazón del Transformer
""")

        # =========================
        # 🔥 CLS → PATCHES 
        # =========================
        att_mean = att_layer.mean(0)

        cls_att = att_mean[0, 1:].cpu().numpy()

        # normalización segura
        cls_att = (cls_att - cls_att.min()) / (cls_att.max() - cls_att.min() + 1e-8)

        num_tokens = len(cls_att)

        # =========================
        # 🔥 GRID DINÁMICO (FIX ERROR)
        # =========================
        grid_h = int(np.floor(np.sqrt(num_tokens)))
        grid_w = int(np.ceil(num_tokens / grid_h))

        pad_size = grid_h * grid_w - num_tokens

        if pad_size > 0:
            cls_att = np.pad(cls_att, (0, pad_size))

        att_map = cls_att.reshape(grid_h, grid_w)

        st.info(f"🔢 Tokens: {num_tokens} | Grid: {grid_h}×{grid_w}")

        # =========================
        # 🖼️ VISUALIZACIÓN CLARA
        # =========================
        image = st.session_state.image.resize((384,384))

        col1, col2 = st.columns(2)

        # 👉 IMAGEN ORIGINAL
        with col1:
            st.markdown("#### 🖼️ Imagen original")
            st.image(image, use_container_width=True)

        # 👉 ATENCIÓN SOBRE IMAGEN
        with col2:
            st.markdown("#### 🔥 Atención del modelo")

            alpha = st.slider("Transparencia", 0.1, 1.0, 0.5)

            fig, ax = plt.subplots()

            ax.imshow(image)

            heatmap = ax.imshow(
                att_map,
                cmap="inferno",
                alpha=alpha,
                extent=(0, image.size[0], image.size[1], 0)
            )

            ax.axis("off")

            plt.colorbar(heatmap, ax=ax)

            st.pyplot(fig)

        # =========================
        # 📊 DATAFRAME
        # =========================
        df_att = pd.DataFrame({
            "Token": np.arange(num_tokens),
            "Atención": cls_att[:num_tokens]  # evitar padding extra
        }).sort_values(by="Atención", ascending=False)

        # =========================
        # 📋 TABLA + 📈 GRÁFICA
        # =========================
        col_tabla, col_graf = st.columns([0.3, 0.7])

        with col_tabla:
            with st.container(border=True):
                st.markdown("#### 📋 Top tokens")
                st.dataframe(df_att.head(15), use_container_width=True)

        with col_graf:
            with st.container(border=True):
                st.markdown("#### 📈 Ranking de atención")

                fig_bar = px.bar(
                    df_att.head(15),
                    x="Token",
                    y="Atención",
                    color="Atención",
                    color_continuous_scale=custom_plotly_scale
                )

                st.plotly_chart(fig_bar, use_container_width=True)

        # =========================
        # 🧠 INTERPRETACIÓN MEJORADA
        # =========================
        st.info("""
🔍 Interpretación:

• La matriz muestra cómo TODOS los tokens interactúan entre sí  
• CLS resume la información global de la imagen  

🔥 En la visualización:

• Zonas ROJAS → el modelo se enfoca más  
• Zonas OSCURAS → menos importantes  

👉 Esto permite ver EXACTAMENTE dónde está "mirando" el modelo

💡 Es la evidencia visual del mecanismo de atención
""")

# =========================
# 🔥 QKV 
# =========================
elif selected == "QKV" and st.session_state.result:

    st.markdown("### 🔥 QKV del modelo")

    st.info("""
🧠 ¿Cómo se generan Q, K y V?

Q = X · Wq  
K = X · Wk  
V = X · Wv  

• X = embedding del token  
• Wq, Wk, Wv = matrices aprendidas  

⚠️ Aquí usamos simulación
""")

    with st.container(border=True):

        # ===== EMBEDDINGS =====
        with torch.no_grad():
            encoder_outputs = model.encoder(
                st.session_state.result["pixel_values"]
            )

        hidden_states = encoder_outputs.last_hidden_state
        tokens = hidden_states[0, 1:, :]

        num_total_tokens = tokens.shape[0]

        # =========================
        # 🎯 TOKEN PRINCIPAL
        # =========================
        idx = st.slider("Seleccionar token", 0, num_total_tokens - 1, 0)
        token_vec = tokens[idx].cpu().numpy()

        st.markdown(f"### 🎯 Token seleccionado: {idx}")

        # =========================
        # 🔥 QKV
        # =========================
        dim = token_vec.shape[0]

        np.random.seed(42)
        Wq = np.random.randn(dim, dim)
        Wk = np.random.randn(dim, dim)
        Wv = np.random.randn(dim, dim)

        Q = token_vec @ Wq
        K = token_vec @ Wk
        V = token_vec @ Wv

        # =========================
        # 📊 MÉTRICAS
        # =========================
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Q norm", round(np.linalg.norm(Q), 2))
        with m2:
            st.metric("K norm", round(np.linalg.norm(K), 2))
        with m3:
            st.metric("V norm", round(np.linalg.norm(V), 2))

        # =========================
        # 🎨 VISUAL
        # =========================
        st.markdown("#### 🎨 Q, K, V")

        col1, col2, col3 = st.columns(3)

        def plot_vector(vec, title):
            vec_2d = vec.reshape(32, 32)
            fig, ax = plt.subplots()
            im = ax.imshow(vec_2d, cmap=custom_cmap)
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            return fig

        with col1:
            st.pyplot(plot_vector(Q, "Q"))

        with col2:
            st.pyplot(plot_vector(K, "K"))

        with col3:
            st.pyplot(plot_vector(V, "V"))

    # =========================
    # 🔥 ATENCIÓN DESDE Q
    # =========================
    all_tokens = tokens.cpu().numpy()
    Q = Q.reshape(-1)

    scores = all_tokens @ Q
    scores = scores / np.sqrt(dim)

    exp_scores = np.exp(scores - np.max(scores))
    att_weights = exp_scores / exp_scores.sum()

    # =========================
    # 📊 DF ATENCIÓN
    # =========================
    df_att = pd.DataFrame({
        "Token": np.arange(num_total_tokens),
        "Atención": att_weights
    }).sort_values(by="Atención", ascending=False)

    # =========================
    # 🔥 COMPARACIÓN
    # =========================
    st.markdown("## 📊 Comparar tokens")

    col_ctrl1, col_ctrl2 = st.columns(2)

    with col_ctrl1:
        modo = st.selectbox(
            "Modo de selección",
            ["Aleatorio", "Más importantes", "Mixto"]
        )

    with col_ctrl2:
        num_tokens = st.slider("Número de tokens", 2, 20, 5)

    # IMPORTANCIA
    norms = []
    for t in range(num_total_tokens):
        vec = tokens[t].cpu().numpy()
        q = vec @ Wq
        norms.append((t, np.linalg.norm(q)))

    df_all = pd.DataFrame(norms, columns=["Token", "Importancia"])
    df_all = df_all.sort_values(by="Importancia", ascending=False)

    # SELECCIÓN
    if modo == "Aleatorio":
        selected_tokens = np.random.choice(num_total_tokens, num_tokens, replace=False)

    elif modo == "Más importantes":
        selected_tokens = df_all.head(num_tokens)["Token"].values

    else:
        top = df_all.head(num_tokens // 2)["Token"].values
        rand = np.random.choice(num_total_tokens, num_tokens - len(top), replace=False)
        selected_tokens = np.concatenate([top, rand])

    df = df_all[df_all["Token"].isin(selected_tokens)]

    # =========================
    # 📋 TABLA + 📈 GRÁFICA
    # =========================
    col_tabla, col_graf = st.columns([0.3, 0.7])

    with col_tabla:
        with st.container(border=True):
            st.markdown("#### 📋 Tokens")
            st.dataframe(df, use_container_width=True)

    with col_graf:
        with st.container(border=True):
            st.markdown("#### 📈 Ranking")

            fig_bar = px.bar(
                df,
                x="Token",
                y="Importancia",
                color="Importancia",
                color_continuous_scale=custom_plotly_scale
            )

            st.plotly_chart(fig_bar, use_container_width=True)
 
# =========================
# 🔥 QKV (PRO + TEORÍA + ATTENTION REAL)
# =========================
elif selected == "QKV" and st.session_state.result:

    st.markdown("### 🔥 QKV del modelo")

    # =========================
    # 🧠 EXPLICACIÓN TEÓRICA (CLAVE PARA SUSTENTACIÓN)
    # =========================
    st.info("""
🧠 ¿Cómo se generan Q, K y V?

En un Transformer:

Q = X · Wq  
K = X · Wk  
V = X · Wv  

Donde:

• X = embedding del token  
• Wq, Wk, Wv = matrices aprendidas  

⚠️ En este modelo (Donut), estas matrices no son accesibles directamente,
por lo que usamos una simulación para visualizar el comportamiento.
""")

    with st.container(border=True):

        # ===== EMBEDDINGS =====
        with torch.no_grad():
            encoder_outputs = model.encoder(
                st.session_state.result["pixel_values"]
            )

        hidden_states = encoder_outputs.last_hidden_state
        tokens = hidden_states[0, 1:, :]  # quitar CLS

        # =========================
        # 🎯 TOKEN PRINCIPAL
        # =========================
        idx = st.slider("Seleccionar token", 0, 575, 0)
        token_vec = tokens[idx].cpu().numpy()

        st.markdown(f"### 🎯 Token seleccionado: {idx}")

        # =========================
        # 🔥 QKV 
        # =========================
        dim = token_vec.shape[0]

        np.random.seed(42)
        Wq = np.random.randn(dim, dim)
        Wk = np.random.randn(dim, dim)
        Wv = np.random.randn(dim, dim)

        Q = token_vec @ Wq
        K = token_vec @ Wk
        V = token_vec @ Wv

        # =========================
        # 📊 MÉTRICAS
        # =========================
        q_norm = np.linalg.norm(Q)
        k_norm = np.linalg.norm(K)
        v_norm = np.linalg.norm(V)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Q norm", round(q_norm, 2))
        with m2:
            st.metric("K norm", round(k_norm, 2))
        with m3:
            st.metric("V norm", round(v_norm, 2))

        st.caption(f"""
🔍 Interpretación:

• Norm alto → mayor influencia del token  
• Norm bajo → menor impacto  

👉 Q={q_norm:.1f}, K={k_norm:.1f}, V={v_norm:.1f}
""")

        # =========================
        # 🎨 VISUAL QKV
        # =========================
        st.markdown("#### 🎨 Q, K, V (espacio latente)")

        col1, col2, col3 = st.columns(3)

        def plot_vector(vec, title):
            vec_2d = vec.reshape(32, 32)
            fig, ax = plt.subplots()
            im = ax.imshow(vec_2d, cmap=custom_cmap)
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            return fig

        with col1:
            st.pyplot(plot_vector(Q, "Query (Q)"))

        with col2:
            st.pyplot(plot_vector(K, "Key (K)"))

        with col3:
            st.pyplot(plot_vector(V, "Value (V)"))

    # =========================
    # 🔥 ATENCIÓN 
    # =========================

    all_tokens = tokens.cpu().numpy()
    Q = Q.reshape(-1)

    num_total_tokens = all_tokens.shape[0]

    # 👉 cálculo atención
    scores = all_tokens @ Q
    scores = scores / np.sqrt(dim)

    # 👉 softmax estable
    exp_scores = np.exp(scores - np.max(scores))
    att_weights = exp_scores / exp_scores.sum()

    att_weights = att_weights.reshape(-1)

    if len(att_weights) != num_total_tokens:
        st.error("Error: dimensiones inconsistentes en attention")
        st.stop()

    # =========================
    # 📊 DATAFRAME
    # =========================
    df_att = pd.DataFrame({
        "Token": np.arange(num_total_tokens),
        "Atención": att_weights
    }).sort_values(by="Atención", ascending=False)

    # =========================
    # 🔥 COMPARACIÓN DE TOKENS
    # =========================
    st.markdown("## 📊 Comparar tokens")

    col_ctrl1, col_ctrl2 = st.columns(2)

    with col_ctrl1:
        modo = st.selectbox(
            "Modo de selección",
            ["Aleatorio", "Más importantes", "Mixto"]
        )

    with col_ctrl2:
        num_tokens = st.slider("Número de tokens", 2, 20, 5)

    # IMPORTANCIA GLOBAL
    num_total_tokens = tokens.shape[0]

    norms = []
    for t in range(num_total_tokens):
        vec = tokens[t].cpu().numpy()
        q = vec @ Wq
        norms.append((t, np.linalg.norm(q)))

    df_all = pd.DataFrame(norms, columns=["Token", "Importancia"])
    df_all = df_all.sort_values(by="Importancia", ascending=False)

    if modo == "Aleatorio":
        selected_tokens = np.random.choice(num_total_tokens, num_tokens, replace=False)

    elif modo == "Más importantes":
        selected_tokens = df_all.head(num_tokens)["Token"].values

    else:
        top = df_all.head(num_tokens // 2)["Token"].values
        rand = np.random.choice(num_total_tokens, num_tokens - len(top), replace=False)
        selected_tokens = np.concatenate([top, rand])
    
    df = df_all[df_all["Token"].isin(selected_tokens)]

    col_tabla, col_graf = st.columns([0.3, 0.7])

    with col_tabla:
        with st.container(border=True):
            st.markdown("#### 📋 Tokens")
            st.dataframe(df, use_container_width=True)

    with col_graf:
        with st.container(border=True):
            st.markdown("#### 📈 Ranking")

            fig_bar = px.bar(
                df,
                x="Token",
                y="Importancia",
                color="Importancia",
                color_continuous_scale=custom_plotly_scale
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        
# =========================
# RESULTADOS (NIVEL PAPER)
# =========================
elif selected == "Resultados" and st.session_state.history:

    from rapidfuzz.distance import Levenshtein

    def anls(pred, gt):
        pred = pred.lower().strip()
        gt = gt.lower().strip()
        dist = Levenshtein.distance(pred, gt)
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            return 1.0
        return max(1 - dist / max_len, 0)

    def exact_match(pred, gt):
        return int(pred.strip().lower() == gt.strip().lower())

    def f1_score(pred, gt):
        pred_tokens = pred.lower().split()
        gt_tokens = gt.lower().split()
        common = set(pred_tokens) & set(gt_tokens)
        if len(common) == 0:
            return 0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    st.markdown("## 📊 Resultados del modelo")

    df = pd.DataFrame(st.session_state.history)
    last = st.session_state.history[-1]

    # =========================
    # INPUT GROUND TRUTH
    # =========================
    gt = st.text_input("Ground Truth (respuesta real)")

    # =========================
    # FILA 1
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 🧾 Resultado")
            st.image(st.session_state.image)

            st.markdown(f"""
**Pregunta:** {last['question']}  
**Predicción:** {last['answer']}
""")

    with col2:
        with st.container(border=True):
            st.markdown("### 🔥 Atención")

            with torch.no_grad():
                outputs = model.encoder(
                    st.session_state.result["pixel_values"],
                    output_attentions=True
                )

            att = outputs.attentions
            att_layer = att[-1][0]
            att_mean = att_layer.mean(0)

            cls_att = att_mean[0, 1:].cpu().numpy()
            cls_att = (cls_att - cls_att.min()) / (cls_att.max() - cls_att.min() + 1e-8)

            size = int(np.ceil(np.sqrt(len(cls_att))))
            padded = np.zeros(size * size)
            padded[:len(cls_att)] = cls_att
            att_map = padded.reshape(size, size)

            image = st.session_state.image.resize((384,384))

            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.imshow(att_map, cmap="inferno", alpha=0.5, extent=(0, 384, 384, 0))
            ax.axis("off")

            st.pyplot(fig)

    # =========================
    # MÉTRICAS
    # =========================
    if gt:

        pred = last["answer"]

        em = exact_match(pred, gt)
        f1 = f1_score(pred, gt)
        score_anls = anls(pred, gt)

        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric("Exact Match", em)

        with col4:
            st.metric("F1 Score", f"{f1:.2f}")

        with col5:
            st.metric("ANLS", f"{score_anls:.2f}")

        # =========================
        # GRÁFICA MÉTRICAS
        # =========================
        df_metrics = pd.DataFrame({
            "Métrica": ["ANLS", "F1", "Exact Match"],
            "Valor": [score_anls, f1, em]
        })

        st.plotly_chart(
            px.bar(df_metrics, x="Métrica", y="Valor", color="Valor"),
            use_container_width=True
        )

    # =========================
    # FILA 2: GRÁFICAS AVANZADAS
    # =========================
    col6, col7, col8 = st.columns(3)

    with col6:
        with st.container(border=True):
            st.markdown("### ⚡ Tokens vs Tiempo")

            fig1 = px.scatter(
                df,
                x="tokens",
                y="latency",
                size="chars"
            )
            st.plotly_chart(fig1, use_container_width=True)

    with col7:
        with st.container(border=True):
            st.markdown("### 📏 Longitud vs Tokens")

            fig2 = px.line(
                df,
                x="tokens",
                y="chars",
                markers=True
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col8:
        with st.container(border=True):
            st.markdown("### ⏱ Tiempo por consulta")

            fig3 = px.bar(
                df,
                y="latency"
            )
            st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # HISTORIAL COMPLETO
    # =========================
    st.markdown("### 📋 Historial completo")
    st.dataframe(df, use_container_width=True)
    
# =========================
# ENTRADA - SALIDA
# =========================
elif selected == "Entrada/Salida":
    st.markdown("### 🔄 Flujo del modelo")

    st.code("""
    Input:
    Imagen + Prompt (ej: <s_docvqa>)

    ↓

    Encoder (Swin Transformer)

    ↓

    Embeddings visuales

    ↓

    Decoder (BART)

    ↓

    Tokens de salida

    ↓

    Texto / JSON
    """)

# =========================
# LIMITACIONES
# =========================
elif selected == "Limitaciones":

    st.markdown("## ⚠️ Limitaciones del modelo")

    st.write("""
• Sensible a resolución de imagen  
• Puede fallar en texto pequeño  
• No usa OCR → puede perder precisión en caracteres  
• Depende de datos de entrenamiento  
""")

    st.info("""
Aunque Donut es potente, aún tiene desafíos en documentos complejos.
""")